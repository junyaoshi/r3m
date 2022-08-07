import argparse
from collections import OrderedDict
import time
from os.path import join
import os
import sys
from pprint import pprint
import pickle

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataset import (
    SomethingSomethingR3M,
    SomethingSomethingDemosR3M
)
from resnet import EndtoEndNet, TransferableNet
from bc_utils import (
    count_parameters_in_M, generate_single_visualization,
    pose_to_joint_depth, unnormalize_bbox,
    CV_TASKS, CLUSTER_TASKS,
    cv_task_to_cluster_task, cluster_task_to_cv_task,
    load_img_from_hand_info
)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Training BC network.')

    # eval
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Evaluate model on training set instead of validation set')
    parser.add_argument('--run_on_cv_server', action='store_true',
                        help='if true, run tasks on cv-server; else, run all tasks on cluster')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataset multiprocessing')
    parser.add_argument('--n_eval_samples', type=int, default=20,
                        help='number of samples for evaluation')
    parser.add_argument('--use_visualizer', action='store_true',
                        help='if true, use opengl visualizer to render results and show on tensorboard')
    parser.add_argument('--dataset', type=str, default='ss',
                        help='which dataset to use',
                        choices=['ss', 'hand_demos', 'robot_demos', 'same_hand_demos'])
    parser.add_argument('--eval_tasks', action='store_true',
                        help='if true, evaluate conditioning on different tasks')
    parser.add_argument('--eval_r3m', action='store_true',
                        help='if true, evaluate conditioning on different r3m (image) input')
    parser.add_argument('--eval_r3m_samples', type=int, default=5,
                        help='number of samples used for evaluating r3m conditioning')
    parser.add_argument('--log_depth', action='store_true',
                        help='if true, log depth evaluation in visualization')
    parser.add_argument('--log_depth_scatter_plots', action='store_true',
                        help='if true, log depth evaluation as scatter plots')
    parser.add_argument('--depth_descriptor', type=str, default='normalized_bbox_size',
                        help='which descriptor to use for hand depth estimation',
                        choices=['wrist_img_z', 'bbox_size', 'scaling_factor', 'normalized_bbox_size'])
    parser.add_argument('--use_current_frame_info', action='store_true',
                        help='if true, use current frame info (pose, bbox, or shape) instead of future frame '
                             'for visualization if the model is not trained to predict certain info')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='if true, dataloader is not shuffled')

    # paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='location of the checkpoint to load for evaluation')
    parser.add_argument('--root', type=str, default='/scratch/junyao/LfHV/r3m/eval_checkpoints',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='debug',
                        help='id used for storing intermediate results')
    parser.add_argument('--data_home_dir', type=str,
                        default='/home/junyao/Datasets/something_something_processed',
                        help='location of the data corpus')
    parser.add_argument('--frankmocap_path', type=str,
                        default='/home/junyao/LfHV/frankmocap',
                        help='location of frank mocap')
    parser.add_argument('--r3m_path', type=str,
                        default='/home/junyao/LfHV/r3m',
                        help='location of R3M')

    args = parser.parse_args()
    return args

def main(eval_args):
    demo_datasets = ['hand_demos', 'robot_demos', 'same_hand_demos']
    eval_args.eval_loss = eval_args.dataset != 'same_hand_demos' # run main eval pass only for other datasets
    eval_args.no_future_info = eval_args.dataset == 'same_hand_demos'  # this dataset doesn't have future info
    eval_args.robot_demos = eval_args.dataset == 'robot_demos'
    pprint(f'eval args: \n{eval_args}')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}.')
    eval_args.save = join(eval_args.root, eval_args.save)
    writer = SummaryWriter(log_dir=eval_args.save, flush_secs=60)

    # visualizer and frank mocap
    print('Loading frankmocap visualizer...')
    sys.path.insert(1, eval_args.frankmocap_path)
    os.chdir(eval_args.frankmocap_path)
    from renderer.visualizer import Visualizer
    from handmocap.hand_mocap_api import HandMocap
    visualizer = Visualizer('opengl') if eval_args.use_visualizer else None
    checkpoint_hand = join(
        eval_args.frankmocap_path,
        'extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
    )
    smpl_dir = join(eval_args.frankmocap_path, 'extra_data/smpl')
    hand_mocap_vis = HandMocap(checkpoint_hand, smpl_dir, device=device)
    hand_mocap_depth = None
    if eval_args.log_depth:
        hand_mocap_depth = HandMocap(checkpoint_hand, smpl_dir, device=device)
    os.chdir(eval_args.r3m_path)
    print('Visualizer loaded.')

    # load a checkpoint
    print('loading the model at:')
    print(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']
    pprint(f'loaded train args: \n{args}')
    eval_task_names = CV_TASKS if eval_args.run_on_cv_server else CLUSTER_TASKS
    if eval_args.dataset in demo_datasets:
        eval_task_names = CLUSTER_TASKS

    # compute dimensions
    r3m_dim, task_dim, cam_dim = 2048, len(CV_TASKS) if args.run_on_cv_server else len(CLUSTER_TASKS), 3
    args.hand_bbox_dim, args.hand_pose_dim, args.hand_shape_dim = 4, 48, 10
    hand_dim = sum([args.hand_bbox_dim, args.hand_pose_dim, args.hand_shape_dim])
    input_dim = sum([r3m_dim, task_dim, hand_dim, cam_dim, cam_dim])
    output_dim = hand_dim

    # load model
    model, model_init_func, residual = None, None, None
    if args.model_type == 'e2e':
        model_init_func = EndtoEndNet
    elif args.model_type == 'transfer':
        model_init_func = TransferableNet
    if args.net_type == 'mlp':
        residual = False
    elif args.net_type == 'residual':
        residual = True
    model = model_init_func(
        in_features=input_dim,
        out_features=output_dim,
        dims=(r3m_dim, task_dim, hand_dim),
        n_blocks=args.n_blocks,
        residual=residual
    ).to(device).float()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f'Loaded model type: {args.model_type}, blocks: {args.n_blocks}, network: {args.net_type}')
    print(f'param size = {count_parameters_in_M(model)}M')

    l2_loss_func = nn.MSELoss()
    l1_loss_func = nn.L1Loss()

    print('Creating data loader...')
    start = time.time()
    if eval_args.dataset == 'hand_demos':
        print('Using something-something hand demos dataset.')
        data = SomethingSomethingDemosR3M(
            eval_task_names, data_home_dir=eval_args.data_home_dir,
            time_interval=args.time_interval,
            debug=args.debug, run_on_cv_server=eval_args.run_on_cv_server,
            demo_type='hand', depth_descriptor=eval_args.depth_descriptor
        )
    elif eval_args.dataset == 'robot_demos':
        print('Using something-something robot demos dataset.')
        data = SomethingSomethingDemosR3M(
            eval_task_names, data_home_dir=eval_args.data_home_dir,
            time_interval=1,
            debug=args.debug, run_on_cv_server=eval_args.run_on_cv_server,
            demo_type='robot', depth_descriptor=eval_args.depth_descriptor
        )
    elif eval_args.dataset == 'same_hand_demos':
        print('Using something-something same hand demos dataset.')
        data = SomethingSomethingDemosR3M(
            eval_task_names, data_home_dir=eval_args.data_home_dir,
            time_interval=1,
            debug=args.debug, run_on_cv_server=eval_args.run_on_cv_server,
            demo_type='same_hand', depth_descriptor=eval_args.depth_descriptor
        )
    else:
        data = SomethingSomethingR3M(
            eval_task_names, eval_args.data_home_dir,
            time_interval=args.time_interval, train=True if eval_args.eval_on_train else False,
            debug=args.debug, run_on_cv_server=eval_args.run_on_cv_server,
            num_cpus=eval_args.num_workers, depth_descriptor=eval_args.depth_descriptor
        )
    end = time.time()
    print(f'Loaded data. Time: {end - start:.3f} seconds')
    print(f'There are {len(data)} data.')
    queue = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False if eval_args.no_shuffle else True,
        num_workers=0, drop_last=True
    )
    eval_r3m_queue = None
    if eval_args.eval_r3m:
        eval_r3m_queue = torch.utils.data.DataLoader(
            data, batch_size=1, shuffle=True,
            num_workers=0, drop_last=True
        )
    print('Creating data loader: done')

    current_hand_depth_estimates = []
    current_wrist_depths_real = []
    future_hand_depth_estimates = []
    future_wrist_depths_real = []
    pred_hand_depth_estimates = []
    for step, data in tqdm(enumerate(queue), 'Going through data...'):
        (
            hand_r3m_embedding, robot_r3m_embedding,
            original_task, hand,
            current_hand_bbox, future_hand_bbox,
            current_camera, future_camera,
            current_img_shape, future_img_shape,
            current_hand_depth_estimate, future_hand_depth_estimate,
            current_wrist_depth_real, future_wrist_depth_real,
            current_hand_pose, future_hand_pose,
            current_hand_shape, future_hand_shape,
            current_hand_pose_path, future_hand_pose_path
        ) = data
        i = torch.argmax(original_task.squeeze())
        original_task_name = eval_task_names[i]

        # visualize model output using original data
        task = torch.zeros(1, len(eval_task_names))
        task[0, i] = 1
        task_input = task.clone()
        if eval_args.dataset not in demo_datasets:
            if args.run_on_cv_server and not eval_args.run_on_cv_server:
                # training on cv evaluating on cluster -> convert cluster task to cv task
                task_input = cluster_task_to_cv_task(task.squeeze()).unsqueeze(0)
            if not args.run_on_cv_server and eval_args.run_on_cv_server:
                # training on cluster evaluating on cv -> convert cv task to cluster task
                task_input = cv_task_to_cluster_task(task.squeeze()).unsqueeze(0)

        current_hand_bbox = current_hand_bbox.to(device).float()
        current_hand_pose = current_hand_pose.to(device)
        current_hand_shape = current_hand_shape.to(device)
        future_hand_bbox = future_hand_bbox.to(device).float()
        future_hand_pose = future_hand_pose.to(device)
        future_hand_shape = future_hand_shape.to(device)
        r3m_embedding = robot_r3m_embedding if eval_args.robot_demos else hand_r3m_embedding
        input = torch.cat((
            r3m_embedding, task_input,
            current_hand_bbox.cpu(), current_hand_pose.cpu(), current_hand_shape.cpu(),
            current_camera, future_camera
        ), dim=1).to(device).float()

        with torch.no_grad():
            output = model(input)
            pred_hand_bbox = torch.sigmoid(
                output[:, :args.hand_bbox_dim]
            )  # force positive values for bbox output
            pred_hand_pose = output[:, args.hand_bbox_dim:(args.hand_bbox_dim + args.hand_pose_dim)]
            pred_hand_shape = output[:, (args.hand_bbox_dim + args.hand_pose_dim):]

            if eval_args.eval_loss:
                hand_bbox_loss = l2_loss_func(pred_hand_bbox, future_hand_bbox)
                hand_pose_loss = l2_loss_func(pred_hand_pose, future_hand_pose)
                hand_shape_loss = l2_loss_func(pred_hand_shape, future_hand_shape)
                loss = args.lambda1 * hand_bbox_loss + \
                       args.lambda2 * hand_pose_loss + \
                       args.lambda3 * hand_shape_loss
                bl_hand_bbox_loss = l2_loss_func(current_hand_bbox, future_hand_bbox)
                bl_hand_pose_loss = l2_loss_func(current_hand_pose, future_hand_pose)
                bl_hand_shape_loss = l2_loss_func(current_hand_shape, future_hand_shape)
                bl_loss = args.lambda1 * bl_hand_bbox_loss + \
                          args.lambda2 * bl_hand_pose_loss + \
                          args.lambda3 * bl_hand_shape_loss

                writer.add_scalar('loss/overall', loss, step)
                writer.add_scalar('loss/hand_bbox', hand_bbox_loss, step)
                writer.add_scalar('loss/hand_pose', hand_pose_loss, step)
                writer.add_scalar('loss/hand_shape', hand_shape_loss, step)
                writer.add_scalar('baseline_loss/overall', bl_loss, step)
                writer.add_scalar('baseline_loss/hand_bbox', bl_hand_bbox_loss, step)
                writer.add_scalar('baseline_loss/hand_pose', bl_hand_pose_loss, step)
                writer.add_scalar('baseline_loss/hand_shape', bl_hand_shape_loss, step)

        placeholder_pose = current_hand_pose if eval_args.use_current_frame_info else future_hand_pose
        placeholder_bbox = current_hand_bbox if eval_args.use_current_frame_info else future_hand_bbox
        placeholder_shape = current_hand_shape if eval_args.use_current_frame_info else future_hand_shape

        # calculate depth
        pred_hand_depth_estimate = None
        if eval_args.log_depth:
            img_y, img_x = future_img_shape[0]
            img_y, img_x = img_y.item(), img_x.item()
            if eval_args.depth_descriptor == 'wrist_img_z':
                pred_hand_depth_estimate = pose_to_joint_depth(
                    hand_mocap=hand_mocap_depth,
                    hand=hand,
                    pose=pred_hand_pose if args.predict_hand_pose else placeholder_pose,
                    bbox=pred_hand_bbox if args.predict_hand_bbox else placeholder_bbox,
                    cam=future_camera.to(device),
                    img_shape=future_img_shape.to(device),
                    device=device,
                    shape=pred_hand_shape if args.predict_hand_shape else placeholder_shape,
                    shape_path=None
                )[0].cpu().detach().item()
            elif eval_args.depth_descriptor == 'bbox_size':
                assert args.predict_hand_bbox
                unnormalized_pred_hand_bbox = unnormalize_bbox(
                    pred_hand_bbox[0].cpu().detach().numpy(),
                    (img_x, img_y)
                )
                *_, w, h = unnormalized_pred_hand_bbox
                pred_bbox_size = w * h
                pred_hand_depth_estimate = 1. / pred_bbox_size
            elif eval_args.depth_descriptor == 'scaling_factor':
                raise NotImplementedError
            elif eval_args.depth_descriptor == 'normalized_bbox_size':
                assert args.predict_hand_bbox
                *_, w, h = pred_hand_bbox[0].cpu().detach().numpy()
                pred_normalized_bbox_size = w * h
                pred_hand_depth_estimate = 1. / pred_normalized_bbox_size

        if eval_args.log_depth_scatter_plots:
            current_hand_depth_estimates.append(current_hand_depth_estimate.item())
            current_wrist_depths_real.append(current_wrist_depth_real.item())
            pred_hand_depth_estimates.append(pred_hand_depth_estimate)
            if not eval_args.no_future_info:
                future_hand_depth_estimates.append(future_hand_depth_estimate.item())
                future_wrist_depths_real.append(future_wrist_depth_real.item())

        task_name = eval_task_names[torch.argmax(original_task.squeeze())]
        vis_img = generate_single_visualization(
            current_hand_pose_path=current_hand_pose_path[0],
            future_hand_pose_path=None if eval_args.no_future_info else future_hand_pose_path[0],
            future_cam=future_camera[0].cpu().numpy(),
            hand=hand[0],
            pred_hand_bbox=pred_hand_bbox[0] if args.predict_hand_bbox else placeholder_bbox[0],
            pred_hand_pose=pred_hand_pose[0] if args.predict_hand_pose else placeholder_pose[0],
            pred_hand_shape=pred_hand_shape[0] if args.predict_hand_shape else placeholder_shape[0],
            task_names=eval_task_names,
            task=task[0],
            visualizer=visualizer,
            hand_mocap=hand_mocap_vis,
            use_visualizer=eval_args.use_visualizer,
            run_on_cv_server=eval_args.run_on_cv_server,
            robot_demos=eval_args.robot_demos,
            log_depth=eval_args.log_depth,
            current_depth=current_hand_depth_estimate.item() if eval_args.log_depth else None,
            future_depth=future_hand_depth_estimate.item() if eval_args.log_depth else None,
            pred_depth=pred_hand_depth_estimate if eval_args.log_depth else None
        )
        writer.add_image(f'vis_images_{task_name}/{step}', vis_img, dataformats='HWC')

        # evaluate robot and human image conditioning together
        if eval_args.robot_demos:
            vis_imgs = [vis_img]  # vis_img with robot r3m as input
            hand_input = torch.cat((
                hand_r3m_embedding, task_input,
                current_hand_bbox.cpu(), current_hand_pose.cpu(), current_hand_shape.cpu(),
                current_camera, future_camera
            ), dim=1).to(device).float()

            with torch.no_grad():
                output = model(hand_input)
                pred_hand_bbox = torch.sigmoid(
                    output[:, :args.hand_bbox_dim]
                )  # force positive values for bbox output
                pred_hand_pose = output[:, args.hand_bbox_dim:(args.hand_bbox_dim + args.hand_pose_dim)]
                pred_hand_shape = output[:, (args.hand_bbox_dim + args.hand_pose_dim):]

            hand_vis_img = generate_single_visualization(
                current_hand_pose_path=current_hand_pose_path[0],
                future_hand_pose_path=None if eval_args.no_future_info else future_hand_pose_path[0],
                future_cam=future_camera[0].cpu().numpy(),
                hand=hand[0],
                pred_hand_bbox=pred_hand_bbox[0] if args.predict_hand_bbox else placeholder_bbox[0],
                pred_hand_pose=pred_hand_pose[0] if args.predict_hand_pose else placeholder_pose[0],
                pred_hand_shape=pred_hand_shape[0] if args.predict_hand_shape else placeholder_shape[0],
                task_names=eval_task_names,
                task=task[0],
                visualizer=visualizer,
                hand_mocap=hand_mocap_vis,
                use_visualizer=eval_args.use_visualizer,
                run_on_cv_server=eval_args.run_on_cv_server,
                robot_demos=False,
                log_depth=eval_args.log_depth,
                current_depth=current_hand_depth_estimate.item() if eval_args.log_depth else None,
                future_depth=future_hand_depth_estimate.item() if eval_args.log_depth else None,
                pred_depth=pred_hand_depth_estimate if eval_args.log_depth else None
            )
            vis_imgs.append(hand_vis_img)
            final_vis_img = np.hstack(vis_imgs)
            writer.add_image(f'vis_hand_robot/{step}', final_vis_img, dataformats='HWC')

        if eval_args.eval_r3m:
            vis_imgs = []
            for k in range(eval_args.eval_r3m_samples):
                if k != 0:
                    data = next(iter(eval_r3m_queue))
                (
                    new_hand_r3m_embedding, new_robot_r3m_embedding,
                    *_,
                    new_current_hand_pose_path, new_future_hand_pose_path
                ) = data
                new_r3m_embedding = new_robot_r3m_embedding if eval_args.robot_demos else new_hand_r3m_embedding
                new_input = torch.cat((
                    new_r3m_embedding, task_input,
                    current_hand_bbox.cpu(), current_hand_pose.cpu(), current_hand_shape.cpu(),
                    current_camera, future_camera
                ), dim=1).to(device).float()

                with torch.no_grad():
                    output = model(new_input)
                    pred_hand_bbox = torch.sigmoid(
                        output[:, :args.hand_bbox_dim]
                    )  # force positive values for bbox output
                    pred_hand_pose = output[:, args.hand_bbox_dim:(args.hand_bbox_dim + args.hand_pose_dim)]
                    pred_hand_shape = output[:, (args.hand_bbox_dim + args.hand_pose_dim):]

                with open(new_current_hand_pose_path[0], 'rb') as f:
                    new_current_hand_info = pickle.load(f)
                current_img = load_img_from_hand_info(
                    new_current_hand_info, eval_args.robot_demos, eval_args.run_on_cv_server
                )

                new_vis_img = generate_single_visualization(
                    current_hand_pose_path=current_hand_pose_path[0],
                    future_hand_pose_path=None if eval_args.no_future_info else future_hand_pose_path[0],
                    future_cam=future_camera[0].cpu().numpy(),
                    hand=hand[0],
                    pred_hand_bbox=pred_hand_bbox[0] if args.predict_hand_bbox else placeholder_bbox[0],
                    pred_hand_pose=pred_hand_pose[0] if args.predict_hand_pose else placeholder_pose[0],
                    pred_hand_shape=pred_hand_shape[0] if args.predict_hand_shape else placeholder_shape[0],
                    task_names=eval_task_names,
                    task=task[0],
                    visualizer=visualizer,
                    hand_mocap=hand_mocap_vis,
                    use_visualizer=eval_args.use_visualizer,
                    run_on_cv_server=eval_args.run_on_cv_server,
                    robot_demos=eval_args.robot_demos,
                    current_img=current_img,
                    future_img=current_img
                )
                vis_imgs.append(new_vis_img)

            final_vis_img = np.hstack(vis_imgs)
            writer.add_image(f'vis_r3m/{step}', final_vis_img, dataformats='HWC')

        # visualize model output conditioning on different task inputs
        if eval_args.eval_tasks:
            vis_imgs = []
            for i, task_name in enumerate(eval_task_names):
                task = torch.zeros(1, len(eval_task_names))
                task[0, i] = 1
                task_input = task.clone()
                if eval_args.dataset not in demo_datasets:
                    if args.run_on_cv_server and not eval_args.run_on_cv_server:
                        task_input = cluster_task_to_cv_task(task.squeeze()).unsqueeze(0)
                    if not args.run_on_cv_server and eval_args.run_on_cv_server:
                        task_input = cv_task_to_cluster_task(task.squeeze()).unsqueeze(0)

                input = torch.cat((
                    r3m_embedding, task_input,
                    current_hand_bbox.cpu(), current_hand_pose.cpu(), current_hand_shape.cpu(),
                    current_camera, future_camera
                ), dim=1).to(device).float()

                with torch.no_grad():
                    output = model(input)
                    pred_hand_bbox = torch.sigmoid(
                        output[:, :args.hand_bbox_dim]
                    )  # force positive values for bbox output
                    pred_hand_pose = output[:, args.hand_bbox_dim:(args.hand_bbox_dim + args.hand_pose_dim)]
                    pred_hand_shape = output[:, (args.hand_bbox_dim + args.hand_pose_dim):]

                vis_img = generate_single_visualization(
                    current_hand_pose_path=current_hand_pose_path[0],
                    future_hand_pose_path=None if eval_args.no_future_info else future_hand_pose_path[0],
                    future_cam=future_camera[0].cpu().numpy(),
                    hand=hand[0],
                    pred_hand_bbox=pred_hand_bbox[0] if args.predict_hand_bbox else placeholder_bbox[0],
                    pred_hand_pose=pred_hand_pose[0] if args.predict_hand_pose else placeholder_pose[0],
                    pred_hand_shape=pred_hand_shape[0] if args.predict_hand_shape else placeholder_shape[0],
                    task_names=eval_task_names,
                    task=task[0],
                    visualizer=visualizer,
                    hand_mocap=hand_mocap_vis,
                    use_visualizer=eval_args.use_visualizer,
                    run_on_cv_server=eval_args.run_on_cv_server,
                    original_task=True if task_name == original_task_name else False,
                    robot_demos=eval_args.robot_demos
                )
                vis_imgs.append(vis_img)
            final_vis_img = np.hstack(vis_imgs)
            writer.add_image(f'vis_tasks/{step}', final_vis_img, dataformats='HWC')


        if step + 1 == eval_args.n_eval_samples:
            break

    if eval_args.log_depth_scatter_plots:
        import matplotlib.pyplot as plt
        current_estimates, current_real = [], []
        for estimate, real in zip(current_hand_depth_estimates, current_wrist_depths_real):
            if real == 0 or real == -999:
                continue
            current_estimates.append(estimate)
            current_real.append(real)
        fig = plt.figure()
        title = 'current vs. current real'
        plt.title(title)
        plt.scatter(current_estimates, current_real)
        writer.add_figure(f'scatter_plots/{title}', fig)
        plt.close(fig)

        if not eval_args.no_future_info:
            future_estimates, future_real = [], []
            for estimate, real in zip(future_hand_depth_estimates, future_wrist_depths_real):
                if real == 0 or real == -999:
                    continue
                future_estimates.append(estimate)
                future_real.append(real)
            fig = plt.figure()
            title = 'future vs. future real'
            plt.title(title)
            plt.scatter(future_estimates, future_real)
            writer.add_figure(f'scatter_plots/{title}', fig)
            plt.close(fig)

            pred_estimates, future_real = [], []
            for estimate, real in zip(pred_hand_depth_estimates, future_wrist_depths_real):
                if real == 0 or real == -999:
                    continue
                pred_estimates.append(estimate)
                future_real.append(real)
            fig = plt.figure()
            title = 'prediction vs. future real'
            plt.title(title)
            plt.scatter(pred_estimates, future_real)
            writer.add_figure(f'scatter_plots/{title}', fig)
            plt.close(fig)

            fig = plt.figure()
            title = 'future vs. prediction'
            plt.title(title)
            plt.scatter(future_hand_depth_estimates, pred_hand_depth_estimates)
            writer.add_figure(f'scatter_plots/{title}', fig)
            plt.close(fig)

if __name__ == '__main__':
    eval_args = parse_args()
    main(eval_args)
