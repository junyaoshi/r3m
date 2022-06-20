import argparse
from collections import OrderedDict
import time
from os.path import join
import os
import sys

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import SomethingSomethingR3M
from resnet import EndtoEndNet, TransferableNet
from bc_utils import (
    count_parameters_in_M, generate_single_visualization,
    CV_TASKS, CLUSTER_TASKS,
    cv_task_to_cluster_task, cluster_task_to_cv_task
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
                        help='number of workers for dataset multiprocessing')
    parser.add_argument('--use_visualizer', action='store_true',
                        help='if true, use opengl visualizer to render results and show on tensorboard')

    # paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='location of the checkpoint to load for evaluation')
    parser.add_argument('--root', type=str, default='eval_checkpoints',
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
    print(f'eval args: \n{eval_args}')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}.')
    eval_args.save = join(eval_args.root, eval_args.save)
    writer = SummaryWriter(log_dir=eval_args.save, flush_secs=60)

    # visualizer and frank mocap
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
    hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device=device)
    os.chdir(eval_args.r3m_path)

    # load a checkpoint
    print('loading the model at:')
    print(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']
    print(f'loaded train args: \n{args}')

    # compute dimensions
    train_task_names = CV_TASKS if args.run_on_cv_server else CLUSTER_TASKS
    r3m_dim, task_dim, hand_pose_dim, bbox_dim, cam_dim = 2048, len(train_task_names), 48, 4, 3
    input_dim = sum([r3m_dim, task_dim, hand_pose_dim, bbox_dim, cam_dim, cam_dim])
    output_dim = sum([hand_pose_dim, bbox_dim])

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
        dims=(r3m_dim, task_dim, hand_pose_dim, bbox_dim, cam_dim),
        n_blocks=args.n_blocks,
        residual=residual
    ).to(device).float()
    print(f'Loaded model type: {args.model_type}, blocks: {args.n_blocks}, '
          f'arch: {args.model_arch}, network: {args.net_type} at epoch {checkpoint["epoch"]}')
    print(f'param size = {count_parameters_in_M(model)}M')

    loss_func = torch.nn.MSELoss()

    print('Creating data loader...')
    eval_task_names = CV_TASKS if eval_args.run_on_cv_server else CLUSTER_TASKS
    start = time.time()
    data = SomethingSomethingR3M(
        eval_task_names, eval_args.data_home_dir,
        time_interval=args.time_interval, train=True if eval_args.eval_on_train else False,
        debug=args.debug, run_on_cv_server=eval_args.run_on_cv_server, num_cpus=eval_args.num_workers
    )
    end = time.time()
    print(f'Loaded train data. Time: {end - start:.3f} seconds')
    print(f'There are {len(data)} data.')
    queue = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=True,
        num_workers=0, drop_last=True
    )
    print('Creating data loader: done')

    for step, data in tqdm(enumerate(queue), 'Going through data...'):
        (
            r3m_embedding, origina_task, hand,
            current_hand_bbox, future_hand_bbox,
            current_camera, future_camera,
            current_hand_pose, future_hand_pose,
            current_hand_pose_path, future_hand_pose_path
        ) = data
        original_task_name = eval_task_names[torch.argmax(origina_task.squeeze())]

        # main pass
        vis_imgs = []
        for i, task_name in enumerate(eval_task_names):
            task = torch.zeros(1, len(eval_task_names))
            task[0, i] = 1
            task_input = task.clone()
            if args.run_on_cv_server and not eval_args.run_on_cv_server:
                task_input = cluster_task_to_cv_task(task.squeeze()).unsqueeze(0)
            if not args.run_on_cv_server and eval_args.run_on_cv_server:
                task_input = cv_task_to_cluster_task(task.squeeze()).unsqueeze(0)
            input = torch.cat((
                r3m_embedding, task_input, current_hand_bbox, current_hand_pose, current_camera, future_camera
            ), dim=1).to(device).float()

            with torch.no_grad():
                output = model(input)
                output[:, :future_hand_bbox.size(1)] = torch.sigmoid(
                    output[:, :future_hand_bbox.size(1)]
                )  # force positive values

            if task_name == original_task_name:
                target = torch.cat((
                    future_hand_bbox, future_hand_pose
                ), dim=1).to(device).float()
                baseline = torch.cat((
                    current_hand_bbox, current_hand_pose
                ), dim=1).to(device).float()
                future_hand_bbox = future_hand_bbox.to(device)
                future_hand_pose = future_hand_pose.to(device)
                with torch.no_grad():
                    loss = loss_func(output, target)
                    bbox_loss = loss_func(output[:, :future_hand_bbox.size(1)], future_hand_bbox)
                    hand_pose_loss = loss_func(output[:, future_hand_bbox.size(1):], future_hand_pose)
                    bl_loss = loss_func(baseline, target)
                    bl_bbox_loss = loss_func(baseline[:, :current_hand_bbox.size(1)], future_hand_bbox)
                    bl_hand_pose_loss = loss_func(baseline[:, current_hand_bbox.size(1):], future_hand_pose)

                writer.add_scalar('loss/overall', loss, step)
                writer.add_scalar('loss/bbox', bbox_loss, step)
                writer.add_scalar('loss/hand_pose', hand_pose_loss, step)
                writer.add_scalar('baseline_loss/overall', bl_loss, step)
                writer.add_scalar('baseline_loss/bbox', bl_bbox_loss, step)
                writer.add_scalar('baseline_loss/hand_pose', bl_hand_pose_loss, step)

            vis_img = generate_single_visualization(
                current_hand_pose_path=current_hand_pose_path[0],
                future_hand_pose_path=future_hand_pose_path[0],
                future_cam=future_camera[0].cpu().numpy(),
                hand=hand[0],
                pred_hand_bbox=output[0, :future_hand_bbox.size(1)].detach().cpu().numpy(),
                pred_hand_pose=output[0, future_hand_bbox.size(1):].detach().cpu().numpy(),
                task_names=eval_task_names,
                task=task[0],
                visualizer=visualizer,
                hand_mocap=hand_mocap,
                use_visualizer=eval_args.use_visualizer,
                device=device,
                run_on_cv_server=eval_args.run_on_cv_server,
                original_task=True if task_name == original_task_name else False
            )
            vis_imgs.append(vis_img)
        final_vis_img = np.hstack(vis_imgs)
        writer.add_image(f'vis_tasks', final_vis_img, step, dataformats='HWC')
        if step + 1 == eval_args.n_eval_samples:
            break

if __name__ == '__main__':
    eval_args = parse_args()
    main(eval_args)