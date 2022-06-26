import argparse
import time
from collections import OrderedDict
import os
from os.path import join
from tqdm import tqdm
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import SomethingSomethingR3M
from bc_utils import (
    count_parameters_in_M, AvgrageMeter, generate_single_visualization, pose_to_joint_depth,
    CV_TASKS, CLUSTER_TASKS
)
from resnet import EndtoEndNet, TransferableNet

SANITY_CHECK_SIZE = 10


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Training BC network.')

    # training
    parser.add_argument("--model_type", type=str, default="e2e",
                        choices=[
                            'e2e',  # end-to-end behavorial cloning network
                            'transfer'  # separate representations for hand and non-hand features
                        ],
                        help="model type to use")
    parser.add_argument("--n_blocks", type=int, default=4,
                        help="number of 2-layer blocks in the network")
    parser.add_argument("--net_type", type=str, default="mlp",
                        choices=[
                            'mlp',  # multilayer perceptrons without residual connections
                            'residual'  # residual network
                        ],
                        help="network architecture to use")
    parser.add_argument('--time_interval', type=int, default=5,
                        help='how many frames into the future to predict')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='weight for hand bbox loss, set to 0 to disable hand bbox prediction')
    parser.add_argument('--lambda2', type=float, default=1.0,
                        help='weight for hand pose lossset to 0 to disable hand pose prediction')
    parser.add_argument('--lambda3', type=float, default=1.0,
                        help='weight for hand shape lossset to 0 to disable hand shape prediction')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='perform evaluation after this many epochs')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save model after this many epochs')
    parser.add_argument('--vis_freq', type=int, default=2000,
                        help='visualize rendered images after this many steps')
    parser.add_argument('--log_depth_freq', type=int, default=2000,
                        help='compute and log predicted and ground truth depth '
                             'using smplx after this many steps of training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--sanity_check', action='store_true', default=False,
                        help='perform sanity check (try to only fit a few examples)')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Evaluate model on training set instead of validation set (for debugging)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--vis_sample_size', type=int, default=5,
                        help='number of samples to visualize on tensorboard')
    parser.add_argument('--task_vis_sample_size', type=int, default=2,
                        help='number of task-conditioned samples to visualize on tensorboard')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataloaders')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    parser.add_argument('--run_on_cv_server', action='store_true',
                        help='if true, run tasks on cv-server; else, run all tasks on cluster')
    parser.add_argument('--debug', action='store_true',
                        help='if true, enter debug mode, load 50 videos and no parallel workers')
    parser.add_argument('--use_visualizer', action='store_true',
                        help='if true, use opengl visualizer to render results and show on tensorboard')

    # paths
    parser.add_argument('--root', type=str, default='checkpoints',
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


def main(args):
    if args.sanity_check:
        args.eval_on_train = True
        assert args.batch_size <= SANITY_CHECK_SIZE
    assert args.vis_sample_size <= args.batch_size
    args.save = join(args.root, args.save)
    assert (args.lambda1 >= 0) and (args.lambda2 >= 0) and (args.lambda3 >= 0)
    args.predict_hand_bbox = args.lambda1 != 0.
    args.predict_hand_pose = args.lambda2 != 0.
    args.predict_hand_shape = args.lambda3 != 0.
    assert args.predict_hand_bbox or args.predict_hand_pose or args.predict_hand_shape
    print(f'args: \n{args}')
    print(f'Predicting hand bbox: {args.predict_hand_bbox}.')
    print(f'Predicting hand pose: {args.predict_hand_pose}.')
    print(f'Predicting hand shape: {args.predict_hand_shape}.')
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'Device: {device}.')
    task_names = CV_TASKS if args.run_on_cv_server else CLUSTER_TASKS
    writer = SummaryWriter(log_dir=args.save, flush_secs=60)

    # visualizer and frank mocap
    print('Loading frankmocap visualizer...')
    sys.path.insert(1, args.frankmocap_path)
    os.chdir(args.frankmocap_path)
    from renderer.visualizer import Visualizer
    from handmocap.hand_mocap_api import HandMocap
    visualizer = Visualizer('opengl') if args.use_visualizer else None
    checkpoint_hand = join(
        args.frankmocap_path,
        'extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
    )
    smpl_dir = join(args.frankmocap_path, 'extra_data/smpl')
    hand_mocap_vis = HandMocap(checkpoint_hand, smpl_dir, device=device, batch_size=1)
    hand_mocap_depth = HandMocap(checkpoint_hand, smpl_dir, device=device, batch_size=args.batch_size)
    os.chdir(args.r3m_path)
    print('Visualizer loaded.')

    # compute dimensions
    r3m_dim, task_dim, cam_dim = 2048, len(task_names), 3
    args.hand_bbox_dim, args.hand_pose_dim, args.hand_shape_dim = 4, 48, 10
    hand_dim = sum([args.hand_bbox_dim, args.hand_pose_dim, args.hand_shape_dim])
    input_dim = sum([r3m_dim, task_dim, hand_dim, cam_dim, cam_dim])
    output_dim = hand_dim

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
    print(f'Loaded model type: {args.model_type}, blocks: {args.n_blocks}, network: {args.net_type}')
    print(f'param size = {count_parameters_in_M(model)}M')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    l2_loss_func = nn.MSELoss()
    l1_loss_func = nn.L1Loss()

    print('Creating data loaders...')
    start = time.time()
    train_data = SomethingSomethingR3M(
        task_names, args.data_home_dir,
        time_interval=args.time_interval, train=True,
        debug=args.debug, run_on_cv_server=args.run_on_cv_server, num_cpus=args.num_workers
    )
    end = time.time()
    print(f'Loaded train data. Time: {end - start:.5f} seconds')
    if args.sanity_check:
        print('Performing sanity check on a few examples.')
        indices = range(1, SANITY_CHECK_SIZE + 1)
        train_data = torch.utils.data.Subset(train_data, indices)
    print(f'There are {len(train_data)} train data.')
    if args.eval_on_train:
        print('Evaluating on training set instead of validation set.')
        valid_data = train_data
    else:
        start = time.time()
        valid_data = SomethingSomethingR3M(
            task_names, args.data_home_dir,
            time_interval=args.time_interval, train=False,
            debug=args.debug, run_on_cv_server=args.run_on_cv_server, num_cpus=args.num_workers
        )
        end = time.time()
        print(f'Loaded valid data. Time: {end - start:.5f} seconds')
    print(f'There are {len(valid_data)} valid data.')
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0 if args.debug else args.num_workers, drop_last=True)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0 if args.debug else args.num_workers, drop_last=True)
    print('Creating data loaders: done')

    if args.cont_training:
        checkpoint_name = sorted([f for f in os.listdir(args.save) if 'checkpoint' in f])[-1]
        checkpoint_file = os.path.join(args.save, checkpoint_name)
        print(f'loading checkpoint: {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        global_step = checkpoint['global_step']
    else:
        global_step, init_epoch = 0, 1

    for epoch in range(init_epoch, args.epochs + 1):
        print(f'epoch {epoch}')

        # Training.
        train_stats = train(
            train_queue, model, optimizer, global_step,
            writer, l2_loss_func, l1_loss_func, device,
            visualizer, hand_mocap_vis, hand_mocap_depth,
            task_names, args
        )
        (
            epoch_loss, epoch_hand_bbox_loss, epoch_hand_pose_loss, epoch_hand_shape_loss,
            epoch_bl_loss, epoch_bl_hand_bbox_loss, epoch_bl_hand_pose_loss, epoch_bl_hand_shape_loss,
            global_step
        ) = train_stats
        print(f'epoch train loss: {epoch_loss}')
        print(f'epoch train hand bbox loss: {epoch_hand_bbox_loss}')
        print(f'epoch train hand pose loss: {epoch_hand_pose_loss}')
        print(f'epoch train hand shape loss: {epoch_hand_shape_loss}')
        writer.add_scalar('train_epoch/loss', epoch_loss, epoch)
        writer.add_scalar('train_epoch/hand_bbox_loss', epoch_hand_bbox_loss, epoch)
        writer.add_scalar('train_epoch/hand_pose_loss', epoch_hand_pose_loss, epoch)
        writer.add_scalar('train_epoch/hand_shape_loss', epoch_hand_shape_loss, epoch)
        writer.add_scalar('train_epoch_baseline/loss', epoch_bl_loss, epoch)
        writer.add_scalar('train_epoch_baseline/hand_bbox_loss', epoch_bl_hand_bbox_loss, epoch)
        writer.add_scalar('train_epoch_baseline/hand_pose_loss', epoch_bl_hand_pose_loss, epoch)
        writer.add_scalar('train_epoch_baseline/hand_shape_loss', epoch_bl_hand_shape_loss, epoch)
        writer.add_scalar('stats/epoch_steps', global_step, epoch)
        writer.add_scalar('stats/lambda1', args.lambda1, epoch)
        writer.add_scalar('stats/lambda2', args.lambda2, epoch)
        writer.add_scalar('stats/lambda3', args.lambda3, epoch)

        # Evaluation.
        if epoch % args.eval_freq == 0 or epoch == (args.epochs - 1):
            valid_stats = test(
                valid_queue, model, global_step,
                writer, l2_loss_func, l1_loss_func, device,
                visualizer, hand_mocap_vis, hand_mocap_depth,
                task_names, args
            )
            (
                epoch_loss, epoch_hand_bbox_loss, epoch_hand_pose_loss, epoch_hand_shape_loss,
                epoch_bl_loss, epoch_bl_hand_bbox_loss, epoch_bl_hand_pose_loss, epoch_bl_hand_shape_loss
            ) = valid_stats
            print(f'epoch valid loss: {epoch_loss}')
            print(f'epoch valid hand bbox loss: {epoch_hand_bbox_loss}')
            print(f'epoch valid hand pose loss: {epoch_hand_pose_loss}')
            print(f'epoch valid hand shape loss: {epoch_hand_shape_loss}')
            writer.add_scalar('valid/loss', epoch_loss, epoch)
            writer.add_scalar('valid/hand_bbox_loss', epoch_hand_bbox_loss, epoch)
            writer.add_scalar('valid/hand_pose_loss', epoch_hand_pose_loss, epoch)
            writer.add_scalar('valid/hand_shape_loss', epoch_hand_shape_loss, epoch)
            writer.add_scalar('valid_baseline/loss', epoch_bl_loss, epoch)
            writer.add_scalar('valid_baseline/hand_bbox_loss', epoch_bl_hand_bbox_loss, epoch)
            writer.add_scalar('valid_baseline/hand_pose_loss', epoch_bl_hand_pose_loss, epoch)
            writer.add_scalar('valid_baseline/hand_shape_loss', epoch_bl_hand_shape_loss, epoch)

        # Save model.
        if epoch % args.save_freq == 0 or epoch == (args.epochs - 1):
            print('saving the model.')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'global_step': global_step,
                        'args': args},
                       join(args.save, f'checkpoint_{epoch:04d}.pt'))

    # Final validation.
    valid_stats = test(
        valid_queue, model, global_step,
        writer, l2_loss_func, l1_loss_func, device,
        visualizer, hand_mocap_vis, hand_mocap_depth,
        task_names, args
    )
    (
        epoch_loss, epoch_hand_bbox_loss, epoch_hand_pose_loss, epoch_hand_shape_loss,
        epoch_bl_loss, epoch_bl_hand_bbox_loss, epoch_bl_hand_pose_loss, epoch_bl_hand_shape_loss
    ) = valid_stats
    print(f'final epoch valid loss: {epoch_loss}')
    print(f'final epoch valid hand bbox loss: {epoch_hand_bbox_loss}')
    print(f'final epoch valid hand pose loss: {epoch_hand_pose_loss}')
    print(f'final epoch valid hand shape loss: {epoch_hand_shape_loss}')
    writer.add_scalar('valid/loss', epoch_loss, args.epochs)
    writer.add_scalar('valid/hand_bbox_loss', epoch_hand_bbox_loss, args.epochs)
    writer.add_scalar('valid/hand_pose_loss', epoch_hand_pose_loss, args.epochs)
    writer.add_scalar('valid/hand_shape_loss', epoch_hand_shape_loss, args.epochs)
    writer.add_scalar('valid_baseline/loss', epoch_bl_loss, args.epochs)
    writer.add_scalar('valid_baseline/hand_bbox_loss', epoch_bl_hand_bbox_loss, args.epochs)
    writer.add_scalar('valid_baseline/hand_pose_loss', epoch_bl_hand_pose_loss, args.epochs)
    writer.add_scalar('valid_baseline/hand_shape_loss', epoch_bl_hand_shape_loss, args.epochs)


def train(
        train_queue, model, optimizer, global_step,
        writer, l2_loss_func, l1_loss_func, device,
        visualizer, hand_mocap_vis, hand_mocap_depth,
        task_names, args
):
    model.train()
    epoch_loss = AvgrageMeter()
    epoch_hand_bbox_loss = AvgrageMeter()
    epoch_hand_pose_loss = AvgrageMeter()
    epoch_hand_shape_loss = AvgrageMeter()
    epoch_bl_loss = AvgrageMeter()
    epoch_bl_hand_bbox_loss = AvgrageMeter()
    epoch_bl_hand_pose_loss = AvgrageMeter()
    epoch_bl_hand_shape_loss = AvgrageMeter()

    for step, data in tqdm(enumerate(train_queue), desc='Going through train data...'):
        (
            r3m_embedding, task, hand,
            current_hand_bbox, future_hand_bbox,
            current_camera, future_camera,
            future_img_shape, future_joint_depth,
            current_hand_pose, future_hand_pose,
            current_hand_shape, future_hand_shape,
            current_hand_pose_path, future_hand_pose_path
        ) = data
        input = torch.cat((
            r3m_embedding, task,
            current_hand_bbox, current_hand_pose, current_hand_shape,
            current_camera, future_camera
        ), dim=1).to(device).float()
        current_hand_bbox = current_hand_bbox.to(device).float()
        current_hand_pose = current_hand_pose.to(device)
        current_hand_shape = current_hand_shape.to(device)
        future_hand_bbox = future_hand_bbox.to(device).float()
        future_hand_pose = future_hand_pose.to(device)
        future_hand_shape = future_hand_shape.to(device)

        optimizer.zero_grad()
        output = model(input)
        pred_hand_bbox = torch.sigmoid(
            output[:, :args.hand_bbox_dim]
        )  # force positive values for bbox output
        pred_hand_pose = output[:, args.hand_bbox_dim:(args.hand_bbox_dim + args.hand_pose_dim)]
        pred_hand_shape = output[:, (args.hand_bbox_dim + args.hand_pose_dim):]
        hand_bbox_loss = l2_loss_func(pred_hand_bbox, future_hand_bbox)
        hand_pose_loss = l2_loss_func(pred_hand_pose, future_hand_pose)
        hand_shape_loss = l2_loss_func(pred_hand_shape, future_hand_shape)
        loss = args.lambda1 * hand_bbox_loss + \
               args.lambda2 * hand_pose_loss + \
               args.lambda3 * hand_shape_loss
        with torch.no_grad():
            bl_hand_bbox_loss = l2_loss_func(current_hand_bbox, future_hand_bbox)
            bl_hand_pose_loss = l2_loss_func(current_hand_pose, future_hand_pose)
            bl_hand_shape_loss = l2_loss_func(current_hand_shape, future_hand_shape)
            bl_loss = args.lambda1 * bl_hand_bbox_loss + \
                      args.lambda2 * bl_hand_pose_loss + \
                      args.lambda3 * bl_hand_shape_loss

        loss.backward()
        optimizer.step()
        epoch_loss.update(loss.data, 1)
        epoch_hand_bbox_loss.update(hand_bbox_loss.data, 1)
        epoch_hand_pose_loss.update(hand_pose_loss.data, 1)
        epoch_hand_shape_loss.update(hand_shape_loss.data, 1)
        epoch_bl_loss.update(bl_loss.data, 1)
        epoch_bl_hand_bbox_loss.update(bl_hand_bbox_loss.data, 1)
        epoch_bl_hand_pose_loss.update(bl_hand_pose_loss.data, 1)
        epoch_bl_hand_shape_loss.update(bl_hand_shape_loss.data, 1)

        # log scalars
        log_freq = 1 if args.sanity_check else 100
        if (global_step + 1) % log_freq == 0:
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/hand_bbox_loss', hand_bbox_loss, global_step)
            writer.add_scalar('train/hand_pose_loss', hand_pose_loss, global_step)
            writer.add_scalar('train/hand_shape_loss', hand_shape_loss, global_step)
            writer.add_scalar('train_baseline/loss', bl_loss, global_step)
            writer.add_scalar('train_baseline/hand_bbox_loss', bl_hand_bbox_loss, global_step)
            writer.add_scalar('train_baseline/hand_pose_loss', bl_hand_pose_loss, global_step)
            writer.add_scalar('train_baseline/hand_shape_loss', bl_hand_shape_loss, global_step)

        # log depth
        if (global_step + 1) % args.log_depth_freq == 0 and not args.sanity_check:
            with torch.no_grad():
                future_joint_depth = future_joint_depth.to(device)
                future_joint_depth_avg = torch.mean(future_joint_depth, dim=1)
                pred_joint_depth = pose_to_joint_depth(
                    hand_mocap=hand_mocap_depth,
                    hand=hand,
                    pose=pred_hand_pose if args.predict_hand_pose else future_hand_pose,
                    bbox=pred_hand_bbox if args.predict_hand_bbox else future_hand_bbox,
                    cam=future_camera.to(device),
                    img_shape=future_img_shape.to(device),
                    device=device,
                    shape=pred_hand_shape if args.predict_hand_shape else future_hand_shape,
                    shape_path=None
                )
                pred_joint_depth_avg = torch.mean(pred_joint_depth, dim=1)
                joint_depth_loss = l1_loss_func(pred_joint_depth, future_joint_depth)
                joint_depth_avg_loss = l1_loss_func(pred_joint_depth_avg, future_joint_depth_avg)

            writer.add_scalar('train/joint_depth_loss', joint_depth_loss, global_step)
            writer.add_scalar('train/joint_depth_avg_loss', joint_depth_avg_loss, global_step)

        # log images
        if (global_step + 1) % args.vis_freq == 0 and not args.sanity_check:
            # visualize some samples in the batch
            vis_imgs = []
            for i in range(args.vis_sample_size):
                vis_img = generate_single_visualization(
                    current_hand_pose_path=current_hand_pose_path[i],
                    future_hand_pose_path=future_hand_pose_path[i],
                    future_cam=future_camera[i].cpu().numpy(),
                    hand=hand[i],
                    pred_hand_bbox=pred_hand_bbox[i] if args.predict_hand_bbox else future_hand_bbox[i],
                    pred_hand_pose=pred_hand_pose[i] if args.predict_hand_pose else future_hand_pose[i],
                    pred_hand_shape=pred_hand_shape[i] if args.predict_hand_shape else future_hand_shape[i],
                    task_names=task_names,
                    task=task[i],
                    visualizer=visualizer,
                    hand_mocap=hand_mocap_vis,
                    use_visualizer=args.use_visualizer,
                    run_on_cv_server=args.run_on_cv_server
                )
                vis_imgs.append(vis_img)
            final_vis_img = np.hstack(vis_imgs)
            writer.add_image(f'train/vis_images', final_vis_img, global_step, dataformats='HWC')

            # visualize different task conditioning
            for i in range(args.task_vis_sample_size):
                original_task_name = task_names[torch.argmax(task[i].squeeze())]
                all_task_instances = []
                for j, task_name in enumerate(task_names):
                    task_instance = torch.zeros(1, len(task_names))
                    task_instance[0, j] = 1
                    all_task_instances.append(task_instance)
                all_task_instances = torch.vstack(all_task_instances)

                task_conditioned_input = torch.cat((
                    r3m_embedding[i].repeat(len(task_names), 1).to(device),
                    all_task_instances.to(device),
                    current_hand_bbox[i].repeat(len(task_names), 1),
                    current_hand_pose[i].repeat(len(task_names), 1),
                    current_hand_shape[i].repeat(len(task_names), 1),
                    current_camera[i].repeat(len(task_names), 1).to(device),
                    future_camera[i].repeat(len(task_names), 1).to(device)
                ), dim=1).float()

                model.eval()
                with torch.no_grad():
                    task_conditioned_output = model(task_conditioned_input)
                    task_pred_hand_bbox = torch.sigmoid(
                        task_conditioned_output[:, :args.hand_bbox_dim]
                    )  # force positive values for bbox output
                    task_pred_hand_pose = task_conditioned_output[:,
                                          args.hand_bbox_dim:(args.hand_bbox_dim + args.hand_pose_dim)]
                    task_pred_hand_shape = task_conditioned_output[:,
                                           (args.hand_bbox_dim + args.hand_pose_dim):]

                task_vis_imgs = []
                for j, task_name in enumerate(task_names):
                    task_vis_img = generate_single_visualization(
                        current_hand_pose_path=current_hand_pose_path[i],
                        future_hand_pose_path=future_hand_pose_path[i],
                        future_cam=future_camera[i].cpu().numpy(),
                        hand=hand[i],
                        pred_hand_bbox=task_pred_hand_bbox[j] if args.predict_hand_bbox else future_hand_bbox[i],
                        pred_hand_pose=task_pred_hand_pose[j] if args.predict_hand_pose else future_hand_pose[i],
                        pred_hand_shape=task_pred_hand_shape[j] if args.predict_hand_shape else future_hand_shape[i],
                        task_names=task_names,
                        task=all_task_instances[j],
                        visualizer=visualizer,
                        hand_mocap=hand_mocap_vis,
                        use_visualizer=args.use_visualizer,
                        run_on_cv_server=args.run_on_cv_server,
                        original_task=True if task_name == original_task_name else False
                    )
                    task_vis_imgs.append(task_vis_img)
                final_task_vis_img = np.hstack(task_vis_imgs)
                writer.add_image(f'train/vis_tasks_{i}', final_task_vis_img, global_step, dataformats='HWC')

            model.train()

        global_step += 1

    train_stats = (
        epoch_loss.avg, epoch_hand_bbox_loss.avg, epoch_hand_pose_loss.avg, epoch_hand_shape_loss.avg,
        epoch_bl_loss.avg, epoch_bl_hand_bbox_loss.avg, epoch_bl_hand_pose_loss.avg, epoch_bl_hand_shape_loss.avg,
        global_step
    )
    return train_stats


def test(
        valid_queue, model, global_step,
        writer, l2_loss_func, l1_loss_func, device,
        visualizer, hand_mocap_vis, hand_mocap_depth,
        task_names, args
):
    epoch_loss = AvgrageMeter()
    epoch_hand_bbox_loss = AvgrageMeter()
    epoch_hand_pose_loss = AvgrageMeter()
    epoch_hand_shape_loss = AvgrageMeter()
    epoch_bl_loss = AvgrageMeter()
    epoch_bl_hand_bbox_loss = AvgrageMeter()
    epoch_bl_hand_pose_loss = AvgrageMeter()
    epoch_bl_hand_shape_loss = AvgrageMeter()
    model.eval()

    for step, data in tqdm(enumerate(valid_queue), 'Going through valid data...'):
        (
            r3m_embedding, task, hand,
            current_hand_bbox, future_hand_bbox,
            current_camera, future_camera,
            future_img_shape, future_joint_depth,
            current_hand_pose, future_hand_pose,
            current_hand_shape, future_hand_shape,
            current_hand_pose_path, future_hand_pose_path
        ) = data
        input = torch.cat((
            r3m_embedding, task,
            current_hand_bbox, current_hand_pose, current_hand_shape,
            current_camera, future_camera
        ), dim=1).to(device).float()
        current_hand_bbox = current_hand_bbox.to(device).float()
        current_hand_pose = current_hand_pose.to(device)
        current_hand_shape = current_hand_shape.to(device)
        future_hand_bbox = future_hand_bbox.to(device).float()
        future_hand_pose = future_hand_pose.to(device)
        future_hand_shape = future_hand_shape.to(device)

        with torch.no_grad():
            output = model(input)
            pred_hand_bbox = torch.sigmoid(
                output[:, :args.hand_bbox_dim]
            )  # force positive values for bbox output
            pred_hand_pose = output[:, args.hand_bbox_dim:(args.hand_bbox_dim + args.hand_pose_dim)]
            pred_hand_shape = output[:, (args.hand_bbox_dim + args.hand_pose_dim):]
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
        epoch_loss.update(loss.data, input.size(0))
        epoch_hand_bbox_loss.update(hand_bbox_loss.data, input.size(0))
        epoch_hand_pose_loss.update(hand_pose_loss.data, input.size(0))
        epoch_hand_shape_loss.update(hand_shape_loss.data, input.size(0))
        epoch_bl_loss.update(bl_loss.data, input.size(0))
        epoch_bl_hand_bbox_loss.update(bl_hand_bbox_loss.data, input.size(0))
        epoch_bl_hand_pose_loss.update(bl_hand_pose_loss.data, input.size(0))
        epoch_bl_hand_shape_loss.update(bl_hand_shape_loss.data, input.size(0))

    # log depth
    with torch.no_grad():
        future_joint_depth = future_joint_depth.to(device)
        future_joint_depth_avg = torch.mean(future_joint_depth, dim=1)
        pred_joint_depth = pose_to_joint_depth(
            hand_mocap=hand_mocap_depth,
            hand=hand,
            pose=pred_hand_pose if args.predict_hand_pose else future_hand_pose,
            bbox=pred_hand_bbox if args.predict_hand_bbox else future_hand_bbox,
            cam=future_camera.to(device),
            img_shape=future_img_shape.to(device),
            device=device,
            shape=pred_hand_shape if args.predict_hand_shape else future_hand_shape,
            shape_path=None
        )
        pred_joint_depth_avg = torch.mean(pred_joint_depth, dim=1)
        joint_depth_loss = l1_loss_func(pred_joint_depth, future_joint_depth)
        joint_depth_avg_loss = l1_loss_func(pred_joint_depth_avg, future_joint_depth_avg)

    writer.add_scalar('valid/joint_depth_loss', joint_depth_loss, global_step)
    writer.add_scalar('valid/joint_depth_avg_loss', joint_depth_avg_loss, global_step)

    # visualize some samples in the batch
    vis_imgs = []
    for i in range(args.vis_sample_size):
        vis_img = generate_single_visualization(
            current_hand_pose_path=current_hand_pose_path[i],
            future_hand_pose_path=future_hand_pose_path[i],
            future_cam=future_camera[i].cpu().numpy(),
            hand=hand[i],
            pred_hand_bbox=pred_hand_bbox[i] if args.predict_hand_bbox else future_hand_bbox[i],
            pred_hand_pose=pred_hand_pose[i] if args.predict_hand_pose else future_hand_pose[i],
            pred_hand_shape=pred_hand_shape[i] if args.predict_hand_shape else future_hand_shape[i],
            task_names=task_names,
            task=task[i],
            visualizer=visualizer,
            hand_mocap=hand_mocap_vis,
            use_visualizer=args.use_visualizer,
            run_on_cv_server=args.run_on_cv_server
        )
        vis_imgs.append(vis_img)
    final_vis_img = np.hstack(vis_imgs)
    writer.add_image(f'valid/vis_images', final_vis_img, global_step, dataformats='HWC')

    # visualize different task conditioning
    for i in range(args.task_vis_sample_size):
        original_task_name = task_names[torch.argmax(task[i].squeeze())]
        all_task_instances = []
        for j, task_name in enumerate(task_names):
            task_instance = torch.zeros(1, len(task_names))
            task_instance[0, j] = 1
            all_task_instances.append(task_instance)
        all_task_instances = torch.vstack(all_task_instances)

        task_conditioned_input = torch.cat((
            r3m_embedding[i].repeat(len(task_names), 1).to(device),
            all_task_instances.to(device),
            current_hand_bbox[i].repeat(len(task_names), 1),
            current_hand_pose[i].repeat(len(task_names), 1),
            current_hand_shape[i].repeat(len(task_names), 1),
            current_camera[i].repeat(len(task_names), 1).to(device),
            future_camera[i].repeat(len(task_names), 1).to(device)
        ), dim=1).float()
        with torch.no_grad():
            task_conditioned_output = model(task_conditioned_input)
            task_pred_hand_bbox = torch.sigmoid(
                task_conditioned_output[:, :args.hand_bbox_dim]
            )  # force positive values for bbox output
            task_pred_hand_pose = task_conditioned_output[:,
                                  args.hand_bbox_dim:(args.hand_bbox_dim + args.hand_pose_dim)]
            task_pred_hand_shape = task_conditioned_output[:,
                                   (args.hand_bbox_dim + args.hand_pose_dim):]

        task_vis_imgs = []
        for j, task_name in enumerate(task_names):
            task_vis_img = generate_single_visualization(
                current_hand_pose_path=current_hand_pose_path[i],
                future_hand_pose_path=future_hand_pose_path[i],
                future_cam=future_camera[i].cpu().numpy(),
                hand=hand[i],
                pred_hand_bbox=task_pred_hand_bbox[j] if args.predict_hand_bbox else future_hand_bbox[i],
                pred_hand_pose=task_pred_hand_pose[j] if args.predict_hand_pose else future_hand_pose[i],
                pred_hand_shape=task_pred_hand_shape[j] if args.predict_hand_shape else future_hand_shape[i],
                task_names=task_names,
                task=all_task_instances[j],
                visualizer=visualizer,
                hand_mocap=hand_mocap_vis,
                use_visualizer=args.use_visualizer,
                run_on_cv_server=args.run_on_cv_server,
                original_task=True if task_name == original_task_name else False
            )
            task_vis_imgs.append(task_vis_img)
        final_task_vis_img = np.hstack(task_vis_imgs)
        writer.add_image(f'valid/vis_tasks_{i}', final_task_vis_img, global_step, dataformats='HWC')

    valid_stats = (
        epoch_loss.avg, epoch_hand_bbox_loss.avg, epoch_hand_pose_loss.avg, epoch_hand_shape_loss.avg,
        epoch_bl_loss.avg, epoch_bl_hand_bbox_loss.avg, epoch_bl_hand_pose_loss.avg, epoch_bl_hand_shape_loss.avg,
    )

    return valid_stats


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    main(args)
