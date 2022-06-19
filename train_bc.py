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
    count_parameters_in_M, AvgrageMeter, generate_single_visualization, CV_TASKS, CLUSTER_TASKS
)
from resnet import FullyConnectedResNet

SANITY_CHECK_SIZE = 10

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Training BC network.')

    # training
    parser.add_argument("--model_type", type=str, default="r3m_bc",
                        choices=[
                            'r3m_bc', # BatchNorm + 2-layer MLP
                            'resnet2',  # Fully Connected ResNet with 2 residual layers
                            'resnet4',  # Fully Connected ResNet with 4 residual layers
                            'resnet8', # Fully Connected ResNet with 8 residual layers
                            'resnet16',  # Fully Connected ResNet with 16 residual layers
                            'resnet32',  # Fully Connected ResNet with 32 residual layers
                        ],
                        help="type of network to use")
    parser.add_argument('--time_interval', type=int, default=5,
                        help='how many frames into the future to predict')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='hyperparameter for balancing hand pose and bbox losses')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='perform evaluation after this many epochs')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save model after this many epochs')
    parser.add_argument('--vis_freq', type=int, default=100,
                        help='visualize rendered images after this many steps')
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
    print(f'args: \n{args}')
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'Device: {device}.')
    task_names = CV_TASKS if args.run_on_cv_server else CLUSTER_TASKS
    writer = SummaryWriter(log_dir=args.save, flush_secs=60)

    # visualizer and frank mocap
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
    hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device=device)
    os.chdir(args.r3m_path)

    # compute dimensions
    r3m_dim, task_dim, hand_pose_dim, bbox_dim, cam_dim = 2048, len(task_names), 48, 4, 3
    input_dim = sum([r3m_dim, task_dim, hand_pose_dim, bbox_dim, cam_dim, cam_dim])
    output_dim = sum([hand_pose_dim, bbox_dim])

    model = None
    if args.model_type == 'r3m_bc':
        model = nn.Sequential(OrderedDict([
            ('batchnorm', nn.BatchNorm1d(input_dim)),
            ('fc1', nn.Linear(input_dim, 256)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(256, output_dim))
        ])).to(device).float()
    elif args.model_type == 'resnet2':
        model = FullyConnectedResNet(
            in_features=input_dim, out_features=output_dim, n_res_blocsk=1
        ).to(device).float()
    elif args.model_type == 'resnet4':
        model = FullyConnectedResNet(
            in_features=input_dim, out_features=output_dim, n_res_blocsk=2
        ).to(device).float()
    elif args.model_type == 'resnet8':
        model = FullyConnectedResNet(
            in_features=input_dim, out_features=output_dim, n_res_blocsk=4
        ).to(device).float()
    elif args.model_type == 'resnet16':
        model = FullyConnectedResNet(
            in_features=input_dim, out_features=output_dim, n_res_blocsk=8
        ).to(device).float()
    elif args.model_type == 'resnet32':
        model = FullyConnectedResNet(
            in_features=input_dim, out_features=output_dim, n_res_blocsk=16
        ).to(device).float()
    print(f'Loaded model {args.model_type}')
    print(f'param size = {count_parameters_in_M(model)}M')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.MSELoss()

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
            writer, loss_func, device, visualizer, hand_mocap, task_names, args
        )
        (
            epoch_loss, epoch_bbox_loss, epoch_hand_pose_loss,
            epoch_bl_loss, epoch_bl_bbox_loss, epoch_bl_hand_pose_loss,
            global_step
        ) = train_stats
        print(f'epoch train loss: {epoch_loss}')
        print(f'epoch train bbox loss: {epoch_bbox_loss}')
        print(f'epoch train hand pose loss: {epoch_hand_pose_loss}')
        writer.add_scalar('train_epoch/loss', epoch_loss, epoch)
        writer.add_scalar('train_epoch/bbox_loss', epoch_bbox_loss, epoch)
        writer.add_scalar('train_epoch/hand_pose_loss', epoch_hand_pose_loss, epoch)
        writer.add_scalar('train_epoch_baseline/loss', epoch_bl_loss, epoch)
        writer.add_scalar('train_epoch_baseline/bbox_loss', epoch_bl_bbox_loss, epoch)
        writer.add_scalar('train_epoch_baseline/hand_pose_loss', epoch_bl_hand_pose_loss, epoch)
        writer.add_scalar('stats/epoch_steps', global_step, epoch)
        writer.add_scalar('stats/beta', args.beta, epoch)

        # Evaluation.
        if epoch % args.eval_freq == 0 or epoch == (args.epochs - 1):
            valid_stats = test(
                valid_queue, model, global_step,
                writer, loss_func, device, visualizer, hand_mocap, task_names, args
            )
            (
                epoch_loss, epoch_bbox_loss, epoch_hand_pose_loss,
                epoch_bl_loss, epoch_bl_bbox_loss, epoch_bl_hand_pose_loss
            ) = valid_stats
            print(f'epoch valid loss: {epoch_loss}')
            print(f'epoch valid bbox loss: {epoch_bbox_loss}')
            print(f'epoch valid hand pose loss: {epoch_hand_pose_loss}')
            writer.add_scalar('valid/loss', epoch_loss, epoch)
            writer.add_scalar('valid/bbox_loss', epoch_bbox_loss, epoch)
            writer.add_scalar('valid/hand_pose_loss', epoch_hand_pose_loss, epoch)
            writer.add_scalar('valid_baseline/loss', epoch_bl_loss, epoch)
            writer.add_scalar('valid_baseline/bbox_loss', epoch_bl_bbox_loss, epoch)
            writer.add_scalar('valid_baseline/hand_pose_loss', epoch_bl_hand_pose_loss, epoch)

        # Save model.
        if epoch % args.save_freq == 0 or epoch == (args.epochs - 1):
            print('saving the model.')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'global_step': global_step,
                        'args': args},
                       join(args.save, f'checkpoint_{epoch:04d}.pt'))

    # Final validation.
    valid_stats = test(
        valid_queue, model, global_step, writer,
        loss_func, device, visualizer, hand_mocap, task_names, args
    )
    (
        epoch_loss, epoch_bbox_loss, epoch_hand_pose_loss,
        epoch_bl_loss, epoch_bl_bbox_loss, epoch_bl_hand_pose_loss
    ) = valid_stats
    print(f'final epoch valid loss: {epoch_loss}')
    print(f'final epoch valid bbox loss: {epoch_bbox_loss}')
    print(f'final epoch valid hand pose loss: {epoch_hand_pose_loss}')
    writer.add_scalar('valid/loss', epoch_loss, args.epochs)
    writer.add_scalar('valid/bbox_loss', epoch_bbox_loss, args.epochs)
    writer.add_scalar('valid/hand_pose_loss', epoch_hand_pose_loss, args.epochs)
    writer.add_scalar('valid_baseline/loss', epoch_bl_loss, args.epochs)
    writer.add_scalar('valid_baseline/bbox_loss', epoch_bl_bbox_loss, args.epochs)
    writer.add_scalar('valid_baseline/hand_pose_loss', epoch_bl_hand_pose_loss, args.epochs)


def train(
        train_queue, model, optimizer, global_step,
        writer, loss_func, device, visualizer, hand_mocap, task_names, args
):
    model.train()
    epoch_loss = AvgrageMeter()
    epoch_bbox_loss = AvgrageMeter()
    epoch_hand_pose_loss = AvgrageMeter()
    epoch_bl_loss = AvgrageMeter()
    epoch_bl_bbox_loss = AvgrageMeter()
    epoch_bl_hand_pose_loss = AvgrageMeter()

    for step, data in tqdm(enumerate(train_queue), desc='Going through train data...'):
        (
            r3m_embedding, task, hand,
            current_hand_bbox, future_hand_bbox,
            current_camera, future_camera,
            current_hand_pose, future_hand_pose,
            current_hand_pose_path, future_hand_pose_path
        ) = data
        input = torch.cat((
            r3m_embedding, task, current_hand_bbox, current_hand_pose, current_camera, future_camera
        ), dim=1).to(device).float()
        target = torch.cat((
            future_hand_bbox, future_hand_pose
        ), dim=1).to(device).float()
        baseline = torch.cat((
            current_hand_bbox, current_hand_pose
        ), dim=1).to(device).float()
        future_hand_bbox = future_hand_bbox.to(device).float()
        future_hand_pose = future_hand_pose.to(device)

        optimizer.zero_grad()
        output = model(input)
        pred_hand_bbox = torch.sigmoid(
            output[:, :future_hand_bbox.size(1)]
        ) # force positive values for bbox output
        pred_hand_pose = output[:, future_hand_bbox.size(1):]
        bbox_loss = loss_func(pred_hand_bbox, future_hand_bbox)
        hand_pose_loss = loss_func(pred_hand_pose, future_hand_pose)
        loss = bbox_loss + args.beta * hand_pose_loss
        with torch.no_grad():
            bl_loss = loss_func(baseline, target)
            bl_bbox_loss = loss_func(baseline[:, :current_hand_bbox.size(1)], future_hand_bbox)
            bl_hand_pose_loss = loss_func(baseline[:, current_hand_bbox.size(1):], future_hand_pose)

        loss.backward()
        optimizer.step()
        epoch_loss.update(loss.data, 1)
        epoch_bbox_loss.update(bbox_loss.data, 1)
        epoch_hand_pose_loss.update(hand_pose_loss.data, 1)
        epoch_bl_loss.update(bl_loss.data, 1)
        epoch_bl_bbox_loss.update(bl_bbox_loss.data, 1)
        epoch_bl_hand_pose_loss.update(bl_hand_pose_loss.data, 1)

        log_freq = 1 if args.sanity_check else 100
        if (global_step + 1) % log_freq == 0:
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/bbox_loss', bbox_loss, global_step)
            writer.add_scalar('train/hand_pose_loss', hand_pose_loss, global_step)
            writer.add_scalar('train_baseline/loss', bl_loss, global_step)
            writer.add_scalar('train_baseline/bbox_loss', bl_bbox_loss, global_step)
            writer.add_scalar('train_baseline/hand_pose_loss', bl_hand_pose_loss, global_step)

        if (global_step + 1) % args.vis_freq == 0 and not args.sanity_check:
            # visualize some samples in the batch
            vis_imgs = []
            for i in range(args.vis_sample_size):
                vis_img = generate_single_visualization(
                    current_hand_pose_path=current_hand_pose_path[i],
                    future_hand_pose_path=future_hand_pose_path[i],
                    future_cam=future_camera[i].cpu().numpy(),
                    hand=hand[i],
                    pred_hand_bbox=pred_hand_bbox[i].detach().cpu().numpy(),
                    pred_hand_pose=pred_hand_pose[i].detach().cpu().numpy(),
                    task_names=task_names,
                    task=task[i],
                    visualizer=visualizer,
                    hand_mocap=hand_mocap,
                    use_visualizer=args.use_visualizer,
                    device=device,
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
                    r3m_embedding[i].repeat(len(task_names), 1),
                    all_task_instances,
                    current_hand_bbox[i].repeat(len(task_names), 1),
                    current_hand_pose[i].repeat(len(task_names), 1),
                    current_camera[i].repeat(len(task_names), 1),
                    future_camera[i].repeat(len(task_names), 1)
                ), dim=1).to(device).float()
                model.eval()
                with torch.no_grad():
                    task_conditioned_output = model(task_conditioned_input)
                    task_pred_hand_bbox = torch.sigmoid(
                        task_conditioned_output[:, :future_hand_bbox.size(1)]
                    )  # force positive values for bbox output
                    task_pred_hand_pose = task_conditioned_output[:, future_hand_bbox.size(1):]

                task_vis_imgs = []
                for j, task_name in enumerate(task_names):
                    task_vis_img = generate_single_visualization(
                        current_hand_pose_path=current_hand_pose_path[i],
                        future_hand_pose_path=future_hand_pose_path[i],
                        future_cam=future_camera[i].cpu().numpy(),
                        hand=hand[i],
                        pred_hand_bbox=task_pred_hand_bbox[j].detach().cpu().numpy(),
                        pred_hand_pose=task_pred_hand_pose[j].detach().cpu().numpy(),
                        task_names=task_names,
                        task=all_task_instances[j],
                        visualizer=visualizer,
                        hand_mocap=hand_mocap,
                        use_visualizer=args.use_visualizer,
                        device=device,
                        run_on_cv_server=args.run_on_cv_server,
                        original_task=True if task_name == original_task_name else False
                    )
                    task_vis_imgs.append(task_vis_img)
                final_task_vis_img = np.hstack(task_vis_imgs)
                writer.add_image(f'train/vis_tasks_{i}', final_task_vis_img, global_step, dataformats='HWC')

            model.train()

        global_step += 1

    train_stats = (
        epoch_loss.avg, epoch_bbox_loss.avg, epoch_hand_pose_loss.avg,
        epoch_bl_loss.avg, epoch_bl_bbox_loss.avg, epoch_bl_hand_pose_loss.avg,
        global_step
    )
    return train_stats


def test(valid_queue, model, global_step,
         writer, loss_func, device, visualizer, hand_mocap, task_names, args):
    epoch_loss = AvgrageMeter()
    epoch_bbox_loss = AvgrageMeter()
    epoch_hand_pose_loss = AvgrageMeter()
    epoch_bl_loss = AvgrageMeter()
    epoch_bl_bbox_loss = AvgrageMeter()
    epoch_bl_hand_pose_loss = AvgrageMeter()
    model.eval()
    for step, data in tqdm(enumerate(valid_queue), 'Going through valid data...'):
        (
            r3m_embedding, task, hand,
            current_hand_bbox, future_hand_bbox,
            current_camera, future_camera,
            current_hand_pose, future_hand_pose,
            current_hand_pose_path, future_hand_pose_path
        ) = data
        input = torch.cat((
            r3m_embedding, task, current_hand_bbox, current_hand_pose, current_camera, future_camera
        ), dim=1).to(device).float()
        target = torch.cat((
            future_hand_bbox, future_hand_pose
        ), dim=1).to(device).float()
        baseline = torch.cat((
            current_hand_bbox, current_hand_pose
        ), dim=1).to(device).float()
        future_hand_bbox = future_hand_bbox.to(device).float()
        future_hand_pose = future_hand_pose.to(device)

        with torch.no_grad():
            output = model(input)
            pred_hand_bbox = torch.sigmoid(
                output[:, :future_hand_bbox.size(1)]
            )  # force positive values for bbox output
            pred_hand_pose = output[:, future_hand_bbox.size(1):]
            bbox_loss = loss_func(pred_hand_bbox, future_hand_bbox)
            hand_pose_loss = loss_func(pred_hand_pose, future_hand_pose)
            loss = bbox_loss + args.beta * hand_pose_loss
            bl_loss = loss_func(baseline, target)
            bl_bbox_loss = loss_func(baseline[:, :current_hand_bbox.size(1)], future_hand_bbox)
            bl_hand_pose_loss = loss_func(baseline[:, current_hand_bbox.size(1):], future_hand_pose)
        epoch_loss.update(loss.data, input.size(0))
        epoch_bbox_loss.update(bbox_loss.data, input.size(0))
        epoch_hand_pose_loss.update(hand_pose_loss.data, input.size(0))
        epoch_bl_loss.update(bl_loss.data, 1)
        epoch_bl_bbox_loss.update(bl_bbox_loss.data, 1)
        epoch_bl_hand_pose_loss.update(bl_hand_pose_loss.data, 1)

    # visualize some samples in the batch
    vis_imgs = []
    for i in range(args.vis_sample_size):
        vis_img = generate_single_visualization(
            current_hand_pose_path=current_hand_pose_path[i],
            future_hand_pose_path=future_hand_pose_path[i],
            future_cam=future_camera[i].cpu().numpy(),
            hand=hand[i],
            pred_hand_bbox=pred_hand_bbox[i].detach().cpu().numpy(),
            pred_hand_pose=pred_hand_pose[i].detach().cpu().numpy(),
            task_names=task_names,
            task=task[i],
            visualizer=visualizer,
            hand_mocap=hand_mocap,
            use_visualizer=args.use_visualizer,
            device=device,
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
            r3m_embedding[i].repeat(len(task_names), 1),
            all_task_instances,
            current_hand_bbox[i].repeat(len(task_names), 1),
            current_hand_pose[i].repeat(len(task_names), 1),
            current_camera[i].repeat(len(task_names), 1),
            future_camera[i].repeat(len(task_names), 1)
        ), dim=1).to(device).float()
        with torch.no_grad():
            task_conditioned_output = model(task_conditioned_input)
            task_pred_hand_bbox = torch.sigmoid(
                task_conditioned_output[:, :future_hand_bbox.size(1)]
            )  # force positive values for bbox output
            task_pred_hand_pose = task_conditioned_output[:, future_hand_bbox.size(1):]

        task_vis_imgs = []
        for j, task_name in enumerate(task_names):
            task_vis_img = generate_single_visualization(
                current_hand_pose_path=current_hand_pose_path[i],
                future_hand_pose_path=future_hand_pose_path[i],
                future_cam=future_camera[i].cpu().numpy(),
                hand=hand[i],
                pred_hand_bbox=task_pred_hand_bbox[j].detach().cpu().numpy(),
                pred_hand_pose=task_pred_hand_pose[j].detach().cpu().numpy(),
                task_names=task_names,
                task=all_task_instances[j],
                visualizer=visualizer,
                hand_mocap=hand_mocap,
                use_visualizer=args.use_visualizer,
                device=device,
                run_on_cv_server=args.run_on_cv_server,
                original_task=True if task_name == original_task_name else False
            )
            task_vis_imgs.append(task_vis_img)
        final_task_vis_img = np.hstack(task_vis_imgs)
        writer.add_image(f'valid/vis_tasks_{i}', final_task_vis_img, global_step, dataformats='HWC')

    valid_stats = (
        epoch_loss.avg, epoch_bbox_loss.avg, epoch_hand_pose_loss.avg,
        epoch_bl_loss.avg, epoch_bl_bbox_loss.avg, epoch_bl_hand_pose_loss.avg
    )

    return valid_stats


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    main(args)
