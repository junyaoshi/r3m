import argparse
from collections import OrderedDict
import os
from os.path import join
from tqdm import tqdm
import sys

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import SomethingSomethingR3M
from bc_utils import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Training BC network.')

    # training
    parser.add_argument("--model_type", type=str, default="r3m_bc",
                        choices=[
                            'r3m_bc', # BatchNorm + 2-layer MLP
                        ],
                        help="type of network to use")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='perform evaluation after this many epochs')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save model after this many epochs')
    parser.add_argument('--vis_freq', type=int, default=100,
                        help='visualize rendered images after this many steps')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataloaders')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    parser.add_argument('--run_on_cv_server', action='store_true',
                        help='if true, run one task on cv-server; else, run all tasks on cluster')
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
    args.save = join(args.root, args.save)
    print(f'args: \n{args}')
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'Device: {device}.')
    if args.run_on_cv_server:
        task_names = ['push_left_right']
    else:
        task_names = ['move_away', 'move_towards', 'move_down', 'move_up',
                      'pull_left', 'pull_right', 'push_left', 'push_right', 'push_slightly']
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
    print(f'Loaded model {args.model_type}')
    print(f'param size = {count_parameters_in_M(model)}M')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.MSELoss()

    print('Creating data loaders...')
    train_data = SomethingSomethingR3M(
        task_names, args.data_home_dir, train=True, debug=args.debug, run_on_cv_server=args.run_on_cv_server
    )
    print(f'There are {len(train_data)} train data.')
    valid_data = SomethingSomethingR3M(
        task_names, args.data_home_dir, train=False, debug=args.debug, run_on_cv_server=args.run_on_cv_server
    )
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
        epoch_loss, epoch_bbox_loss, epoch_hand_pose_loss, global_step = train(
            train_queue, model, optimizer, global_step,
            writer, loss_func, device, visualizer, hand_mocap, args
        )
        print(f'epoch train loss: {epoch_loss}')
        print(f'epoch train bbox loss: {epoch_bbox_loss}')
        print(f'epoch train hand pose loss: {epoch_hand_pose_loss}')
        writer.add_scalar('train/epoch_loss', epoch_loss, global_step)
        writer.add_scalar('train/epoch_bbox_loss', epoch_bbox_loss, global_step)
        writer.add_scalar('train/epoch_hand_pose_loss', epoch_hand_pose_loss, global_step)

        # Evaluation.
        if epoch % args.eval_freq == 0 or epoch == (args.epochs - 1):
            epoch_loss, epoch_bbox_loss, epoch_hand_pose_loss = test(
                valid_queue, model, global_step, writer, loss_func, device, visualizer, hand_mocap, args
            )
            print(f'epoch valid loss: {epoch_loss}')
            print(f'epoch valid bbox loss: {epoch_bbox_loss}')
            print(f'epoch valid hand pose loss: {epoch_hand_pose_loss}')
            writer.add_scalar('valid/epoch_loss', epoch_loss, global_step)
            writer.add_scalar('valid/epoch_bbox_loss', epoch_bbox_loss, global_step)
            writer.add_scalar('valid/epoch_hand_pose_loss', epoch_hand_pose_loss, global_step)

        # Save model.
        if epoch % args.save_freq == 0 or epoch == (args.epochs - 1):
            print('saving the model.')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'global_step': global_step,
                        'args': args},
                       join(args.save, f'checkpoint_{epoch:04d}.pt'))

    # Final validation.
    epoch_loss, epoch_bbox_loss, epoch_hand_pose_loss = test(
        valid_queue, model, global_step, writer, loss_func, device, visualizer, hand_mocap, args
    )
    print(f'final epoch valid loss: {epoch_loss}')
    print(f'final epoch valid bbox loss: {epoch_bbox_loss}')
    print(f'final epoch valid hand pose loss: {epoch_hand_pose_loss}')
    writer.add_scalar('valid/epoch_loss', epoch_loss, global_step)
    writer.add_scalar('valid/epoch_bbox_loss', epoch_bbox_loss, global_step)
    writer.add_scalar('valid/epoch_hand_pose_loss', epoch_hand_pose_loss, global_step)


def train(
        train_queue, model, optimizer, global_step,
        writer, loss_func, device, visualizer, hand_mocap, args
):
    model.train()
    epoch_loss = AvgrageMeter()
    epoch_bbox_loss = AvgrageMeter()
    epoch_hand_pose_loss = AvgrageMeter()
    for step, data in tqdm(enumerate(train_queue), desc='Going through train data...'):
        r3m_embedding, task, hand, \
        current_hand_bbox, future_hand_bbox, \
        current_camera, future_camera, \
        current_hand_pose, future_hand_pose, \
        current_hand_pose_path, future_hand_pose_path = data
        input = torch.cat((
            r3m_embedding, task, current_hand_bbox, current_hand_pose, current_camera, future_camera
        ), dim=1).to(device).float()
        target = torch.cat((
            future_hand_bbox, future_hand_pose
        ), dim=1).to(device).float()
        future_hand_bbox = future_hand_bbox.to(device)
        future_hand_pose = future_hand_pose.to(device)

        optimizer.zero_grad()
        output = model(input)
        output[:, :future_hand_bbox.size(1)] = torch.sigmoid(
            output[:, :future_hand_bbox.size(1)]
        ) # force positive values
        loss = loss_func(output, target)
        bbox_loss = loss_func(output[:, :future_hand_bbox.size(1)], future_hand_bbox)
        hand_pose_loss = loss_func(output[:, future_hand_bbox.size(1):], future_hand_pose)
        loss.backward()
        optimizer.step()
        epoch_loss.update(loss.data, 1)
        epoch_bbox_loss.update(bbox_loss.data, 1)
        epoch_hand_pose_loss.update(hand_pose_loss.data, 1)

        if (global_step + 1) % 100 == 0:
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/bbox_loss', bbox_loss, global_step)
            writer.add_scalar('train/hand_pose_loss', hand_pose_loss, global_step)

        if (global_step + 1) % args.vis_freq == 0:
            vis_img = generate_single_visualization(
                current_hand_pose_path=current_hand_pose_path[0],
                future_hand_pose_path=future_hand_pose_path[0],
                future_cam=future_camera[0].cpu().numpy(),
                hand=hand[0],
                pred_hand_bbox=output[0, :future_hand_bbox.size(1)].detach().cpu().numpy(),
                pred_hand_pose=output[0, future_hand_bbox.size(1):].detach().cpu().numpy(),
                visualizer=visualizer,
                hand_mocap=hand_mocap,
                use_visualizer=args.use_visualizer,
                device=device,
                run_on_cv_server=args.run_on_cv_server
            )
            writer.add_image(f'train/vis', vis_img, global_step, dataformats='HWC')

        global_step += 1

    return epoch_loss.avg, epoch_bbox_loss.avg, epoch_hand_pose_loss.avg, global_step


def test(valid_queue, model, global_step, writer, loss_func, device, visualizer, hand_mocap, args):
    epoch_loss = AvgrageMeter()
    epoch_bbox_loss = AvgrageMeter()
    epoch_hand_pose_loss = AvgrageMeter()
    model.eval()
    for step, data in tqdm(enumerate(valid_queue), 'Going through valid data...'):
        r3m_embedding, task, hand, \
        current_hand_bbox, future_hand_bbox, \
        current_camera, future_camera, \
        current_hand_pose, future_hand_pose, \
        current_hand_pose_path, future_hand_pose_path = data
        input = torch.cat((
            r3m_embedding, task, current_hand_bbox, current_hand_pose, current_camera, future_camera
        ), dim=1).to(device).float()
        target = torch.cat((
            future_hand_bbox, future_hand_pose
        ), dim=1).to(device).float()
        future_hand_bbox = future_hand_bbox.to(device)
        future_hand_pose = future_hand_pose.to(device)

        with torch.no_grad():
            output = model(input)
            output[:, :future_hand_bbox.size(1)] = torch.sigmoid(
                output[:, :future_hand_bbox.size(1)]
            )  # force positive values
            loss = loss_func(output, target)
            bbox_loss = loss_func(output[:, :future_hand_bbox.size(1)], future_hand_bbox)
            hand_pose_loss = loss_func(output[:, future_hand_bbox.size(1):], future_hand_pose)
        epoch_loss.update(loss.data, input.size(0))
        epoch_bbox_loss.update(bbox_loss.data, input.size(0))
        epoch_hand_pose_loss.update(hand_pose_loss.data, input.size(0))

    vis_img = generate_single_visualization(
        current_hand_pose_path=current_hand_pose_path[0],
        future_hand_pose_path=future_hand_pose_path[0],
        future_cam=future_camera[0].cpu().numpy(),
        hand=hand[0],
        pred_hand_bbox=output[0, :future_hand_bbox.size(1)].detach().cpu().numpy(),
        pred_hand_pose=output[0, future_hand_bbox.size(1):].detach().cpu().numpy(),
        visualizer=visualizer,
        hand_mocap=hand_mocap,
        use_visualizer=args.use_visualizer,
        device=device,
        run_on_cv_server=args.run_on_cv_server
    )
    writer.add_image(f'valid/vis', vis_img, global_step, dataformats='HWC')

    return epoch_loss.avg, epoch_bbox_loss.avg, epoch_hand_pose_loss.avg


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    main(args)
