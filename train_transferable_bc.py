import argparse
import time
import os
from os.path import join
from tqdm import tqdm
import sys
from collections import namedtuple
from pprint import pprint
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataset import AgentTransferable
from utils.bc_utils import (
    count_parameters_in_M, AvgrageMeter, load_pkl,
    evaluate_transferable_metric, evaluate_transferable_metric_batch,
    CV_TASKS, CLUSTER_TASKS
)
from bc_models.resnet import EndtoEndNet
from utils.vis_utils import generate_transferable_visualization


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Training BC network.')

    # model
    parser.add_argument("--n_blocks", type=int, default=8,
                        help="number of 2-layer blocks in the network")
    parser.add_argument("--net_type", type=str, default="residual",
                        choices=[
                            'mlp',  # multilayer perceptrons without residual connections
                            'residual'  # residual network
                        ],
                        help="network architecture to use")

    # data
    parser.add_argument('--time_interval', type=int, default=15,
                        help='how many frames into the future to predict')
    parser.add_argument('--iou_thresh', type=float, default=0.7,
                        help='IoU threshold for filtering the data')
    parser.add_argument("--depth_descriptor", type=str, default="scaling_factor",
                        choices=['scaling_factor', 'normalized_bbox_size'],
                        help="descriptor for estimating hand depth")
    parser.add_argument('--no_shuffle', action='store_true',
                        help='if true, dataloader is not shuffled')

    # training and evaluation
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='weight for xy loss')
    parser.add_argument('--lambda2', type=float, default=1.0,
                        help='weight for depth loss')
    parser.add_argument('--lambda3', type=float, default=1.0,
                        help='weight for orientation loss')
    parser.add_argument('--lambda4', type=float, default=1.0,
                        help='weight for contact loss')
    parser.add_argument('--pred_mode', type=str, default="residual",
                        choices=[
                            'original', # use original target values
                            'residual'  # use residual values (target - current)
                        ],
                        help='prediction mode for the model')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='perform evaluation after this many epochs')
    parser.add_argument('--log_scalar_freq', type=int, default=100,
                        help='log scalars after this many steps')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save model after this many epochs')
    parser.add_argument('--vis_freq', type=int, default=2000,
                        help='visualize rendered images after this many steps')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Evaluate model on training set instead of validation set (for debugging)')
    parser.add_argument('--eval_tasks', action='store_true',
                        help='if true, perform task-conditioned evaluatuation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--vis_sample_size', type=int, default=5,
                        help='number of samples to visualize on tensorboard, set to 0 to disable')
    parser.add_argument('--task_vis_sample_size', type=int, default=2,
                        help='number of task-conditioned samples to visualize on tensorboard, '
                             'set to 0 to disable task-conditioned visualization')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataloaders')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    parser.add_argument('--run_on_cv_server', action='store_true',
                        help='if true, run tasks on cv-server; else, run all tasks on cluster')
    parser.add_argument('--use_visualizer', action='store_true',
                        help='if true, use opengl visualizer to render results and show on tensorboard')

    # debugging and sanity check
    parser.add_argument('--sanity_check', action='store_true', default=False,
                        help='perform sanity check (try to only fit a few examples)')
    parser.add_argument('--sanity_check_size', type=int, default=32,
                        help='number of data for sanity check')
    parser.add_argument('--debug', action='store_true',
                        help='if true, enter debug mode, load 50 videos and no parallel workers')

    # paths
    parser.add_argument('--root', type=str, default='/home/junyao/LfHV/r3m/transferable_bc_ckpts',
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
    parser.add_argument('--depth_norm_params_path', type=str,
                        default='/home/junyao/LfHV/frankmocap/ss_utils/depth_normalization_params.pkl',
                        help='location of depth normalization params')
    parser.add_argument('--ori_norm_params_path', type=str,
                        default='/home/junyao/LfHV/frankmocap/ss_utils/ori_normalization_params.pkl',
                        help='location of orientation normalization params')
    parser.add_argument('--contact_count_path', type=str,
                        default='/home/junyao/LfHV/frankmocap/ss_utils/contact_count.pkl',
                        help='location of orientation normalization params')

    args = parser.parse_args()
    return args


def main(args):
    program_start = time.time()
    if args.sanity_check:
        args.no_shuffle = True
        args.eval_on_train = True
        args.log_scalar_freq = 1
        torch.manual_seed(5157)
        np.random.seed(5157)
        assert args.batch_size <= args.sanity_check_size
    args.pred_residual = args.pred_mode == 'residual'
    assert args.vis_sample_size <= args.batch_size
    args.save = join(args.root, args.save)
    assert (args.lambda1 > 0) and (args.lambda2 > 0) and (args.lambda3 > 0) and (args.lambda4 > 0)
    task_names = CV_TASKS if args.run_on_cv_server else CLUSTER_TASKS
    args.task_names = task_names
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}.')
    writer = SummaryWriter(log_dir=args.save, flush_secs=60)

    # visualizer and frank mocap
    print('Loading frankmocap visualizer...')
    sys.path.insert(1, args.frankmocap_path)
    os.chdir(args.frankmocap_path)
    from renderer.visualizer import Visualizer
    from handmocap.hand_mocap_api import HandMocap
    visualizer = Visualizer('opengl') if args.use_visualizer else None
    checkpoint_hand = join(args.frankmocap_path, 'extra_data/hand_module/pretrained_weights/pose_shape_best.pth')
    smpl_dir = join(args.frankmocap_path, 'extra_data/smpl')
    hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device=device, batch_size=1)
    os.chdir(args.r3m_path)
    print('Visualizer loaded.')

    # compute dimensions
    args.r3m_dim, args.task_dim = 2048, len(task_names)
    args.xy_dim, args.depth_dim, args.ori_dim, args.contact_dim = 2, 1, 3, 1
    args.input_dim = sum([args.r3m_dim, args.task_dim, args.xy_dim, args.depth_dim, args.ori_dim, args.contact_dim])
    args.output_dim = sum([args.xy_dim, args.depth_dim, args.ori_dim, args.contact_dim])

    # log args
    pprint(f'args: \n{args}')
    for arg in vars(args):
        writer.add_text(arg, str(getattr(args, arg)))

    # initialize model
    model = EndtoEndNet(
        in_features=args.input_dim,
        out_features=args.output_dim,
        dims=(),
        n_blocks=args.n_blocks,
        residual=args.net_type == 'residual'
    ).to(device).float()
    print(f'Loaded transferable BC model. Blocks: {args.n_blocks}, Network: {args.net_type}.')
    print(f'param size = {count_parameters_in_M(model)}M')

    # load pkl
    args.depth_norm_params = load_pkl(args.depth_norm_params_path)[args.depth_descriptor]
    args.ori_norm_params = load_pkl(args.ori_norm_params_path)
    args.contact_count = load_pkl(args.contact_count_path)
    args.bce_pos_weight = args.contact_count['n_neg'] / args.contact_count['n_pos']
    print(f'Positive Weight for BCE: {args.bce_pos_weight:.4f}')

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    l2_loss_func = nn.MSELoss()
    bce_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.bce_pos_weight).to(device))

    # create data loaders
    print('Creating data loaders...')
    data_start = time.time()
    train_data = AgentTransferable(
        data_home_dir=args.data_home_dir,
        task_names=task_names,
        split='train',
        iou_thresh=args.iou_thresh,
        time_interval=args.time_interval,
        depth_descriptor=args.depth_descriptor,
        depth_norm_params=args.depth_norm_params,
        ori_norm_params=args.ori_norm_params,
        debug=args.debug,
        run_on_cv_server=args.run_on_cv_server,
        num_cpus=args.num_workers,
        has_task_labels=True,
        has_future_labels=True
    )
    data_end = time.time()
    print(f'Loaded train data. Time: {data_end - data_start:.5f} seconds')
    if args.sanity_check:
        print(f'Performing sanity check on {args.sanity_check_size} examples.')
        indices = np.random.choice(range(1, len(train_data)), size=args.sanity_check_size)
        train_data = torch.utils.data.Subset(train_data, indices)
    print(f'There are {len(train_data)} train data.')
    if args.eval_on_train:
        print('Evaluating on training set instead of validation set.')
        valid_data = train_data
    else:
        data_start = time.time()
        valid_data = AgentTransferable(
            data_home_dir=args.data_home_dir,
            task_names=task_names,
            split='valid',
            iou_thresh=args.iou_thresh,
            time_interval=args.time_interval,
            depth_descriptor=args.depth_descriptor,
            depth_norm_params=args.depth_norm_params,
            ori_norm_params=args.ori_norm_params,
            debug=args.debug,
            run_on_cv_server=args.run_on_cv_server,
            num_cpus=args.num_workers,
            has_task_labels=True,
            has_future_labels=True
        )
        data_end = time.time()
        print(f'Loaded valid data. Time: {data_end - data_start:.5f} seconds')
    print(f'There are {len(valid_data)} valid data.')

    dataloader_num_workers = args.num_workers
    if args.debug or args.sanity_check:
        dataloader_num_workers = 0
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.no_shuffle,
        num_workers=dataloader_num_workers, drop_last=True
    )
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=not args.no_shuffle,
        num_workers=dataloader_num_workers, drop_last=True
    )
    print('Creating data loaders: done')

    if args.cont_training:
        if not [f for f in os.listdir(args.save) if 'checkpoint' in f]:
            print(f'No checkpoint found in directory: {args.save}')
            print('Begin training from scratch...')
            global_step, init_epoch = 0, 1
        else:
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
        print('Begin training from scratch...')
        global_step, init_epoch = 0, 1

    for epoch in range(init_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f'\nepoch {epoch}')

        # Training.
        train_stats = train(
            train_queue, model, optimizer, device,
            l2_loss_func, bce_loss_func,
            global_step, writer,
            visualizer, hand_mocap,
            task_names, args
        )
        global_step = train_stats.global_step
        log_epoch_stats(train_stats, writer, args, global_step, epoch, train=True)

        # Evaluation.
        if epoch % args.eval_freq == 0 or epoch == (args.epochs - 1):
            valid_stats = test(
                valid_queue, model, device,
                l2_loss_func, bce_loss_func,
                global_step, writer,
                visualizer, hand_mocap,
                task_names, args,
            )
            log_epoch_stats(valid_stats, writer, args, global_step, epoch, train=False)

        # Save model.
        if epoch % args.save_freq == 0 or epoch == (args.epochs - 1):
            print('saving the model.')
            torch.save(
                {'epoch': epoch, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'global_step': global_step,
                 'args': args},
                join(args.save, f'checkpoint_{epoch:04d}.pt')
            )

        epoch_end = time.time()
        epoch_elapsed = str(timedelta(seconds=round(epoch_end - epoch_start, 2))).split('.')[0]
        program_elapsed = str(timedelta(seconds=round(epoch_end - program_start, 2))).split('.')[0]
        print(f'Epoch elapsed time: {epoch_elapsed}')
        print(f'Program elapsed time: {program_elapsed}')

    # Final validation.
    print('\nFinal validation.')
    valid_stats = test(
        valid_queue, model, device,
        l2_loss_func, bce_loss_func,
        global_step, writer,
        visualizer, hand_mocap,
        task_names, args,
    )

    log_epoch_stats(valid_stats, writer, args, global_step, args.epochs, train=False)

    program_end = time.time()
    program_elapsed = str(timedelta(seconds=round(program_end - program_start, 2))).split('.')[0]
    print(f'\nDone. Program elapsed time: {program_elapsed}')


def train(
        train_queue, model, optimizer, device,
        l2_loss_func, bce_loss_func,
        global_step, writer,
        visualizer, hand_mocap,
        task_names, args
):
    model.train()

    epoch_loss = AvgrageMeter()
    epoch_xy_loss = AvgrageMeter()
    epoch_depth_loss = AvgrageMeter()
    epoch_ori_loss = AvgrageMeter()
    epoch_contact_loss = AvgrageMeter()
    epoch_contact_acc = AvgrageMeter()

    # using current hand info for prediction as baseline
    epoch_bl_loss = AvgrageMeter()
    epoch_bl_xy_loss = AvgrageMeter()
    epoch_bl_depth_loss = AvgrageMeter()
    epoch_bl_ori_loss = AvgrageMeter()
    epoch_bl_contact_loss = AvgrageMeter()
    epoch_bl_contact_acc = AvgrageMeter()

    # using mean of batch for prediction as baseline
    epoch_mean_loss = AvgrageMeter()
    epoch_mean_xy_loss = AvgrageMeter()
    epoch_mean_depth_loss = AvgrageMeter()
    epoch_mean_ori_loss = AvgrageMeter()
    epoch_mean_contact_loss = AvgrageMeter()
    epoch_mean_contact_acc = AvgrageMeter()

    epoch_metric_stats = {task_name: {'total': 0, 'pred_success': 0, 'gt_success': 0} for task_name in task_names}

    t0 = time.time()
    for step, data in tqdm(enumerate(train_queue), desc='Going through train data...'):
        if step == 0:
            t1 = time.time()
            print(f'\nDataloader iterator init time: {t1 - t0:.4f}s')
        # process batch data
        input = torch.cat((
            data.hand_r3m,
            data.task,
            data.current_x.unsqueeze(1),
            data.current_y.unsqueeze(1),
            data.current_depth.unsqueeze(1),
            data.current_ori,
            data.current_contact.unsqueeze(1),
        ), dim=1).to(device).float()

        current_x, current_y = data.current_x.to(device).float(), data.current_y.to(device).float()
        current_xy = torch.cat((current_x.unsqueeze(1), current_y.unsqueeze(1)), dim=1)
        current_depth = data.current_depth.to(device).float()
        current_ori = data.current_ori.to(device)
        current_contact = data.current_contact.to(device).float()

        future_x, future_y = data.future_x.to(device).float(), data.future_y.to(device).float()
        future_xy = torch.cat((future_x.unsqueeze(1), future_y.unsqueeze(1)), dim=1)
        future_depth = data.future_depth.to(device).float()
        future_ori = data.future_ori.to(device)
        future_contact = data.future_contact.to(device).float()

        # handle residual prediction mode
        target_xy = future_xy - current_xy if args.pred_residual else future_xy
        target_depth = future_depth - current_depth if args.pred_residual else future_depth
        target_ori = future_ori - current_ori if args.pred_residual else future_ori

        # forward through network
        optimizer.zero_grad()
        output = model(input)

        # process network output
        pred_xy, pred_depth = output[:, 0:2], output[:, 2]
        pred_ori, pred_contact = output[:, 3:6], output[:, 6]
        if not args.pred_residual:
            pred_xy = torch.sigmoid(pred_xy)  # force xy to be positive bbox coords

        # process loss
        xy_loss = l2_loss_func(pred_xy, target_xy)
        depth_loss = l2_loss_func(pred_depth, target_depth)
        ori_loss = l2_loss_func(pred_ori, target_ori)
        contact_loss = bce_loss_func(pred_contact, future_contact)
        loss = args.lambda1 * xy_loss + \
               args.lambda2 * depth_loss + \
               args.lambda3 * ori_loss + \
               args.lambda4 * contact_loss

        with torch.no_grad():
            pred_contact_binary = torch.round(torch.sigmoid(pred_contact))
            pred_contact_correct = (pred_contact_binary == future_contact).sum().float()
            contact_acc = pred_contact_correct / pred_contact.size(0)

        # process baseline loss
        batch_size = target_xy.size(0)
        with torch.no_grad():
            bl_xy = torch.zeros_like(current_xy) if args.pred_residual else current_xy
            bl_depth = torch.zeros_like(current_depth) if args.pred_residual else current_depth
            bl_ori = torch.zeros_like(current_ori) if args.pred_residual else current_ori

            bl_xy_loss = l2_loss_func(bl_xy, target_xy)
            bl_depth_loss = l2_loss_func(bl_depth, target_depth)
            bl_ori_loss = l2_loss_func(bl_ori, target_ori)
            bl_contact_loss = bce_loss_func(current_contact, future_contact)
            bl_loss = args.lambda1 * bl_xy_loss + \
                      args.lambda2 * bl_depth_loss + \
                      args.lambda3 * bl_ori_loss + \
                      args.lambda4 * bl_contact_loss

            bl_contact_correct = (current_contact == future_contact).sum().float()
            bl_contact_acc = bl_contact_correct / batch_size

            mean_xy = torch.mean(target_xy, dim=0).unsqueeze(0).repeat(batch_size, 1)
            mean_depth = torch.mean(target_depth).repeat(batch_size)
            mean_ori = torch.mean(target_ori, dim=0).unsqueeze(0).repeat(batch_size, 1)
            mean_contact = torch.mean(future_contact).repeat(batch_size)

            mean_xy_loss = l2_loss_func(mean_xy, target_xy)
            mean_depth_loss = l2_loss_func(mean_depth, target_depth)
            mean_ori_loss = l2_loss_func(mean_ori, target_ori)
            mean_contact_loss = bce_loss_func(mean_contact, future_contact)
            mean_loss = args.lambda1 * mean_xy_loss + \
                        args.lambda2 * mean_depth_loss + \
                        args.lambda3 * mean_ori_loss + \
                        args.lambda4 * mean_contact_loss

            mean_contact_binary = torch.round(torch.sigmoid(mean_contact))
            mean_contact_correct = (mean_contact_binary == future_contact).sum().float()
            mean_contact_acc = mean_contact_correct / batch_size

        # back propogate
        loss.backward()
        optimizer.step()

        # process metric evaluation
        metric_stats = evaluate_transferable_metric_batch(
            task_names=task_names,
            task=data.task,
            device=device,
            current_x=current_x,
            pred_x=current_x + pred_xy[:, 0] if args.pred_residual else pred_xy[:, 0],
            future_x=future_x,
            current_y=current_y,
            pred_y=current_y + pred_xy[:, 1] if args.pred_residual else pred_xy[:, 1],
            future_y=future_y,
            current_depth=current_depth,
            pred_depth=current_depth + pred_depth if args.pred_residual else pred_depth,
            future_depth=future_depth,
            evaluate_gt=True
        )
        for k, v in metric_stats.items():
            epoch_metric_stats[k]['total'] += v['total']
            epoch_metric_stats[k]['pred_success'] += v['pred_success']
            epoch_metric_stats[k]['gt_success'] += v['gt_success']

        # update epoch average meters
        epoch_loss.update(loss.data, 1)
        epoch_xy_loss.update(xy_loss.data, 1)
        epoch_depth_loss.update(depth_loss.data, 1)
        epoch_ori_loss.update(ori_loss.data, 1)
        epoch_contact_loss.update(contact_loss.data, 1)
        epoch_contact_acc.update(contact_acc.data, 1)

        epoch_bl_loss.update(bl_loss.data, 1)
        epoch_bl_xy_loss.update(bl_xy_loss.data, 1)
        epoch_bl_depth_loss.update(bl_depth_loss.data, 1)
        epoch_bl_ori_loss.update(bl_ori_loss.data, 1)
        epoch_bl_contact_loss.update(bl_contact_loss.data, 1)
        epoch_bl_contact_acc.update(bl_contact_acc.data, 1)

        epoch_mean_loss.update(mean_loss.data, 1)
        epoch_mean_xy_loss.update(mean_xy_loss.data, 1)
        epoch_mean_depth_loss.update(mean_depth_loss.data, 1)
        epoch_mean_ori_loss.update(mean_ori_loss.data, 1)
        epoch_mean_contact_loss.update(mean_contact_loss.data, 1)
        epoch_mean_contact_acc.update(mean_contact_acc.data, 1)

        # log scalars
        if (global_step + 1) % args.log_scalar_freq == 0:
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/xy_loss', xy_loss, global_step)
            writer.add_scalar('train/depth_loss', depth_loss, global_step)
            writer.add_scalar('train/ori_loss', ori_loss, global_step)
            writer.add_scalar('train/contact_loss', contact_loss, global_step)
            writer.add_scalar('train/contact_acc', contact_acc, global_step)

            writer.add_scalar('train_baseline/current_loss', bl_loss, global_step)
            writer.add_scalar('train_baseline/current_xy_loss', bl_xy_loss, global_step)
            writer.add_scalar('train_baseline/current_depth_loss', bl_depth_loss, global_step)
            writer.add_scalar('train_baseline/current_ori_loss', bl_ori_loss, global_step)
            writer.add_scalar('train_baseline/current_contact_loss', bl_contact_loss, global_step)
            writer.add_scalar('train_baseline/current_contact_acc', bl_contact_acc, global_step)

            writer.add_scalar('train_baseline/mean_loss', mean_loss, global_step)
            writer.add_scalar('train_baseline/mean_xy_loss', mean_xy_loss, global_step)
            writer.add_scalar('train_baseline/mean_depth_loss', mean_depth_loss, global_step)
            writer.add_scalar('train_baseline/mean_ori_loss', mean_ori_loss, global_step)
            writer.add_scalar('train_baseline/mean_contact_loss', mean_contact_loss, global_step)
            writer.add_scalar('train_baseline/mean_contact_acc', mean_contact_acc, global_step)

        # log images
        if (global_step + 1) % args.vis_freq == 0 and not args.sanity_check:
            # visualize some samples in the batch
            vis_imgs = []
            for i in range(args.vis_sample_size):
                pred_delta_x = pred_xy[i, 0] if args.pred_residual else pred_xy[i, 0] - current_x[i]
                pred_delta_y = pred_xy[i, 1] if args.pred_residual else pred_xy[i, 1] - current_y[i]
                pred_delta_depth = pred_depth[i] if args.pred_residual else pred_depth[i] - current_depth[i]
                pred_delta_ori = pred_ori[i] if args.pred_residual else pred_ori[i] - current_ori[i]
                task_name = task_names[torch.argmax(data.task[i].squeeze())]

                passed_metric = evaluate_transferable_metric(
                    task_name=task_name,
                    current_x=current_x[i],
                    pred_x=current_x[i] + pred_xy[i, 0] if args.pred_residual else pred_xy[i, 0],
                    current_y=current_y[i],
                    pred_y=current_y[i] + pred_xy[i, 1] if args.pred_residual else pred_xy[i, 1],
                    current_depth=current_depth[i],
                    pred_depth=current_depth[i] + pred_depth[i] if args.pred_residual else pred_depth[i],
                )

                vis_img = generate_transferable_visualization(
                    current_hand_pose_path=data.current_info_path[i],
                    future_hand_pose_path=data.future_info_path[i],
                    run_on_cv_server=args.run_on_cv_server,
                    hand=data.hand[i],
                    pred_delta_x=pred_delta_x.item(),
                    pred_delta_y=pred_delta_y.item(),
                    pred_delta_depth=pred_delta_depth.item(),
                    pred_delta_ori=pred_delta_ori,
                    pred_contact=pred_contact_binary[i],
                    depth_norm_params=args.depth_norm_params,
                    ori_norm_params=args.ori_norm_params,
                    task_name=task_name,
                    visualizer=visualizer,
                    use_visualizer=args.use_visualizer,
                    hand_mocap=hand_mocap,
                    device=device,
                    log_metric=True,
                    passed_metric=passed_metric.item()
                )
                vis_imgs.append(vis_img)
            final_vis_img = np.hstack(vis_imgs)
            writer.add_image(f'train/vis_images', final_vis_img, global_step, dataformats='HWC')

            # visualize different task conditioning
            if args.eval_tasks:
                for i in range(args.task_vis_sample_size):
                    # process task input
                    original_task_name = task_names[torch.argmax(data.task[i].squeeze())]
                    all_task_instances = []
                    for j, task_name in enumerate(task_names):
                        task_instance = torch.zeros(1, len(task_names))
                        task_instance[0, j] = 1
                        all_task_instances.append(task_instance)
                    all_task_instances = torch.vstack(all_task_instances)

                    task_conditioned_input = torch.cat((
                        data.hand_r3m[i].repeat(len(task_names), 1),
                        all_task_instances,
                        data.current_x[i].repeat(len(task_names), 1),
                        data.current_y[i].repeat(len(task_names), 1),
                        data.current_depth[i].repeat(len(task_names), 1),
                        data.current_ori[i].repeat(len(task_names), 1),
                        data.current_contact[i].repeat(len(task_names), 1),
                    ), dim=1).to(device).float()

                    # forward through network and process output
                    model.eval()
                    with torch.no_grad():
                        task_conditioned_output = model(task_conditioned_input)
                        t_pred_xy, t_pred_depth = task_conditioned_output[:, 0:2], task_conditioned_output[:, 2]
                        t_pred_ori, t_pred_contact = task_conditioned_output[:, 3:6], task_conditioned_output[:, 6]
                        if not args.pred_residual:
                            t_pred_xy = torch.sigmoid(t_pred_xy)  # force xy to be positive bbox coords

                    task_vis_imgs = []
                    for j, task_name in enumerate(task_names):
                        pred_delta_x = t_pred_xy[j, 0] if args.pred_residual else t_pred_xy[j, 0] - current_x[i]
                        pred_delta_y = t_pred_xy[j, 1] if args.pred_residual else t_pred_xy[j, 1] - current_y[i]
                        pred_delta_depth = t_pred_depth[j] if args.pred_residual else t_pred_depth[j] - current_depth[i]
                        pred_delta_ori = t_pred_ori[j] if args.pred_residual else t_pred_ori[j] - current_ori[i]
                        pred_contact_binary = torch.round(torch.sigmoid(t_pred_contact[j]))

                        passed_metric = evaluate_transferable_metric(
                            task_name=task_name,
                            current_x=current_x[i],
                            pred_x=current_x[i] + t_pred_xy[j, 0] if args.pred_residual else t_pred_xy[j, 0],
                            current_y=current_y[i],
                            pred_y=current_y[i] + t_pred_xy[j, 1] if args.pred_residual else t_pred_xy[j, 1],
                            current_depth=current_depth[i],
                            pred_depth=current_depth[i] + t_pred_depth[j] if args.pred_residual else t_pred_depth[j],
                        )

                        task_vis_img = generate_transferable_visualization(
                            current_hand_pose_path=data.current_info_path[i],
                            future_hand_pose_path=None,
                            run_on_cv_server=args.run_on_cv_server,
                            hand=data.hand[i],
                            pred_delta_x=pred_delta_x.item(),
                            pred_delta_y=pred_delta_y.item(),
                            pred_delta_depth=pred_delta_depth.item(),
                            pred_delta_ori=pred_delta_ori,
                            pred_contact=pred_contact_binary,
                            depth_norm_params=args.depth_norm_params,
                            ori_norm_params=args.ori_norm_params,
                            task_name=task_name,
                            visualizer=visualizer,
                            use_visualizer=args.use_visualizer,
                            hand_mocap=hand_mocap,
                            device=device,
                            log_metric=True,
                            passed_metric=passed_metric.item(),
                            original_task=task_name == original_task_name,
                            vis_groundtruth=False
                        )
                        task_vis_imgs.append(task_vis_img)
                    final_task_vis_img = np.hstack(task_vis_imgs)
                    writer.add_image(
                        f'train_eval_tasks/vis_tasks_{i}', final_task_vis_img, global_step, dataformats='HWC'
                    )

            model.train()

        global_step += 1

    stats = namedtuple('stats', [
        'loss', 'xy_loss', 'depth_loss', 'ori_loss', 'contact_loss', 'contact_acc',
        'bl_loss', 'bl_xy_loss', 'bl_depth_loss', 'bl_ori_loss', 'bl_contact_loss', 'bl_contact_acc',
        'mean_loss', 'mean_xy_loss', 'mean_depth_loss', 'mean_ori_loss', 'mean_contact_loss', 'mean_contact_acc',
        'global_step', 'epoch_metric_stats'
    ])

    return stats(
        loss=epoch_loss.avg,
        xy_loss=epoch_xy_loss.avg,
        depth_loss=epoch_depth_loss.avg,
        ori_loss=epoch_ori_loss.avg,
        contact_loss=epoch_contact_loss.avg,
        contact_acc=epoch_contact_acc.avg,
        bl_loss=epoch_bl_loss.avg,
        bl_xy_loss=epoch_bl_xy_loss.avg,
        bl_depth_loss=epoch_bl_depth_loss.avg,
        bl_ori_loss=epoch_bl_ori_loss.avg,
        bl_contact_loss=epoch_bl_contact_loss.avg,
        bl_contact_acc=epoch_bl_contact_acc.avg,
        mean_loss=epoch_mean_loss.avg,
        mean_xy_loss=epoch_mean_xy_loss.avg,
        mean_depth_loss=epoch_mean_depth_loss.avg,
        mean_ori_loss=epoch_mean_ori_loss.avg,
        mean_contact_loss=epoch_mean_contact_loss.avg,
        mean_contact_acc=epoch_mean_contact_acc.avg,
        global_step=global_step,
        epoch_metric_stats=epoch_metric_stats
    )


def test(
        valid_queue, model, device,
        l2_loss_func, bce_loss_func,
        global_step, writer,
        visualizer, hand_mocap,
        task_names, args
):
    model.eval()

    epoch_loss = AvgrageMeter()
    epoch_xy_loss = AvgrageMeter()
    epoch_depth_loss = AvgrageMeter()
    epoch_ori_loss = AvgrageMeter()
    epoch_contact_loss = AvgrageMeter()
    epoch_contact_acc = AvgrageMeter()

    # using current hand info for prediction as baseline
    epoch_bl_loss = AvgrageMeter()
    epoch_bl_xy_loss = AvgrageMeter()
    epoch_bl_depth_loss = AvgrageMeter()
    epoch_bl_ori_loss = AvgrageMeter()
    epoch_bl_contact_loss = AvgrageMeter()
    epoch_bl_contact_acc = AvgrageMeter()

    # using mean of batch for prediction as baseline
    epoch_mean_loss = AvgrageMeter()
    epoch_mean_xy_loss = AvgrageMeter()
    epoch_mean_depth_loss = AvgrageMeter()
    epoch_mean_ori_loss = AvgrageMeter()
    epoch_mean_contact_loss = AvgrageMeter()
    epoch_mean_contact_acc = AvgrageMeter()

    data, current_x, current_y, current_depth, current_ori = None, None, None, None, None
    epoch_metric_stats = {task_name: {'total': 0, 'pred_success': 0, 'gt_success': 0} for task_name in task_names}
    task_metric_stats = None

    for step, data in tqdm(enumerate(valid_queue), 'Going through valid data...'):
        # process batch data
        input = torch.cat((
            data.hand_r3m,
            data.task,
            data.current_x.unsqueeze(1),
            data.current_y.unsqueeze(1),
            data.current_depth.unsqueeze(1),
            data.current_ori,
            data.current_contact.unsqueeze(1),
        ), dim=1).to(device).float()

        current_x, current_y = data.current_x.to(device).float(), data.current_y.to(device).float()
        current_xy = torch.cat((current_x.unsqueeze(1), current_y.unsqueeze(1)), dim=1)
        current_depth = data.current_depth.to(device).float()
        current_ori = data.current_ori.to(device)
        current_contact = data.current_contact.to(device).float()

        future_x, future_y = data.future_x.to(device).float(), data.future_y.to(device).float()
        future_xy = torch.cat((future_x.unsqueeze(1), future_y.unsqueeze(1)), dim=1)
        future_depth = data.future_depth.to(device).float()
        future_ori = data.future_ori.to(device)
        future_contact = data.future_contact.to(device).float()

        # handle residual prediction mode
        target_xy = future_xy - current_xy if args.pred_residual else future_xy
        target_depth = future_depth - current_depth if args.pred_residual else future_depth
        target_ori = future_ori - current_ori if args.pred_residual else future_ori

        batch_size = target_xy.size(0)
        with torch.no_grad():
            # forward through model and process output
            output = model(input)
            pred_xy, pred_depth = output[:, 0:2], output[:, 2]
            pred_ori, pred_contact = output[:, 3:6], output[:, 6]
            if not args.pred_residual:
                pred_xy = torch.sigmoid(pred_xy)  # force xy to be positive bbox coords

            # process loss
            xy_loss = l2_loss_func(pred_xy, target_xy)
            depth_loss = l2_loss_func(pred_depth, target_depth)
            ori_loss = l2_loss_func(pred_ori, target_ori)
            contact_loss = bce_loss_func(pred_contact, future_contact)
            loss = args.lambda1 * xy_loss + \
                   args.lambda2 * depth_loss + \
                   args.lambda3 * ori_loss + \
                   args.lambda4 * contact_loss

            pred_contact_binary = torch.round(torch.sigmoid(pred_contact))
            pred_contact_correct = (pred_contact_binary == future_contact).sum().float()
            contact_acc = pred_contact_correct / pred_contact.size(0)

            # process baseline loss
            bl_xy = torch.zeros_like(current_xy) if args.pred_residual else current_xy
            bl_depth = torch.zeros_like(current_depth) if args.pred_residual else current_depth
            bl_ori = torch.zeros_like(current_ori) if args.pred_residual else current_ori

            bl_xy_loss = l2_loss_func(bl_xy, target_xy)
            bl_depth_loss = l2_loss_func(bl_depth, target_depth)
            bl_ori_loss = l2_loss_func(bl_ori, target_ori)
            bl_contact_loss = bce_loss_func(current_contact, future_contact)
            bl_loss = args.lambda1 * bl_xy_loss + \
                      args.lambda2 * bl_depth_loss + \
                      args.lambda3 * bl_ori_loss + \
                      args.lambda4 * bl_contact_loss

            bl_contact_correct = (current_contact == future_contact).sum().float()
            bl_contact_acc = bl_contact_correct / current_contact.size(0)

            mean_xy = torch.mean(target_xy, dim=0).unsqueeze(0).repeat(batch_size, 1)
            mean_depth = torch.mean(target_depth).repeat(batch_size)
            mean_ori = torch.mean(target_ori, dim=0).unsqueeze(0).repeat(batch_size, 1)
            mean_contact = torch.mean(future_contact).repeat(batch_size)

            mean_xy_loss = l2_loss_func(mean_xy, target_xy)
            mean_depth_loss = l2_loss_func(mean_depth, target_depth)
            mean_ori_loss = l2_loss_func(mean_ori, target_ori)
            mean_contact_loss = bce_loss_func(mean_contact, future_contact)
            mean_loss = args.lambda1 * mean_xy_loss + \
                        args.lambda2 * mean_depth_loss + \
                        args.lambda3 * mean_ori_loss + \
                        args.lambda4 * mean_contact_loss

            mean_contact_binary = torch.round(torch.sigmoid(mean_contact))
            mean_contact_correct = (mean_contact_binary == future_contact).sum().float()
            mean_contact_acc = mean_contact_correct / batch_size

            # process metric evaluation
            metric_stats = evaluate_transferable_metric_batch(
                task_names=task_names,
                task=data.task,
                device=device,
                current_x=current_x,
                pred_x=current_x + pred_xy[:, 0] if args.pred_residual else pred_xy[:, 0],
                future_x=future_x,
                current_y=current_y,
                pred_y=current_y + pred_xy[:, 1] if args.pred_residual else pred_xy[:, 1],
                future_y=future_y,
                current_depth=current_depth,
                pred_depth=current_depth + pred_depth if args.pred_residual else pred_depth,
                future_depth=future_depth,
                evaluate_gt=True
            )
            for k, v in metric_stats.items():
                epoch_metric_stats[k]['total'] += v['total']
                epoch_metric_stats[k]['pred_success'] += v['pred_success']
                epoch_metric_stats[k]['gt_success'] += v['gt_success']

        # update epoch average meters
        batch_size = input.size(0)
        epoch_loss.update(loss.data, batch_size)
        epoch_xy_loss.update(xy_loss.data, batch_size)
        epoch_depth_loss.update(depth_loss.data, batch_size)
        epoch_ori_loss.update(ori_loss.data, batch_size)
        epoch_contact_loss.update(contact_loss.data, batch_size)
        epoch_contact_acc.update(contact_acc.data, batch_size)

        epoch_bl_loss.update(bl_loss.data, batch_size)
        epoch_bl_xy_loss.update(bl_xy_loss.data, batch_size)
        epoch_bl_depth_loss.update(bl_depth_loss.data, batch_size)
        epoch_bl_ori_loss.update(bl_ori_loss.data, batch_size)
        epoch_bl_contact_loss.update(bl_contact_loss.data, batch_size)
        epoch_bl_contact_acc.update(bl_contact_acc.data, batch_size)

        epoch_mean_loss.update(mean_loss.data, 1)
        epoch_mean_xy_loss.update(mean_xy_loss.data, 1)
        epoch_mean_depth_loss.update(mean_depth_loss.data, 1)
        epoch_mean_ori_loss.update(mean_ori_loss.data, 1)
        epoch_mean_contact_loss.update(mean_contact_loss.data, 1)
        epoch_mean_contact_acc.update(mean_contact_acc.data, 1)

    # visualize some samples in the last batch
    vis_imgs = []
    for i in range(args.vis_sample_size):
        pred_delta_x = pred_xy[i, 0] if args.pred_residual else pred_xy[i, 0] - current_x[i]
        pred_delta_y = pred_xy[i, 1] if args.pred_residual else pred_xy[i, 1] - current_y[i]
        pred_delta_depth = pred_depth[i] if args.pred_residual else pred_depth[i] - current_depth[i]
        pred_delta_ori = pred_ori[i] if args.pred_residual else pred_ori[i] - current_ori[i]
        task_name = task_names[torch.argmax(data.task[i].squeeze())]

        passed_metric = evaluate_transferable_metric(
            task_name=task_name,
            current_x=current_x[i],
            pred_x=current_x[i] + pred_xy[i, 0] if args.pred_residual else pred_xy[i, 0],
            current_y=current_y[i],
            pred_y=current_y[i] + pred_xy[i, 1] if args.pred_residual else pred_xy[i, 1],
            current_depth=current_depth[i],
            pred_depth=current_depth[i] + pred_depth[i] if args.pred_residual else pred_depth[i],
        )

        vis_img = generate_transferable_visualization(
            current_hand_pose_path=data.current_info_path[i],
            future_hand_pose_path=data.future_info_path[i],
            run_on_cv_server=args.run_on_cv_server,
            hand=data.hand[i],
            pred_delta_x=pred_delta_x.item(),
            pred_delta_y=pred_delta_y.item(),
            pred_delta_depth=pred_delta_depth.item(),
            pred_delta_ori=pred_delta_ori,
            pred_contact=pred_contact_binary[i],
            depth_norm_params=args.depth_norm_params,
            ori_norm_params=args.ori_norm_params,
            task_name=task_name,
            visualizer=visualizer,
            use_visualizer=args.use_visualizer,
            hand_mocap=hand_mocap,
            device=device,
            log_metric=True,
            passed_metric=passed_metric.item()
        )
        vis_imgs.append(vis_img)
    final_vis_img = np.hstack(vis_imgs)
    writer.add_image(f'valid/vis_images', final_vis_img, global_step, dataformats='HWC')

    # task-conditioned evaluation
    if args.eval_tasks:
        task_vis_sample_count = 0
        task_metric_stats = {task_name: {'total': 0, 'pred_success': 0} for task_name in task_names}
        all_task_instances = []
        for j, task_name in enumerate(task_names):
            task_instance = torch.zeros(1, len(task_names))
            task_instance[0, j] = 1
            all_task_instances.append(task_instance)
        all_task_instances = torch.vstack(all_task_instances)
        for i in range(len(data)):
            original_task_name = task_names[torch.argmax(data.task[i].squeeze())]
            task_conditioned_input = torch.cat((
                data.hand_r3m[i].repeat(len(task_names), 1),
                all_task_instances,
                data.current_x[i].repeat(len(task_names), 1),
                data.current_y[i].repeat(len(task_names), 1),
                data.current_depth[i].repeat(len(task_names), 1),
                data.current_ori[i].repeat(len(task_names), 1),
                data.current_contact[i].repeat(len(task_names), 1),
            ), dim=1).to(device).float()

            with torch.no_grad():
                task_conditioned_output = model(task_conditioned_input)
                t_pred_xy, t_pred_depth = task_conditioned_output[:, 0:2], task_conditioned_output[:, 2]
                t_pred_ori, t_pred_contact = task_conditioned_output[:, 3:6], task_conditioned_output[:, 6]
                if not args.pred_residual:
                    t_pred_xy = torch.sigmoid(t_pred_xy)  # force xy to be positive bbox coords

            # process metric evaluation
            t_current_x = data.current_x[i].repeat(len(task_names)).to(device).float()
            t_current_y = data.current_y[i].repeat(len(task_names)).to(device).float()
            t_current_depth = data.current_depth[i].repeat(len(task_names)).to(device).float()

            metric_stats = evaluate_transferable_metric_batch(
                task_names=task_names,
                task=all_task_instances,
                device=device,
                current_x=t_current_x,
                pred_x=t_current_x + t_pred_xy[:, 0] if args.pred_residual else t_pred_xy[:, 0],
                current_y=t_current_y,
                pred_y=t_current_y + t_pred_xy[:, 1] if args.pred_residual else t_pred_xy[:, 1],
                current_depth=t_current_depth,
                pred_depth=t_current_depth + t_pred_depth if args.pred_residual else t_pred_depth,
                evaluate_gt=False
            )
            for k, v in metric_stats.items():
                task_metric_stats[k]['total'] += v['total']
                task_metric_stats[k]['pred_success'] += v['pred_success']

            # visualize some samples of task-conditioned evaluation
            if task_vis_sample_count < args.task_vis_sample_size:
                task_vis_imgs = []
                for j, task_name in enumerate(task_names):
                    pred_delta_x = t_pred_xy[j, 0] if args.pred_residual else t_pred_xy[j, 0] - current_x[i]
                    pred_delta_y = t_pred_xy[j, 1] if args.pred_residual else t_pred_xy[j, 1] - current_y[i]
                    pred_delta_depth = t_pred_depth[j] if args.pred_residual else t_pred_depth[j] - current_depth[i]
                    pred_delta_ori = t_pred_ori[j] if args.pred_residual else t_pred_ori[j] - current_ori[i]
                    pred_contact_binary = torch.round(torch.sigmoid(t_pred_contact[j]))

                    passed_metric = evaluate_transferable_metric(
                        task_name=task_name,
                        current_x=current_x[i],
                        pred_x=current_x[i] + t_pred_xy[j, 0] if args.pred_residual else t_pred_xy[j, 0],
                        current_y=current_y[i],
                        pred_y=current_y[i] + t_pred_xy[j, 1] if args.pred_residual else t_pred_xy[j, 1],
                        current_depth=current_depth[i],
                        pred_depth=current_depth[i] + t_pred_depth[j] if args.pred_residual else t_pred_depth[j],
                    )

                    task_vis_img = generate_transferable_visualization(
                        current_hand_pose_path=data.current_info_path[i],
                        future_hand_pose_path=None,
                        run_on_cv_server=args.run_on_cv_server,
                        hand=data.hand[i],
                        pred_delta_x=pred_delta_x.item(),
                        pred_delta_y=pred_delta_y.item(),
                        pred_delta_depth=pred_delta_depth.item(),
                        pred_delta_ori=pred_delta_ori,
                        pred_contact=pred_contact_binary,
                        depth_norm_params=args.depth_norm_params,
                        ori_norm_params=args.ori_norm_params,
                        task_name=task_name,
                        visualizer=visualizer,
                        use_visualizer=args.use_visualizer,
                        hand_mocap=hand_mocap,
                        device=device,
                        log_metric=True,
                        passed_metric=passed_metric.item(),
                        original_task=task_name == original_task_name,
                        vis_groundtruth=False
                    )
                    task_vis_imgs.append(task_vis_img)
                final_task_vis_img = np.hstack(task_vis_imgs)
                writer.add_image(f'valid_eval_tasks/vis_tasks_{i}', final_task_vis_img, global_step, dataformats='HWC')
                task_vis_sample_count += 1

    stats = namedtuple('stats', [
        'loss', 'xy_loss', 'depth_loss', 'ori_loss', 'contact_loss', 'contact_acc',
        'bl_loss', 'bl_xy_loss', 'bl_depth_loss', 'bl_ori_loss', 'bl_contact_loss', 'bl_contact_acc',
        'mean_loss', 'mean_xy_loss', 'mean_depth_loss', 'mean_ori_loss', 'mean_contact_loss', 'mean_contact_acc',
        'epoch_metric_stats', 'task_metric_stats'
    ])

    return stats(
        loss=epoch_loss.avg,
        xy_loss=epoch_xy_loss.avg,
        depth_loss=epoch_depth_loss.avg,
        ori_loss=epoch_ori_loss.avg,
        contact_loss=epoch_contact_loss.avg,
        contact_acc=epoch_contact_acc.avg,
        bl_loss=epoch_bl_loss.avg,
        bl_xy_loss=epoch_bl_xy_loss.avg,
        bl_depth_loss=epoch_bl_depth_loss.avg,
        bl_ori_loss=epoch_bl_ori_loss.avg,
        bl_contact_loss=epoch_bl_contact_loss.avg,
        bl_contact_acc=epoch_bl_contact_acc.avg,
        mean_loss=epoch_mean_loss.avg,
        mean_xy_loss=epoch_mean_xy_loss.avg,
        mean_depth_loss=epoch_mean_depth_loss.avg,
        mean_ori_loss=epoch_mean_ori_loss.avg,
        mean_contact_loss=epoch_mean_contact_loss.avg,
        mean_contact_acc=epoch_mean_contact_acc.avg,
        epoch_metric_stats=epoch_metric_stats,
        task_metric_stats=task_metric_stats
    )


def log_epoch_stats(stats, writer, args, global_step, epoch, train=True):
    mode = 'train' if train else 'valid'
    metric_stats = stats.epoch_metric_stats
    task_metric_stats = stats.task_metric_stats if not train else None

    # log train epoch stats
    print(f'epoch {mode} loss: {stats.loss}')
    print(f'epoch {mode} xy loss: {stats.xy_loss}')
    print(f'epoch {mode} depth loss: {stats.depth_loss}')
    print(f'epoch {mode} orientation loss: {stats.ori_loss}')
    print(f'epoch {mode} contact loss: {stats.contact_loss}')
    print(f'epoch {mode} contact acc: {stats.contact_acc}')

    writer.add_scalar(f'{mode}_epoch/loss', stats.loss, epoch)
    writer.add_scalar(f'{mode}_epoch/xy_loss', stats.xy_loss, epoch)
    writer.add_scalar(f'{mode}_epoch/depth_loss', stats.depth_loss, epoch)
    writer.add_scalar(f'{mode}_epoch/ori_loss', stats.ori_loss, epoch)
    writer.add_scalar(f'{mode}_epoch/contact_loss', stats.contact_loss, epoch)
    writer.add_scalar(f'{mode}_epoch/contact_acc', stats.contact_acc, epoch)

    writer.add_scalar(f'{mode}_epoch_baseline/current_loss', stats.bl_loss, epoch)
    writer.add_scalar(f'{mode}_epoch_baseline/current_xy_loss', stats.bl_xy_loss, epoch)
    writer.add_scalar(f'{mode}_epoch_baseline/current_depth_loss', stats.bl_depth_loss, epoch)
    writer.add_scalar(f'{mode}_epoch_baseline/current_ori_loss', stats.bl_ori_loss, epoch)
    writer.add_scalar(f'{mode}_epoch_baseline/current_contact_loss', stats.bl_contact_loss, epoch)
    writer.add_scalar(f'{mode}_epoch_baseline/current_contact_acc', stats.bl_contact_acc, epoch)

    writer.add_scalar(f'{mode}_epoch_baseline/mean_loss', stats.mean_loss, epoch)
    writer.add_scalar(f'{mode}_epoch_baseline/mean_xy_loss', stats.mean_xy_loss, epoch)
    writer.add_scalar(f'{mode}_epoch_baseline/mean_depth_loss', stats.mean_depth_loss, epoch)
    writer.add_scalar(f'{mode}_epoch_baseline/mean_ori_loss', stats.mean_ori_loss, epoch)
    writer.add_scalar(f'{mode}_epoch_baseline/mean_contact_loss', stats.mean_contact_loss, epoch)
    writer.add_scalar(f'{mode}_epoch_baseline/mean_contact_acc', stats.mean_contact_acc, epoch)

    if train:
        writer.add_scalar('stats/epoch_steps', global_step, epoch)
        writer.add_scalar('stats/lambda1', args.lambda1, epoch)
        writer.add_scalar('stats/lambda2', args.lambda2, epoch)
        writer.add_scalar('stats/lambda3', args.lambda3, epoch)
        writer.add_scalar('stats/lambda4', args.lambda4, epoch)
        writer.add_scalar('stats/bce_pos_weight', args.bce_pos_weight, epoch)

    # visualize metric evaluation success rate by bar plot
    bar_width = 0.4
    fig_height = 10
    n_task = len(args.task_names)

    fig = plt.figure(figsize=(2 * n_task , fig_height))
    bar1 = np.arange(len(metric_stats))
    bar2 = [x + bar_width for x in bar1]
    epoch_pred_success_rate = [
        v['pred_success'] / v['total'] if v['total'] != 0 else 0
        for v in metric_stats.values()
    ]
    epoch_gt_success_rate = [
        v['gt_success'] / v['total'] if v['total'] != 0 else 0
        for v in metric_stats.values()
    ]

    plt.bar(bar1, epoch_pred_success_rate, width=bar_width, color='peachpuff', label='pred')
    plt.bar(bar2, epoch_gt_success_rate, width=bar_width, color='lavender', label='gt')
    for b1, ps, b2, gs in zip(bar1, epoch_pred_success_rate, bar2, epoch_gt_success_rate):
        plt.text(b1, ps / 2, f'{ps:.2f}', ha='center', fontsize=13)
        plt.text(b2, gs / 2, f'{gs:.2f}', ha='center', fontsize=13)
    plt.xlabel('Tasks', fontweight='bold', size=13)
    plt.ylabel('Success rate', fontweight='bold', size=13)
    plt.xticks([r + bar_width / 2 for r in bar1], list(metric_stats.keys()))
    plt.ylim([0, 1])
    plt.title('1D Metric Evaluation')
    plt.legend()
    plt.tight_layout()
    writer.add_figure(f'{mode}_metric/success_rate', fig, epoch)
    plt.close(fig)

    # visualize metric evaluation count by bar plot
    bar_width = 0.25
    fig = plt.figure(figsize=(3 * n_task, fig_height))
    pred_success = [v['pred_success'] for v in metric_stats.values()]
    gt_success = [v['gt_success'] for v in metric_stats.values()]
    total = [v['total'] for v in metric_stats.values()]
    bar1 = np.arange(len(metric_stats))
    bar2 = [x + bar_width for x in bar1]
    bar3 = [x + bar_width for x in bar2]

    plt.bar(bar1, pred_success, color='peachpuff', width=bar_width, label='pred success')
    plt.bar(bar2, gt_success, color='lavender', width=bar_width, label='gt success')
    plt.bar(bar3, total, color='skyblue', width=bar_width, label='total')
    for b1, ps, b2, gs, b3, t in zip(bar1, pred_success, bar2, gt_success, bar3, total):
        plt.text(b1, ps / 2, ps, ha='center', fontsize=8)
        plt.text(b2, gs / 2, gs, ha='center', fontsize=8)
        plt.text(b3, t / 2, t, ha='center', fontsize=8)

    plt.xlabel('Tasks', fontweight='bold', size=13)
    plt.ylabel('Count', fontweight='bold', size=13)
    plt.xticks([r + bar_width for r in bar1], list(metric_stats.keys()))
    plt.title('1D Metric Evaluation')
    plt.legend()
    plt.tight_layout()
    writer.add_figure(f'{mode}_metric/count', fig, epoch)
    plt.close(fig)

    if not train and args.eval_tasks:
        # visualize metric evaluation success rate by bar plot
        bar_width = 0.7
        fig = plt.figure(figsize=(2 * n_task, fig_height))
        epoch_success_rate = [
            v['pred_success'] / v['total'] if v['total'] != 0 else 0
            for v in task_metric_stats.values()
        ]
        plt.bar(list(task_metric_stats.keys()), epoch_success_rate, width=bar_width, color='skyblue')
        for i, r in enumerate(epoch_success_rate):
            plt.text(i, r / 2, f'{r:.2f}', ha='center', fontsize=15)
        plt.xlabel('Tasks', fontweight='bold', size=13)
        plt.ylabel('Success rate', fontweight='bold', size=13)
        plt.ylim([0, 1])
        plt.title('1D Metric Evaluation')
        plt.tight_layout()
        writer.add_figure(f'{mode}_eval_tasks_metric/task_conditioning_success_rate', fig, epoch)
        plt.close(fig)

        # visualize metric evaluation count by bar plot
        bar_width = 0.35
        fig = plt.figure(figsize=(3 * n_task, fig_height))

        success = [v['pred_success'] for v in task_metric_stats.values()]
        total = [v['total'] for v in task_metric_stats.values()]
        bar1 = np.arange(len(task_metric_stats))
        bar2 = [x + bar_width for x in bar1]

        plt.bar(bar1, success, color='peachpuff', width=bar_width, label='success')
        plt.bar(bar2, total, color='lavender', width=bar_width, label='total')
        for b1, s, b2, t in zip(bar1, success, bar2, total):
            plt.text(b1, s / 2, s, ha='center', fontsize=8)
            plt.text(b2, t / 2, t, ha='center', fontsize=8)

        plt.xlabel('Tasks', fontweight='bold', size=13)
        plt.ylabel('Count', fontweight='bold', size=13)
        plt.xticks([r + bar_width / 2 for r in bar1], list(task_metric_stats.keys()))
        plt.legend()
        plt.tight_layout()
        writer.add_figure(f'{mode}_eval_tasks_metric/task_conditioning_count', fig, epoch)
        plt.close(fig)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    main(args)
