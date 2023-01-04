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
    count_parameters_in_M, AvgrageMeter, evaluate_transferable_metric, evaluate_transferable_metric_batch
)
from utils.data_utils import CLUSTER_TASKS, load_pkl, zscore_unnormalize
from bc_models.resnet import EndtoEndNet
from utils.vis_utils import generate_transferable_visualization


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluating BC network.')

    # data
    parser.add_argument('--time_interval', type=int, default=15,
                        help='how many frames into the future to predict')
    parser.add_argument('--iou_thresh', type=float, default=0.7,
                        help='IoU threshold for filtering the data')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='if true, dataloader is not shuffled')
    parser.add_argument('--split', default='None', choices=['train', 'valid', 'None'],
                        help='which split of data to evaluate on; choose None if data has no train/valid splits')
    parser.add_argument('--has_task_labels', action='store_true',
                        help='set to true if dataset has task labels')
    parser.add_argument('--has_future_labels', action='store_true',
                        help='set to true if dataset has future labels')
    parser.add_argument("--stage", type=str, default="all",
                        choices=[
                            'all',  # use all data, should also use this if using lab snapshot demo data
                            'pre',  # use pre-interaction data
                            'during'  # use during-interaction data
                        ],
                        help="stage used to filter the dataset")

    # eval
    parser.add_argument('--eval_tasks', action='store_true',
                        help='if true, evaluate conditioning on different tasks')
    parser.add_argument('--eval_robot', action='store_true',
                        help='if true, evaluate on robot r3m embeddings as well as hand r3m embeddings')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--vis_freq', type=int, default=1,
                        help='visualize rendered images after this many steps')
    parser.add_argument('--vis_sample_size', type=int, default=5,
                        help='number of samples per batch to visualize on tensorboard, set to 0 to disable')
    parser.add_argument('--task_vis_sample_size', type=int, default=2,
                        help='number of task-conditioned samples per batch to visualize on tensorboard, '
                             'set to 0 to disable task-conditioned visualization')
    parser.add_argument('--log_depth_scatter_plots', action='store_true',
                        help='if true, log depth evaluation as scatter plots')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataloaders')
    parser.add_argument('--run_on_cv_server', action='store_true',
                        help='if true, run tasks on cv-server; else, run all tasks on cluster')
    parser.add_argument('--use_visualizer', action='store_true',
                        help='if true, use opengl visualizer to render results and show on tensorboard')

    # debugging and sanity check
    parser.add_argument('--debug', action='store_true',
                        help='if true, enter debug mode, load 50 videos and no parallel workers')

    # paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='location of the checkpoint to load for evaluation')
    parser.add_argument('--root', type=str, default='/scratch/junyao/LfHV/r3m/eval_transferable_bc_ckpts',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='debug',
                        help='id used for storing intermediate results')
    parser.add_argument('--data_home_dirs', nargs='+',
                        default='/home/junyao/Datasets/something_something_processed',
                        help='list of locations of the data corpus, example: dir1 dir2 dir3')
    parser.add_argument('--frankmocap_path', type=str,
                        default='/home/junyao/LfHV/frankmocap',
                        help='location of frank mocap')
    parser.add_argument('--r3m_path', type=str,
                        default='/home/junyao/LfHV/r3m',
                        help='location of R3M')
    parser.add_argument('--depth_lf_params_path', type=str,
                        default='/home/junyao/LfHV/frankmocap/ss_utils/depth_linear_fit_params.pkl',
                        help='location of depth linear fit params')

    args = parser.parse_args()
    return args


def main(eval_args):
    program_start = time.time()
    assert eval_args.vis_sample_size <= eval_args.batch_size
    assert eval_args.task_vis_sample_size <= eval_args.batch_size
    eval_args.save = join(eval_args.root, eval_args.save)
    eval_task_names = CLUSTER_TASKS
    eval_args.task_names = eval_task_names
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}.')
    writer = SummaryWriter(log_dir=eval_args.save, flush_secs=60)

    # visualizer and frank mocap
    print('Loading frankmocap visualizer...')
    sys.path.insert(1, eval_args.frankmocap_path)
    os.chdir(eval_args.frankmocap_path)
    from renderer.visualizer import Visualizer
    from handmocap.hand_mocap_api import HandMocap
    visualizer = Visualizer('opengl') if eval_args.use_visualizer else None
    checkpoint_hand = join(eval_args.frankmocap_path, 'extra_data/hand_module/pretrained_weights/pose_shape_best.pth')
    smpl_dir = join(eval_args.frankmocap_path, 'extra_data/smpl')
    hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device=device, batch_size=1)
    os.chdir(eval_args.r3m_path)
    print('Visualizer loaded.')

    # log eval args
    pprint(f'eval args: \n{eval_args}')
    for arg in vars(eval_args):
        writer.add_text(arg, str(getattr(eval_args, arg)))

    # load a checkpoint
    print('loading the model at:')
    print(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']
    pprint(f'loaded train args: \n{args}')
    assert args.task_names == eval_args.task_names, \
        'Currently, difference between evaluation and training tassks is not supported'

    # load depth linear fit params
    eval_args.depth_lf_params = load_pkl(eval_args.depth_lf_params_path)[args.depth_descriptor]

    # initialize model
    model = EndtoEndNet(
        in_features=args.input_dim,
        out_features=args.output_dim,
        dims=(),
        n_blocks=args.n_blocks,
        residual=args.net_type == 'residual'
    ).to(device).float()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f'Loaded transferable BC model. Blocks: {args.n_blocks}, Network: {args.net_type}.')
    print(f'param size = {count_parameters_in_M(model)}M')

    # optimizer and loss
    l2_loss_func = nn.MSELoss()
    bce_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.bce_pos_weight).to(device))

    # create data loaders
    print('Creating data loaders...')
    data_start = time.time()
    eval_data = AgentTransferable(
        data_home_dirs=eval_args.data_home_dirs,
        task_names=eval_args.task_names,
        split=None if eval_args.split == 'None' else eval_args.split,
        iou_thresh=eval_args.iou_thresh,
        time_interval=eval_args.time_interval,
        stage=eval_args.stage,
        depth_descriptor=args.depth_descriptor,
        depth_norm_params=args.depth_norm_params,
        ori_norm_params=args.ori_norm_params,
        debug=eval_args.debug,
        run_on_cv_server=eval_args.run_on_cv_server,
        num_cpus=eval_args.num_workers,
        has_task_labels=eval_args.has_task_labels,
        has_future_labels=eval_args.has_future_labels,
        load_robot_r3m=eval_args.eval_robot,
        load_real_depth=True
    )
    data_end = time.time()
    print(f'Loaded valid data. Time: {data_end - data_start:.5f} seconds')
    print(f'There are {len(eval_data)} evaluation data.')

    eval_queue = torch.utils.data.DataLoader(
        eval_data, batch_size=eval_args.batch_size, shuffle=not eval_args.no_shuffle,
        num_workers=eval_args.num_workers, drop_last=False
    )
    print('Creating data loaders: done')

    hand_start = time.time()
    hand_eval_stats = eval(
        eval_queue, model, device,
        l2_loss_func, bce_loss_func,
        writer, visualizer, hand_mocap,
        eval_args.task_names, args, eval_args, agent='hand'
    )
    log_eval_stats(hand_eval_stats, writer, args, eval_args, agent='hand')
    hand_end = time.time()
    hand_elapsed = str(timedelta(seconds=round(hand_end - hand_start, 2))).split('.')[0]
    print(f'Hand evaluation elapsed time: {hand_elapsed}')
    if eval_args.eval_robot:
        robot_start = time.time()
        robot_eval_stats = eval(
            eval_queue, model, device,
            l2_loss_func, bce_loss_func,
            writer, visualizer, hand_mocap,
            eval_args.task_names, args, eval_args, agent='robot'
        )
        log_eval_stats(robot_eval_stats, writer, args, eval_args, agent='robot')
        robot_end = time.time()
        robot_elapsed = str(timedelta(seconds=round(robot_end - robot_start, 2))).split('.')[0]
        print(f'Robot evaluation elapsed time: {robot_elapsed}')

    program_end = time.time()
    program_elapsed = str(timedelta(seconds=round(program_end - program_start, 2))).split('.')[0]
    print(f'\nDone. Program elapsed time: {program_elapsed}')


def eval(
        eval_queue, model, device,
        l2_loss_func, bce_loss_func,
        writer, visualizer, hand_mocap,
        task_names, args, eval_args, agent='hand'
):
    model.eval()

    epoch_loss = AvgrageMeter()
    epoch_xy_loss = AvgrageMeter()
    epoch_depth_loss = AvgrageMeter()
    epoch_ori_loss = AvgrageMeter()
    epoch_contact_loss = AvgrageMeter()
    epoch_contact_acc = AvgrageMeter()

    # using current hand info for prediction as baseline
    epoch_cur_loss = AvgrageMeter()
    epoch_cur_xy_loss = AvgrageMeter()
    epoch_cur_depth_loss = AvgrageMeter()
    epoch_cur_ori_loss = AvgrageMeter()
    epoch_cur_contact_loss = AvgrageMeter()
    epoch_cur_contact_acc = AvgrageMeter()

    # using mean of batch for prediction as baseline
    epoch_mean_loss = AvgrageMeter()
    epoch_mean_xy_loss = AvgrageMeter()
    epoch_mean_depth_loss = AvgrageMeter()
    epoch_mean_ori_loss = AvgrageMeter()
    epoch_mean_contact_loss = AvgrageMeter()
    epoch_mean_contact_acc = AvgrageMeter()

    epoch_metric_stats = {task_name: {'total': 0, 'pred_success': 0, 'gt_success': 0} for task_name in task_names}
    task_metric_stats = {task_name: {'total': 0, 'pred_success': 0} for task_name in task_names}
    contact_stats = {c: {'gt': 0, 'pred': 0, 'bl_cur': 0, 'bl_mean': 0} for c in ['pos', 'neg']}
    task_contact_stats = {c: {'pred': 0} for c in ['pos', 'neg']}
    depth_stats = {'current_estimate': [], 'current_real': [],
                   'future_estimate': [], 'future_real': [], 'pred_estimate': []}

    for step, data in tqdm(enumerate(eval_queue), f'Going through {agent} eval data...'):
        current_x, current_y = data.current_x.to(device).float(), data.current_y.to(device).float()
        current_xy = torch.cat((current_x.unsqueeze(1), current_y.unsqueeze(1)), dim=1)
        current_depth = data.current_depth.to(device).float()
        current_ori = data.current_ori.to(device)
        current_contact = data.current_contact.to(device).float()

        if eval_args.log_depth_scatter_plots:
            depth_stats['current_real'].extend(data.current_depth_real.tolist())
            depth_stats['current_estimate'].extend(data.current_depth.tolist())

        if eval_args.has_task_labels:
            # process batch data
            input = torch.cat((
                data.hand_r3m if agent == 'hand' else data.robot_r3m,
                data.task,
                data.current_x.unsqueeze(1),
                data.current_y.unsqueeze(1),
                data.current_depth.unsqueeze(1),
                data.current_ori,
                data.current_contact.unsqueeze(1),
            ), dim=1).to(device).float()
            batch_size = input.size(0)

            with torch.no_grad():
                # forward through model and process output
                output = model(input)
                pred_xy, pred_depth = output[:, 0:2], output[:, 2]
                pred_ori, pred_contact = output[:, 3:6], output[:, 6]
                if not args.pred_residual:
                    pred_xy = torch.sigmoid(pred_xy)  # force xy to be positive bbox coords
                pred_contact_binary = torch.round(torch.sigmoid(pred_contact))

            if eval_args.log_depth_scatter_plots:
                pred_depth_estimate = current_depth + pred_depth if args.pred_residual else pred_depth
                depth_stats['pred_estimate'].extend(pred_depth_estimate.tolist())

            if eval_args.has_future_labels:
                future_x, future_y = data.future_x.to(device).float(), data.future_y.to(device).float()
                future_xy = torch.cat((future_x.unsqueeze(1), future_y.unsqueeze(1)), dim=1)
                future_depth = data.future_depth.to(device).float()
                future_ori = data.future_ori.to(device)
                future_contact = data.future_contact.to(device).float()

                if eval_args.log_depth_scatter_plots:
                    depth_stats['future_real'].extend(data.future_depth_real.tolist())
                    depth_stats['future_estimate'].extend(data.future_depth.tolist())

                # handle residual prediction mode
                target_xy = future_xy - current_xy if args.pred_residual else future_xy
                target_depth = future_depth - current_depth if args.pred_residual else future_depth
                target_ori = future_ori - current_ori if args.pred_residual else future_ori

                # process loss
                xy_loss = l2_loss_func(pred_xy, target_xy)
                depth_loss = l2_loss_func(pred_depth, target_depth)
                ori_loss = l2_loss_func(pred_ori, target_ori)
                contact_loss = bce_loss_func(pred_contact, future_contact)
                loss_unweighted = xy_loss + depth_loss + ori_loss + contact_loss

                pred_contact_correct = (pred_contact_binary == future_contact).sum().float()
                contact_acc = pred_contact_correct / pred_contact.size(0)

                pos_inds, neg_inds = future_contact == 1, future_contact == 0
                pred_contact_pos, pred_contact_neg = pred_contact_binary[pos_inds], pred_contact_binary[neg_inds]
                future_contact_pos, future_contact_neg = future_contact[pos_inds], future_contact[neg_inds]
                n_pos, n_neg = future_contact_pos.size(0), future_contact_neg.size(0)
                pred_n_pos = (pred_contact_pos == future_contact_pos).sum().item()
                pred_n_neg = (pred_contact_neg == future_contact_neg).sum().item()
                contact_stats['pos']['gt'] += n_pos
                contact_stats['neg']['gt'] += n_neg
                contact_stats['pos']['pred'] += pred_n_pos
                contact_stats['neg']['pred'] += pred_n_neg

                # process baseline loss
                cur_xy = torch.zeros_like(current_xy) if args.pred_residual else current_xy
                cur_depth = torch.zeros_like(current_depth) if args.pred_residual else current_depth
                cur_ori = torch.zeros_like(current_ori) if args.pred_residual else current_ori

                cur_xy_loss = l2_loss_func(cur_xy, target_xy)
                cur_depth_loss = l2_loss_func(cur_depth, target_depth)
                cur_ori_loss = l2_loss_func(cur_ori, target_ori)
                cur_contact_loss = bce_loss_func(current_contact, future_contact)
                cur_loss_unweighted = cur_xy_loss + cur_depth_loss + cur_ori_loss + cur_contact_loss

                cur_contact_correct = (current_contact == future_contact).sum().float()
                cur_contact_acc = cur_contact_correct / current_contact.size(0)

                cur_pred_contact_pos, cur_pred_contact_neg = current_contact[pos_inds], current_contact[neg_inds]
                cur_n_pos = (cur_pred_contact_pos == future_contact_pos).sum().item()
                cur_n_neg = (cur_pred_contact_neg == future_contact_neg).sum().item()
                contact_stats['pos']['bl_cur'] += cur_n_pos
                contact_stats['neg']['bl_cur'] += cur_n_neg

                mean_xy = torch.mean(target_xy, dim=0).unsqueeze(0).repeat(batch_size, 1)
                mean_depth = torch.mean(target_depth).repeat(batch_size)
                mean_ori = torch.mean(target_ori, dim=0).unsqueeze(0).repeat(batch_size, 1)
                mean_contact = torch.mean(future_contact).repeat(batch_size)

                mean_xy_loss = l2_loss_func(mean_xy, target_xy)
                mean_depth_loss = l2_loss_func(mean_depth, target_depth)
                mean_ori_loss = l2_loss_func(mean_ori, target_ori)
                mean_contact_loss = bce_loss_func(mean_contact, future_contact)
                mean_loss_unweighted = mean_xy_loss + mean_depth_loss + mean_ori_loss + mean_contact_loss

                mean_contact_binary = torch.round(torch.sigmoid(mean_contact))
                mean_contact_correct = (mean_contact_binary == future_contact).sum().float()
                mean_contact_acc = mean_contact_correct / batch_size

                mean_pred_contact_pos = mean_contact_binary[pos_inds]
                mean_pred_contact_neg = mean_contact_binary[neg_inds]
                mean_n_pos = (mean_pred_contact_pos == future_contact_pos).sum().item()
                mean_n_neg = (mean_pred_contact_neg == future_contact_neg).sum().item()
                contact_stats['pos']['bl_mean'] += mean_n_pos
                contact_stats['neg']['bl_mean'] += mean_n_neg

                # update epoch average meters
                epoch_loss.update(loss_unweighted.data, batch_size)
                epoch_xy_loss.update(xy_loss.data, batch_size)
                epoch_depth_loss.update(depth_loss.data, batch_size)
                epoch_ori_loss.update(ori_loss.data, batch_size)
                epoch_contact_loss.update(contact_loss.data, batch_size)
                epoch_contact_acc.update(contact_acc.data, batch_size)

                epoch_cur_loss.update(cur_loss_unweighted.data, batch_size)
                epoch_cur_xy_loss.update(cur_xy_loss.data, batch_size)
                epoch_cur_depth_loss.update(cur_depth_loss.data, batch_size)
                epoch_cur_ori_loss.update(cur_ori_loss.data, batch_size)
                epoch_cur_contact_loss.update(cur_contact_loss.data, batch_size)
                epoch_cur_contact_acc.update(cur_contact_acc.data, batch_size)

                epoch_mean_loss.update(mean_loss_unweighted.data, 1)
                epoch_mean_xy_loss.update(mean_xy_loss.data, 1)
                epoch_mean_depth_loss.update(mean_depth_loss.data, 1)
                epoch_mean_ori_loss.update(mean_ori_loss.data, 1)
                epoch_mean_contact_loss.update(mean_contact_loss.data, 1)
                epoch_mean_contact_acc.update(mean_contact_acc.data, 1)

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
            else:
                metric_stats = evaluate_transferable_metric_batch(
                    task_names=task_names,
                    task=data.task,
                    device=device,
                    current_x=current_x,
                    pred_x=current_x + pred_xy[:, 0] if args.pred_residual else pred_xy[:, 0],
                    current_y=current_y,
                    pred_y=current_y + pred_xy[:, 1] if args.pred_residual else pred_xy[:, 1],
                    current_depth=current_depth,
                    pred_depth=current_depth + pred_depth if args.pred_residual else pred_depth,
                    evaluate_gt=False
                )

            for k, v in metric_stats.items():
                epoch_metric_stats[k]['total'] += v['total']
                epoch_metric_stats[k]['pred_success'] += v['pred_success']
                if eval_args.has_future_labels:
                    epoch_metric_stats[k]['gt_success'] += v['gt_success']

            # log images
            if (step + 1) % eval_args.vis_freq == 0:
                for i in range(eval_args.vis_sample_size):
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
                        run_on_cv_server=eval_args.run_on_cv_server,
                        hand=data.hand[i],
                        pred_delta_x=pred_delta_x.item(),
                        pred_delta_y=pred_delta_y.item(),
                        pred_delta_depth=pred_delta_depth.item(),
                        pred_delta_ori=pred_delta_ori,
                        pred_contact=pred_contact_binary[i],
                        depth_norm_params=args.depth_norm_params,
                        ori_norm_params=args.ori_norm_params,
                        vis_robot=agent == 'robot',
                        task_name=task_name,
                        visualizer=visualizer,
                        use_visualizer=eval_args.use_visualizer,
                        hand_mocap=hand_mocap,
                        device=device,
                        log_metric=True,
                        passed_metric=passed_metric.item()
                    )
                    writer.add_image(f'eval_{agent}_{task_name}/sample{i}', vis_img, step, dataformats='HWC')

        # task-conditioned evaluation
        if eval_args.eval_tasks:
            task_vis_sample_count = 0
            all_task_instances = []
            for j, task_name in enumerate(task_names):
                task_instance = torch.zeros(1, len(task_names))
                task_instance[0, j] = 1
                all_task_instances.append(task_instance)
            all_task_instances = torch.vstack(all_task_instances)
            for i in range(data.current_x.size(0)):
                if eval_args.has_task_labels:
                    original_task_name = task_names[torch.argmax(data.task[i].squeeze())]
                else:
                    original_task_name = 'None'
                task_conditioned_input = torch.cat((
                    data.hand_r3m[i].repeat(len(task_names), 1) if agent == 'hand' \
                        else data.robot_r3m[i].repeat(len(task_names), 1),
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
                    t_pred_contact_binary = torch.round(torch.sigmoid(t_pred_contact))

                task_contact_stats['pos']['pred'] += (t_pred_contact_binary == 1).sum().item()
                task_contact_stats['neg']['pred'] += (t_pred_contact_binary == 0).sum().item()

                # process metric evaluation
                t_current_x = data.current_x[i].repeat(len(task_names)).to(device).float()
                t_current_y = data.current_y[i].repeat(len(task_names)).to(device).float()
                t_current_depth = data.current_depth[i].repeat(len(task_names)).to(device).float()

                if not eval_args.has_task_labels and eval_args.log_depth_scatter_plots:
                    t_pred_depth_estimate = t_current_depth + t_pred_depth if args.pred_residual else t_pred_depth
                    depth_stats['pred_estimate'].extend(t_pred_depth_estimate.tolist())

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
                        pred_delta_depth = t_pred_depth[j] if args.pred_residual \
                            else t_pred_depth[j] - current_depth[i]
                        pred_delta_ori = t_pred_ori[j] if args.pred_residual else t_pred_ori[j] - current_ori[i]
                        pred_contact_binary = torch.round(torch.sigmoid(t_pred_contact[j]))

                        passed_metric = evaluate_transferable_metric(
                            task_name=task_name,
                            current_x=current_x[i],
                            pred_x=current_x[i] + t_pred_xy[j, 0] if args.pred_residual else t_pred_xy[j, 0],
                            current_y=current_y[i],
                            pred_y=current_y[i] + t_pred_xy[j, 1] if args.pred_residual else t_pred_xy[j, 1],
                            current_depth=current_depth[i],
                            pred_depth=current_depth[i] + t_pred_depth[j] if args.pred_residual \
                                else t_pred_depth[j],
                        )

                        task_vis_img = generate_transferable_visualization(
                            current_hand_pose_path=data.current_info_path[i],
                            future_hand_pose_path=None,
                            run_on_cv_server=eval_args.run_on_cv_server,
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
                            use_visualizer=eval_args.use_visualizer,
                            hand_mocap=hand_mocap,
                            device=device,
                            log_metric=True,
                            passed_metric=passed_metric.item(),
                            original_task=task_name == original_task_name,
                            vis_robot=agent == 'robot',
                            vis_groundtruth=False
                        )
                        task_vis_imgs.append(task_vis_img)
                    final_task_vis_img = np.hstack(task_vis_imgs)
                    writer.add_image(f'eval_tasks_{agent}/vis_tasks_{i}', final_task_vis_img, step, dataformats='HWC')
                    task_vis_sample_count += 1

    stats = namedtuple('stats', [
        'loss', 'xy_loss', 'depth_loss', 'ori_loss', 'contact_loss', 'contact_acc',
        'cur_loss', 'cur_xy_loss', 'cur_depth_loss', 'cur_ori_loss', 'cur_contact_loss', 'cur_contact_acc',
        'mean_loss', 'mean_xy_loss', 'mean_depth_loss', 'mean_ori_loss', 'mean_contact_loss', 'mean_contact_acc',
        'epoch_metric_stats', 'task_metric_stats', 'contact_stats', 'task_contact_stats', 'depth_stats'
    ])

    return stats(
        loss=epoch_loss.avg,
        xy_loss=epoch_xy_loss.avg,
        depth_loss=epoch_depth_loss.avg,
        ori_loss=epoch_ori_loss.avg,
        contact_loss=epoch_contact_loss.avg,
        contact_acc=epoch_contact_acc.avg,
        cur_loss=epoch_cur_loss.avg,
        cur_xy_loss=epoch_cur_xy_loss.avg,
        cur_depth_loss=epoch_cur_depth_loss.avg,
        cur_ori_loss=epoch_cur_ori_loss.avg,
        cur_contact_loss=epoch_cur_contact_loss.avg,
        cur_contact_acc=epoch_cur_contact_acc.avg,
        mean_loss=epoch_mean_loss.avg,
        mean_xy_loss=epoch_mean_xy_loss.avg,
        mean_depth_loss=epoch_mean_depth_loss.avg,
        mean_ori_loss=epoch_mean_ori_loss.avg,
        mean_contact_loss=epoch_mean_contact_loss.avg,
        mean_contact_acc=epoch_mean_contact_acc.avg,
        epoch_metric_stats=epoch_metric_stats,
        task_metric_stats=task_metric_stats,
        contact_stats=contact_stats,
        task_contact_stats=task_contact_stats,
        depth_stats=depth_stats
    )


def log_eval_stats(stats, writer, args, eval_args, agent='hand'):
    metric_stats = stats.epoch_metric_stats
    task_metric_stats = stats.task_metric_stats
    contact_stats = stats.contact_stats
    task_contact_stats = stats.task_contact_stats
    depth_stats = stats.depth_stats

    # visualize metric evaluation success rate by bar plot
    bar_width = 0.4
    fig_height = 10
    n_task = len(args.task_names)

    # log train epoch stats
    if eval_args.has_future_labels and eval_args.has_task_labels:
        print(f'{agent} loss: {stats.loss}')
        print(f'{agent} xy loss: {stats.xy_loss}')
        print(f'{agent} depth loss: {stats.depth_loss}')
        print(f'{agent} orientation loss: {stats.ori_loss}')
        print(f'{agent} contact loss: {stats.contact_loss}')
        print(f'{agent} contact acc: {stats.contact_acc}')

        writer.add_scalar(f'{agent}/loss', stats.loss)
        writer.add_scalar(f'{agent}/xy_loss', stats.xy_loss)
        writer.add_scalar(f'{agent}/depth_loss', stats.depth_loss)
        writer.add_scalar(f'{agent}/ori_loss', stats.ori_loss)
        writer.add_scalar(f'{agent}/contact_loss', stats.contact_loss)
        writer.add_scalar(f'{agent}/contact_acc', stats.contact_acc)

        writer.add_scalar(f'{agent}_baseline_current/loss', stats.cur_loss)
        writer.add_scalar(f'{agent}_baseline_current/xy_loss', stats.cur_xy_loss)
        writer.add_scalar(f'{agent}_baseline_current/depth_loss', stats.cur_depth_loss)
        writer.add_scalar(f'{agent}_baseline_current/ori_loss', stats.cur_ori_loss)
        writer.add_scalar(f'{agent}_baseline_current/contact_loss', stats.cur_contact_loss)
        writer.add_scalar(f'{agent}_baseline_current/contact_acc', stats.cur_contact_acc)

        writer.add_scalar(f'{agent}_baseline_mean/loss', stats.mean_loss)
        writer.add_scalar(f'{agent}_baseline_mean/xy_loss', stats.mean_xy_loss)
        writer.add_scalar(f'{agent}_baseline_mean/depth_loss', stats.mean_depth_loss)
        writer.add_scalar(f'{agent}_baseline_mean/ori_loss', stats.mean_ori_loss)
        writer.add_scalar(f'{agent}_baseline_mean/contact_loss', stats.mean_contact_loss)
        writer.add_scalar(f'{agent}_baseline_mean/contact_acc', stats.mean_contact_acc)

        fig = plt.figure(figsize=(2 * n_task , fig_height))
        bar1 = np.arange(len(metric_stats) + 1)
        bar2 = [x + bar_width for x in bar1]
        epoch_pred_success_rate = [
            v['pred_success'] / v['total'] if v['total'] != 0 else 0
            for v in metric_stats.values()
        ]
        epoch_gt_success_rate = [
            v['gt_success'] / v['total'] if v['total'] != 0 else 0
            for v in metric_stats.values()
        ]
        total_sum = sum([v['total'] for v in metric_stats.values()])
        pred_success_sum = sum([v['pred_success'] for v in metric_stats.values()])
        gt_success_sum = sum([v['gt_success'] for v in metric_stats.values()])
        epoch_pred_success_rate.append(pred_success_sum / total_sum if total_sum != 0 else 0)
        epoch_gt_success_rate.append(gt_success_sum / total_sum if total_sum != 0 else 0)

        plt.bar(bar1, epoch_pred_success_rate, width=bar_width, color='peachpuff', label='pred')
        plt.bar(bar2, epoch_gt_success_rate, width=bar_width, color='lavender', label='gt')
        for b1, ps, b2, gs in zip(bar1, epoch_pred_success_rate, bar2, epoch_gt_success_rate):
            plt.text(b1, ps / 2, f'{ps:.2f}', ha='center', fontsize=13)
            plt.text(b2, gs / 2, f'{gs:.2f}', ha='center', fontsize=13)
        plt.xlabel('Tasks', fontweight='bold', size=13)
        plt.ylabel('Success rate', fontweight='bold', size=13)
        xticks = list(metric_stats.keys())
        xticks.append('total')
        plt.xticks([r + bar_width / 2 for r in bar1], xticks)
        plt.ylim([0, 1])
        plt.title('1D Metric Evaluation')
        plt.legend()
        plt.tight_layout()
        writer.add_figure(f'{agent}_metric/success_rate', fig)
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
        writer.add_figure(f'{agent}_metric/count', fig)
        plt.close(fig)

        # log contact plots
        gt_n_pos, gt_n_neg = contact_stats['pos']['gt'], contact_stats['neg']['gt']
        gt_n_total = gt_n_pos + gt_n_neg
        pred_n_pos, pred_n_neg = contact_stats['pos']['pred'], contact_stats['neg']['pred']
        cur_n_pos, cur_n_neg = contact_stats['pos']['bl_cur'], contact_stats['neg']['bl_cur']
        mean_n_pos, mean_n_neg = contact_stats['pos']['bl_mean'], contact_stats['neg']['bl_mean']

        # visualize contact success rate by bar plot
        bar_width = 0.25
        fig = plt.figure(figsize=(4 * 3, fig_height))

        pred_success_rate = [
            pred_n_pos / gt_n_pos if gt_n_pos != 0 else 0,
            pred_n_neg / gt_n_neg if gt_n_neg != 0 else 0,
            (pred_n_pos + pred_n_neg) / gt_n_total if gt_n_total != 0 else 0
        ]
        cur_success_rate = [
            cur_n_pos / gt_n_pos if gt_n_pos != 0 else 0,
            cur_n_neg / gt_n_neg if gt_n_neg != 0 else 0,
            (cur_n_pos + cur_n_neg) / gt_n_total if gt_n_total != 0 else 0
        ]
        mean_success_rate = [
            mean_n_pos / gt_n_pos if gt_n_pos != 0 else 0,
            mean_n_neg / gt_n_neg if gt_n_neg != 0 else 0,
            (mean_n_pos + mean_n_neg) / gt_n_total if gt_n_total != 0 else 0
        ]
        bar1 = np.arange(3)
        bar2 = [x + bar_width for x in bar1]
        bar3 = [x + bar_width for x in bar2]

        plt.bar(bar1, pred_success_rate, color='peachpuff', width=bar_width, label='pred')
        plt.bar(bar2, cur_success_rate, color='lavender', width=bar_width, label='bl_cur')
        plt.bar(bar3, mean_success_rate, color='skyblue', width=bar_width, label='bl_mean')
        for b1, ps, b2, cs, b3, ms in zip(bar1, pred_success_rate, bar2, cur_success_rate, bar3, mean_success_rate):
            plt.text(b1, ps / 2, f'{ps:.2f}', ha='center', fontsize=8)
            plt.text(b2, cs / 2, f'{cs:.2f}', ha='center', fontsize=8)
            plt.text(b3, ms / 2, f'{ms:.2f}', ha='center', fontsize=8)

        plt.xticks([r + bar_width for r in bar1], ['pos', 'neg', 'combined'])
        plt.ylabel('Success rate', fontweight='bold', size=13)
        plt.ylim([0, 1])
        plt.title('Contact')
        plt.legend()
        plt.tight_layout()
        writer.add_figure(f'{agent}_contact/success_rate', fig)
        plt.close(fig)

        # visualize contact success count by bar plot
        bar_width = 0.2
        fig = plt.figure(figsize=(5 * 3, fig_height))

        pred_success = [pred_n_pos, pred_n_neg, pred_n_pos + pred_n_neg]
        cur_success = [cur_n_pos, cur_n_neg, cur_n_pos + cur_n_neg]
        mean_success = [mean_n_pos, mean_n_neg, mean_n_pos + mean_n_neg]
        total = [gt_n_pos, gt_n_neg, gt_n_total]
        bar1 = np.arange(3)
        bar2 = [x + bar_width for x in bar1]
        bar3 = [x + bar_width for x in bar2]
        bar4 = [x + bar_width for x in bar3]

        plt.bar(bar1, pred_success, color='peachpuff', width=bar_width, label='pred')
        plt.bar(bar2, cur_success, color='lavender', width=bar_width, label='bl_cur')
        plt.bar(bar3, mean_success, color='skyblue', width=bar_width, label='bl_mean')
        plt.bar(bar4, total, color='wheat', width=bar_width, label='total')

        for b1, ps, b2, cs, b3, ms, b4, t in zip(
                bar1, pred_success, bar2, cur_success, bar3, mean_success, bar4, total
        ):
            plt.text(b1, ps / 2, ps, ha='center', fontsize=8)
            plt.text(b2, cs / 2, cs, ha='center', fontsize=8)
            plt.text(b3, ms / 2, ms, ha='center', fontsize=8)
            plt.text(b4, t / 2, t, ha='center', fontsize=8)

        plt.xticks([r + bar_width * 1.5 for r in bar1], ['pos', 'neg', 'combined'])
        plt.ylabel('Count', fontweight='bold', size=13)
        plt.title('Contact')
        plt.legend()
        plt.tight_layout()
        writer.add_figure(f'{agent}_contact/count', fig)
        plt.close(fig)

    # log task conditioning plots
    if eval_args.eval_tasks:
        # visualize metric evaluation success rate by bar plot
        bar_width = 0.7
        fig = plt.figure(figsize=(2 * n_task, fig_height))
        epoch_success_rate = [
            v['pred_success'] / v['total'] if v['total'] != 0 else 0
            for v in task_metric_stats.values()
        ]
        total_sum = sum([v['total'] for v in task_metric_stats.values()])
        pred_success_sum = sum([v['pred_success'] for v in task_metric_stats.values()])
        epoch_success_rate.append(pred_success_sum / total_sum if total_sum != 0 else 0)
        xticks = list(task_metric_stats.keys())
        xticks.append('total')
        plt.bar(xticks, epoch_success_rate, width=bar_width, color='skyblue')
        for i, r in enumerate(epoch_success_rate):
            plt.text(i, r / 2, f'{r:.2f}', ha='center', fontsize=15)
        plt.xlabel('Tasks', fontweight='bold', size=13)
        plt.ylabel('Success rate', fontweight='bold', size=13)
        plt.ylim([0, 1])
        plt.title('1D Metric Evaluation')
        plt.tight_layout()
        writer.add_figure(f'{agent}_task_metric/task_conditioning_success_rate', fig)
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
        writer.add_figure(f'{agent}_task_metric/task_conditioning_count', fig)
        plt.close(fig)

        # log contact plots
        pred_n_pos, pred_n_neg = task_contact_stats['pos']['pred'], task_contact_stats['neg']['pred']

        # visualize contact success count by bar plot
        bar_width = 0.75
        fig = plt.figure(figsize=(5 * 3, fig_height))

        pred_count = [pred_n_pos, pred_n_neg, pred_n_pos + pred_n_neg]
        bar1 = np.arange(3)
        plt.bar(bar1, pred_count, color='peachpuff', width=bar_width, label='pred')

        for b1, pc in zip(bar1, pred_count):
            plt.text(b1, pc / 2, pc, ha='center', fontsize=8)

        plt.xticks([r for r in bar1], ['pos', 'neg', 'combined'])
        plt.ylabel('Count', fontweight='bold', size=13)
        plt.title('Contact')
        plt.legend()
        plt.tight_layout()
        writer.add_figure(f'{agent}_task_contact/count', fig)
        plt.close(fig)

    if eval_args.log_depth_scatter_plots:
        m, b = eval_args.depth_lf_params['m'], eval_args.depth_lf_params['b']
        if agent == 'hand':
            current_estimates, current_real = [], []
            for estimate, real in zip(depth_stats['current_estimate'], depth_stats['current_real']):
                if real == 0 or real == -999:
                    continue
                estimate_unnormalized = zscore_unnormalize(estimate, args.depth_norm_params)
                current_estimates.append(m * estimate_unnormalized + b)
                current_real.append(real)
            fig = plt.figure()
            title = 'current vs. current real'
            plt.title(title)
            plt.scatter(current_estimates, current_real)
            writer.add_figure(f'scatter_plots/{title}', fig)
            plt.close(fig)

        if eval_args.has_future_labels:
            pred_estimates, future_real = [], []
            for estimate, real in zip(depth_stats['pred_estimate'], depth_stats['future_real']):
                if real == 0 or real == -999:
                    continue
                estimate_unnormalized = zscore_unnormalize(estimate, args.depth_norm_params)
                pred_estimates.append(m * estimate_unnormalized + b)
                future_real.append(real)
            fig = plt.figure()
            title = 'prediction vs. future real'
            plt.title(title)
            plt.scatter(pred_estimates, future_real)
            writer.add_figure(f'scatter_plots/{agent} {title}', fig)
            plt.close(fig)

            fig = plt.figure()
            title = 'prediction vs. future'
            plt.title(title)
            plt.scatter(depth_stats['pred_estimate'], depth_stats['future_estimate'])
            writer.add_figure(f'scatter_plots/{agent} {title}', fig)
            plt.close(fig)


if __name__ == '__main__':
    eval_args = parse_args()
    main(eval_args)
    time.sleep(3)  # give some time for tensorboard to upload image
