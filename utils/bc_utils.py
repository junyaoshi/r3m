import json
import pickle
import sys
from os.path import join
import os
from pprint import pprint
from collections import namedtuple
import argparse

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from bc_models.resnet import EndtoEndNet, PartiallyTransferableNet
from utils.data_utils import HALF_TASKS, ALL_TASKS, xywh_to_xyxy, normalize_bbox, unnormalize_bbox, \
    mocap_path_to_rendered_path, pose_to_joint_depth, zscore_normalize


EpochStats = namedtuple('epoch_stats', [
    'LossMonitor', 'BLcurMonitor', 'BLmeanMonitor', 'EpochMetric', 'TaskMetric', 'ContactStats', 'global_step',
])

LossStats = namedtuple('loss_stats', [
    'total_loss', 'total_unweighted_loss', 'xy_loss', 'depth_loss', 'ori_loss', 'contact_loss', 'contact_acc'
])


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def safe_div(a, b):
    """a/b, but returns 0 when b=0"""
    return a / b  if b != 0 else 0


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_eval_bc_model_and_args(eval_args, device):
    print('loading the model at:')
    print(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']
    pprint(f'loaded train args: \n{args}')

    # compute dimensions
    r3m_dim, task_dim, cam_dim = 2048, len(HALF_TASKS) if args.run_on_cv_server else len(ALL_TASKS), 3
    args.hand_bbox_dim, args.hand_pose_dim, args.hand_shape_dim = 4, 48, 10
    hand_dim = sum([args.hand_bbox_dim, args.hand_pose_dim, args.hand_shape_dim])
    input_dim = sum([r3m_dim, task_dim, hand_dim, cam_dim, cam_dim])
    output_dim = hand_dim

    # load model
    model, model_init_func, residual = None, None, None
    if args.model_type == 'e2e':
        model_init_func = EndtoEndNet
    elif args.model_type == 'transfer':
        model_init_func = PartiallyTransferableNet
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

    return model, args


def evaluate_transferable_metric_batch_old(
        task_names, task,
        current_x, pred_x,
        current_y, pred_y,
        current_depth, pred_depth
):
    # Note: this implementation is not as efficient as the one below
    metric_stats = {task_name: {'total': 0, 'success': 0} for task_name in task_names}
    for i, t in enumerate(task):
        task_name = task_names[torch.argmax(t)]
        passed_metric = evaluate_transferable_metric(
            task_name=task_name,
            current_x=current_x[i], pred_x=pred_x[i],
            current_y=current_y[i], pred_y=pred_y[i],
            current_depth=current_depth[i], pred_depth=pred_depth[i],
        )
        metric_stats[task_name]['total'] += 1
        metric_stats[task_name]['success'] += passed_metric.item()
    return metric_stats


def evaluate_transferable_metric_batch(
        task_names, task, device,
        current_x, pred_x,
        current_y, pred_y,
        current_depth, pred_depth,
        evaluate_gt=False, future_x=None, future_y=None, future_depth=None
):
    metric_stats = {task_name: {'total': 0, 'pred_success': 0, 'gt_success': 0} for task_name in task_names}
    all_pred_task_metric = []
    for task_name in task_names:
        if task_name == 'move_away':
            all_pred_task_metric.append((pred_depth > current_depth).unsqueeze(1))
        elif task_name == 'move_towards':
            all_pred_task_metric.append((pred_depth < current_depth).unsqueeze(1))
        elif task_name == 'move_down':
            all_pred_task_metric.append((pred_y > current_y).unsqueeze(1))
        elif task_name == 'move_up':
            all_pred_task_metric.append((pred_y < current_y).unsqueeze(1))
        elif task_name == 'pull_left' or task_name == 'push_left':
            all_pred_task_metric.append((pred_x < current_x).unsqueeze(1))
        elif task_name == 'pull_right' or task_name == 'push_right':
            all_pred_task_metric.append((pred_x > current_x).unsqueeze(1))
    all_pred_task_metric = torch.hstack(all_pred_task_metric)
    task_inds = torch.argmax(task, dim=1).to(device)
    pred_task_metric = all_pred_task_metric.gather(1, task_inds.unsqueeze(1))

    for ind, pred_metric in zip(task_inds, pred_task_metric):
        task_name = task_names[ind]
        metric_stats[task_name]['total'] += 1
        metric_stats[task_name]['pred_success'] += pred_metric.item()

    if evaluate_gt:
        assert future_x is not None and future_y is not None and future_depth is not None
        all_gt_task_metric = []
        for task_name in task_names:
            if task_name == 'move_away':
                all_gt_task_metric.append((future_depth > current_depth).unsqueeze(1))
            elif task_name == 'move_towards':
                all_gt_task_metric.append((future_depth < current_depth).unsqueeze(1))
            elif task_name == 'move_down':
                all_gt_task_metric.append((future_y > current_y).unsqueeze(1))
            elif task_name == 'move_up':
                all_gt_task_metric.append((future_y < current_y).unsqueeze(1))
            elif task_name == 'pull_left' or task_name == 'push_left':
                all_gt_task_metric.append((future_x < current_x).unsqueeze(1))
            elif task_name == 'pull_right' or task_name == 'push_right':
                all_gt_task_metric.append((future_x > current_x).unsqueeze(1))
        all_gt_task_metric = torch.hstack(all_gt_task_metric)
        task_inds = torch.argmax(task, dim=1).to(device)
        gt_task_metric = all_gt_task_metric.gather(1, task_inds.unsqueeze(1))

        for ind, gt_metric in zip(task_inds, gt_task_metric):
            task_name = task_names[ind]
            metric_stats[task_name]['gt_success'] += gt_metric.item()

    return metric_stats


def evaluate_transferable_metric(
        task_name,
        current_x, pred_x,
        current_y, pred_y,
        current_depth, pred_depth,
):
    assert task_name in [
        'move_away', 'move_towards',
        'move_down', 'move_up',
        'pull_left', 'pull_right',
        'push_left', 'push_right',
    ], f'Unknown task name: {task_name}'
    if task_name == 'move_away':
        return pred_depth > current_depth
    elif task_name == 'move_towards':
        return pred_depth < current_depth
    elif task_name == 'move_down':
        return pred_y > current_y
    elif task_name == 'move_up':
        return pred_y < current_y
    elif task_name == 'pull_left' or task_name == 'push_left':
        return pred_x < current_x
    elif task_name == 'pull_right' or task_name == 'push_right':
        return pred_x > current_x


def evaluate_metric(
        task_name,
        current_bbox=None, pred_bbox=None,
        current_depth=None, pred_depth=None,
        pre_interaction=False, object_label=None
):
    if pre_interaction:
        raise NotImplementedError
        assert object_label is not None
        assert current_bbox is not None and pred_bbox is not None
        assert current_depth is not None and pred_depth is not None
        current_xyxy, future_xyxy = xywh_to_xyxy(current_bbox), xywh_to_xyxy(pred_bbox)
        c_x1, c_y1, c_x2, c_y2 = current_xyxy
        f_x1, f_y1, f_x2, f_y2 = future_xyxy
        c_xc, c_yc = (c_x2 + c_x1) / 2, (c_y2 + c_y1) / 2  # current xy center
        f_xc, f_yc = (f_x2 + f_x1) / 2, (f_y2 + f_y1) / 2  # future xy center
        c_center = np.array([c_xc, c_yc])
        f_center = np.array([f_xc, f_yc])
        lr_movement = np.mean([f_x1 - c_x1, f_x2 - c_x2])
        tb_movement = np.mean([f_y1 - c_y1, f_y2 - c_y2])
        at_movement = pred_depth - current_depth
        lr, tb, at = object_label
        if lr == 0:  # left_right: same
            passed_lr = np.abs(f_xc - c_xc) < 20
        elif lr == 1: # left_right: object is to the left of hand
            passed_lr = lr_movement < 0

    else:
        """Returns a boolean flag: whether it passes the eval metric"""
        if task_name in ['move_away', 'move_towards']:
            assert current_depth is not None and pred_depth is not None
            if task_name == 'move_away':
                return pred_depth > current_depth
            elif task_name == 'move_towards':
                return pred_depth < current_depth
        elif task_name in [
            'move_down', 'move_up',
            'pull_left', 'pull_right',
            'push_left', 'push_right',
            'push_slightly'
        ]:
            assert current_bbox is not None and pred_bbox is not None
            current_xyxy, future_xyxy = xywh_to_xyxy(current_bbox), xywh_to_xyxy(pred_bbox)
            c_x1, c_y1, c_x2, c_y2 = current_xyxy
            f_x1, f_y1, f_x2, f_y2 = future_xyxy
            c_xc, c_yc = (c_x2 + c_x1) / 2, (c_y2 + c_y1) / 2  # current xy center
            f_xc, f_yc = (f_x2 + f_x1) / 2, (f_y2 + f_y1) / 2  # future xy center
            c_center = np.array([c_xc, c_yc])
            f_center = np.array([f_xc, f_yc])
            if task_name == 'move_down':
                return np.mean([f_y1 - c_y1, f_y2 - c_y2]) > 0
            elif task_name == 'move_up':
                return np.mean([f_y1 - c_y1, f_y2 - c_y2]) < 0
            elif task_name == 'pull_left' or task_name == 'push_left':
                return np.mean([f_x1 - c_x1, f_x2 - c_x2]) < 0
            elif task_name == 'pull_right' or task_name == 'push_right':
                return np.mean([f_x1 - c_x1, f_x2 - c_x2]) > 0
            elif task_name == 'push_slightly':
                print(f'c_x1, c_y1, c_x2, c_y2: {current_xyxy}')
                print(f'f_x1, f_y1, f_x2, f_y2: {future_xyxy}')
                print(f'c_xc: {c_xc}; c_yc: {c_yc}; f_xc: {f_xc}; f_yc: {f_yc}')
                print(f'fcenter: {f_center}; ccenter: {c_center}; distance: {np.linalg.norm(f_center - c_center)}')
                return np.linalg.norm(f_center - c_center) < 50 and np.abs(pred_depth - current_depth) < 10
        else:
            raise ValueError(f'Encountered unknown task name [{task_name}] during metric evaluation.')


def write_loss_to_tb(loss_stats, writer, tb_step, tag, log_contact=False):
    """Write instance of LossStats to tensorboard"""
    writer.add_scalar(f'{tag}/total_loss', loss_stats.total_loss, tb_step)
    writer.add_scalar(f'{tag}/total_unweighted_loss', loss_stats.total_unweighted_loss, tb_step)
    writer.add_scalar(f'{tag}/xy_loss', loss_stats.xy_loss, tb_step)
    writer.add_scalar(f'{tag}/depth_loss', loss_stats.depth_loss, tb_step)
    writer.add_scalar(f'{tag}/ori_loss', loss_stats.ori_loss, tb_step)
    if log_contact:
        writer.add_scalar(f'{tag}/contact_loss', loss_stats.contact_loss, tb_step)
        writer.add_scalar(f'{tag}/contact_acc', loss_stats.contact_acc, tb_step)


def print_epoch_loss(loss_monitor, mode, log_contact=False):
    """Print instance of EpochLossMonitor"""
    print(f'epoch {mode} total loss: {loss_monitor.total_loss.avg}')
    print(f'epoch {mode} total unweighted loss: {loss_monitor.total_unweighted_loss.avg}')
    print(f'epoch {mode} xy loss: {loss_monitor.xy_loss.avg}')
    print(f'epoch {mode} depth loss: {loss_monitor.depth_loss.avg}')
    print(f'epoch {mode} orientation loss: {loss_monitor.ori_loss.avg}')
    if log_contact:
        print(f'epoch {mode} contact loss: {loss_monitor.contact_loss.avg}')
        print(f'epoch {mode} contact acc: {loss_monitor.contact_acc.avg}')


def write_epoch_loss_to_tb(loss_monitor, writer, tb_step, tag, log_contact=False):
    """Write instance of EpochLossMonitor to tensorboard"""
    writer.add_scalar(f'{tag}/total_loss', loss_monitor.total_loss.avg, tb_step)
    writer.add_scalar(f'{tag}/total_unweighted_loss', loss_monitor.total_unweighted_loss.avg, tb_step)
    writer.add_scalar(f'{tag}/xy_loss', loss_monitor.xy_loss.avg, tb_step)
    writer.add_scalar(f'{tag}/depth_loss', loss_monitor.depth_loss.avg, tb_step)
    writer.add_scalar(f'{tag}/ori_loss', loss_monitor.ori_loss.avg, tb_step)
    if log_contact:
        writer.add_scalar(f'{tag}/contact_loss', loss_monitor.contact_loss.avg, tb_step)
        writer.add_scalar(f'{tag}/contact_acc', loss_monitor.contact_acc.avg, tb_step)


def write_epoch_stats_to_tb(args, writer, epoch, global_step, log_contact=False):
    """Write epoch statistics to tensorboard"""
    writer.add_scalar('stats/epoch_steps', global_step, epoch)
    writer.add_scalar('stats/lambda1', args.lambda1, epoch)
    writer.add_scalar('stats/lambda2', args.lambda2, epoch)
    writer.add_scalar('stats/lambda3', args.lambda3, epoch)
    writer.add_scalar('stats/lambda4', args.lambda4, epoch)
    if log_contact:
        writer.add_scalar('stats/bce_pos_weight', args.bce_pos_weight, epoch)
        writer.add_scalar('stats/bce_weight_mult', args.bce_weight_mult, epoch)


def process_contact_stats(current_contact, future_contact, pred_contact, mean_contact, contact_stats):
    """Calculate contact accuracies and update contact statistics
    Return:
        contact_accs: (contact_acc, cur_contact_acc, mean_contact_acc)
        contact_stats (updated)
    """
    batch_size = future_contact.size(0)

    pred_contact_binary = torch.round(torch.sigmoid(pred_contact))
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

    # using current hand info for prediction as baseline
    cur_contact_correct = (current_contact == future_contact).sum().float()
    cur_contact_acc = cur_contact_correct / batch_size

    cur_pred_contact_pos, cur_pred_contact_neg = current_contact[pos_inds], current_contact[neg_inds]
    cur_n_pos = (cur_pred_contact_pos == future_contact_pos).sum().item()
    cur_n_neg = (cur_pred_contact_neg == future_contact_neg).sum().item()
    contact_stats['pos']['bl_cur'] += cur_n_pos
    contact_stats['neg']['bl_cur'] += cur_n_neg

    # using mean of batch for prediction as baseline
    mean_contact_binary = torch.round(torch.sigmoid(mean_contact))
    mean_contact_correct = (mean_contact_binary == future_contact).sum().float()
    mean_contact_acc = mean_contact_correct / batch_size

    mean_pred_contact_pos, mean_pred_contact_neg = mean_contact_binary[pos_inds], mean_contact_binary[neg_inds]
    mean_n_pos = (mean_pred_contact_pos == future_contact_pos).sum().item()
    mean_n_neg = (mean_pred_contact_neg == future_contact_neg).sum().item()
    contact_stats['pos']['bl_mean'] += mean_n_pos
    contact_stats['neg']['bl_mean'] += mean_n_neg

    contact_accs = (contact_acc, cur_contact_acc, mean_contact_acc)
    return contact_accs, contact_stats


def process_baseline_loss(current_data, target_data, contact_accs, loss_funcs, args, mode='current'):
    """Calculate baseline losses for batch data
    current_data: (current_xy, current_depth, current_ori, current_contact)
    target_data: (target_xy, target_depth, target_ori, future_contact)
    contact_info: (cur_contact_acc, mean_contact_acc)
    loss_funcs: (l2_loss_func, bce_loss_func)
    mode:
        'current': use current hand info for prediction as baseline
        'mean': use mean of batch for prediction as baseline
    Return:
        loss_stats
    """
    target_xy, target_depth, target_ori, future_contact = target_data
    current_xy, current_depth, current_ori, current_contact = current_data
    cur_contact_acc, mean_contact_acc = contact_accs
    l2_loss_func, bce_loss_func = loss_funcs
    batch_size = target_xy.size(0)

    if mode == 'current':
        bl_xy = torch.zeros_like(current_xy) if args.pred_residual else current_xy
        bl_depth = torch.zeros_like(current_depth) if args.pred_residual else current_depth
        bl_ori = torch.zeros_like(current_ori) if args.pred_residual else current_ori
        bl_contact = current_contact
        bl_contact_acc = cur_contact_acc
    elif mode == 'mean':
        mean_x, mean_y = args.data_params['x']['mean'], args.data_params['y']['mean']
        mean_xy = torch.tensor([[mean_x, mean_y]]).repeat(batch_size, 1).to(current_xy.device)
        mean_contact = args.data_params['contact']['mean']
        bl_xy = mean_xy - current_xy if args.pred_residual else mean_xy
        # data is z-score normalized -> mean = 0
        bl_depth = -current_depth if args.pred_residual else torch.zeros(batch_size)
        bl_ori = -current_ori if args.pred_residual else torch.zeros(batch_size, 3)
        bl_contact = torch.tensor(mean_contact).repeat(batch_size).to(current_contact.device) \
            if args.pred_contact else None
        bl_contact_acc = mean_contact_acc
    else:
        raise ValueError(f'Invalid mode: {mode}')

    bl_xy_loss = l2_loss_func(bl_xy, target_xy)
    bl_depth_loss = l2_loss_func(bl_depth, target_depth)
    bl_ori_loss = l2_loss_func(bl_ori, target_ori)
    bl_contact_loss = bce_loss_func(bl_contact, future_contact) if args.pred_contact else 0.

    bl_loss = args.lambda1 * bl_xy_loss + \
              args.lambda2 * bl_depth_loss + \
              args.lambda3 * bl_ori_loss + \
              args.lambda4 * bl_contact_loss
    bl_loss_unweighted = bl_xy_loss + bl_depth_loss + bl_ori_loss + bl_contact_loss

    loss_stats = LossStats(
        total_loss=bl_loss, total_unweighted_loss=bl_loss_unweighted, xy_loss=bl_xy_loss, depth_loss=bl_depth_loss,
        ori_loss=bl_ori_loss, contact_loss=bl_contact_loss if args.pred_contact else None, contact_acc=bl_contact_acc
    )

    return loss_stats


def add_bar_plot_to_tb(
        vals_list, bars, bar_labels, colors,
        bar_width, figsize, text_fontsize, ylim, axis_labels, title,
        tag, epoch, writer, use_float_text=True,
        xticks=None, xticks_bar_width_mult=0.5, use_plt_xticks=True, add_legend=True
):
    """Add success rate bar plot figure to tensorboard"""
    fig = plt.figure(figsize=figsize)
    for vals, b, b_lb, c in zip(vals_list, bars, bar_labels, colors):
        if b_lb:
            plt.bar(b, vals, width=bar_width, color=c, label=b_lb)
        else:
            plt.bar(b, vals, width=bar_width, color=c)
        for pos, val in zip(b, vals):
            plt.text(pos, val / 2, f'{val:.2f}' if use_float_text else val, ha='center', fontsize=text_fontsize)

    xlabel, ylabel = axis_labels
    if xlabel:
        plt.xlabel(xlabel, fontweight='bold', size=13)
    if ylabel:
        plt.ylabel(ylabel, fontweight='bold', size=13)
    if use_plt_xticks:
        assert xticks is not None
        plt.xticks([r + bar_width * xticks_bar_width_mult for r in bars[0]], xticks)
    if ylim:
        plt.ylim(ylim)  # 0% and 100% are success rate limits
    plt.title(title)
    if add_legend:
        plt.legend()
    plt.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def log_epoch_stats(stats, writer, args, global_step, epoch, train=True):
    mode = 'train' if train else 'valid'
    EpochLoss, BLcur, BLmean = stats.LossMonitor, stats.BLcurMonitor, stats.BLmeanMonitor
    EpochMetric, TaskMetric = stats.EpochMetric, stats.TaskMetric
    ContactStats = stats.ContactStats

    # log train epoch stats
    print_epoch_loss(EpochLoss, mode, log_contact=args.pred_contact)
    write_epoch_loss_to_tb(EpochLoss, writer, epoch, tag=f'{mode}_epoch', log_contact=args.pred_contact)
    write_epoch_loss_to_tb(BLcur, writer, epoch, tag=f'{mode}_epoch_baseline_current', log_contact=args.pred_contact)
    write_epoch_loss_to_tb(BLmean, writer, epoch, tag=f'{mode}_epoch_baseline_mean', log_contact=args.pred_contact)
    write_epoch_stats_to_tb(args, writer, epoch, global_step, log_contact=args.pred_contact)

    fig_height = 10
    n_task = len(args.task_names)
    if args.eval_metric:
        # visualize metric evaluation success rate by bar plot
        bar_width = 0.4
        bar1 = np.arange(len(EpochMetric) + 1)
        bar2 = [x + bar_width for x in bar1]
        epoch_pred_success_rate = [safe_div(v['pred_success'], v['total']) for v in EpochMetric.values()]
        epoch_gt_success_rate = [safe_div(v['gt_success'] , v['total']) for v in EpochMetric.values()]
        total_sum = sum([v['total'] for v in EpochMetric.values()])
        pred_success_sum = sum([v['pred_success'] for v in EpochMetric.values()])
        gt_success_sum = sum([v['gt_success'] for v in EpochMetric.values()])
        epoch_pred_success_rate.append(safe_div(pred_success_sum , total_sum))
        epoch_gt_success_rate.append(safe_div(gt_success_sum, total_sum))
        xticks = list(EpochMetric.keys())
        xticks.append('total')

        add_bar_plot_to_tb(
            vals_list=[epoch_pred_success_rate, epoch_gt_success_rate], bars=[bar1, bar2], bar_labels=['pred', 'gt'],
            colors=['peachpuff', 'lavender'], bar_width=bar_width, figsize=(2 * n_task , fig_height),
            text_fontsize=13, ylim=[0, 1], axis_labels=('Tasks', 'Success rate'), title='1D Metric Evaluation',
            tag=f'{mode}_metric/success_rate', epoch=epoch, writer=writer, xticks=xticks
        )

        # visualize metric evaluation count by bar plot
        bar_width = 0.25
        pred_success = [v['pred_success'] for v in EpochMetric.values()]
        gt_success = [v['gt_success'] for v in EpochMetric.values()]
        total = [v['total'] for v in EpochMetric.values()]
        bar1 = np.arange(len(EpochMetric))
        bar2 = [x + bar_width for x in bar1]
        bar3 = [x + bar_width for x in bar2]

        add_bar_plot_to_tb(
            vals_list=[pred_success, gt_success, total], bars=[bar1, bar2, bar3],
            bar_labels=['pred success', 'gt success', 'total'], colors=['peachpuff', 'lavender', 'skyblue'],
            bar_width=bar_width, figsize=(3 * n_task, fig_height), text_fontsize=8, ylim=[],
            axis_labels=['Tasks', 'Count'], title='1D Metric Evaluation', tag=f'{mode}_metric/count',
            epoch=epoch, writer=writer, use_float_text=False, xticks=list(EpochMetric.keys()), xticks_bar_width_mult=1
        )

    # log task conditioning plots
    if not train and args.eval_tasks and args.eval_metric:
        # visualize metric evaluation success rate by bar plot
        bar_width = 0.7
        epoch_success_rate = [safe_div(v['pred_success'], v['total']) for v in TaskMetric.values()]
        total_sum = sum([v['total'] for v in TaskMetric.values()])
        pred_success_sum = sum([v['pred_success'] for v in TaskMetric.values()])
        epoch_success_rate.append(safe_div(pred_success_sum, total_sum))
        xticks = list(TaskMetric.keys())
        xticks.append('total')

        add_bar_plot_to_tb(
            vals_list=[epoch_success_rate], bars=[xticks], bar_labels=[''], colors=['skyblue'],
            bar_width=bar_width, figsize=(2 * n_task, fig_height), text_fontsize=15, ylim=[0, 1],
            axis_labels=('Tasks', 'Success rate'), title='1D Metric Evaluation',
            tag=f'{mode}_eval_tasks_metric/task_conditioning_success_rate', epoch=epoch, writer=writer,
            xticks=None, use_plt_xticks=False, add_legend=False
        )

        # visualize metric evaluation count by bar plot
        bar_width = 0.35
        success = [v['pred_success'] for v in TaskMetric.values()]
        total = [v['total'] for v in TaskMetric.values()]
        bar1 = np.arange(len(TaskMetric))
        bar2 = [x + bar_width for x in bar1]

        add_bar_plot_to_tb(
            vals_list=[success, total], bars=[bar1, bar2], bar_labels=['success', 'total'],
            colors=['peachpuff', 'lavender'], bar_width=bar_width, figsize=(3 * n_task, fig_height),
            text_fontsize=8, ylim=[], axis_labels=['Tasks', 'Count'], title='1D Metric Evaluation',
            tag=f'{mode}_eval_tasks_metric/task_conditioning_count', epoch=epoch, writer=writer,
            use_float_text=False, xticks=list(TaskMetric.keys()), xticks_bar_width_mult=0.5
        )

    # log contact plots
    if args.pred_contact:
        gt_n_pos, gt_n_neg = ContactStats['pos']['gt'], ContactStats['neg']['gt']
        gt_n_total = gt_n_pos + gt_n_neg
        pred_n_pos, pred_n_neg = ContactStats['pos']['pred'], ContactStats['neg']['pred']
        cur_n_pos, cur_n_neg = ContactStats['pos']['bl_cur'], ContactStats['neg']['bl_cur']
        mean_n_pos, mean_n_neg = ContactStats['pos']['bl_mean'], ContactStats['neg']['bl_mean']

        # visualize contact success rate by bar plot
        bar_width = 0.25
        pred_success_rate = [
            safe_div(pred_n_pos, gt_n_pos), safe_div(pred_n_neg, gt_n_neg),
            safe_div((pred_n_pos + pred_n_neg), gt_n_total)
        ]
        cur_success_rate = [
            safe_div(cur_n_pos, gt_n_pos), safe_div(cur_n_neg, gt_n_neg),
            safe_div((cur_n_pos + cur_n_neg), gt_n_total)
        ]
        mean_success_rate = [
            safe_div(mean_n_pos, gt_n_pos), safe_div(mean_n_neg, gt_n_neg),
            safe_div((mean_n_pos + mean_n_neg), gt_n_total)
        ]
        bar1 = np.arange(3)
        bar2 = [x + bar_width for x in bar1]
        bar3 = [x + bar_width for x in bar2]

        add_bar_plot_to_tb(
            vals_list=[pred_success_rate, cur_success_rate, mean_success_rate], bars=[bar1, bar2, bar3],
            bar_labels=['pred', 'bl_cur', 'bl_mean'], colors=['peachpuff', 'lavender', 'skyblue'],
            bar_width=bar_width, figsize=(2 * n_task, fig_height), text_fontsize=8, axis_labels=('', 'Success rate'),
            ylim=[0, 1], title='Contact', tag=f'{mode}_contact/success_rate', epoch=epoch, writer=writer,
            xticks=['pos', 'neg', 'combined'], xticks_bar_width_mult=1, use_plt_xticks=True, add_legend=True
        )

        # visualize contact success count by bar plot
        bar_width = 0.2
        pred_success = [pred_n_pos, pred_n_neg, pred_n_pos + pred_n_neg]
        cur_success = [cur_n_pos, cur_n_neg, cur_n_pos + cur_n_neg]
        mean_success = [mean_n_pos, mean_n_neg, mean_n_pos + mean_n_neg]
        total = [gt_n_pos, gt_n_neg, gt_n_total]
        bar1 = np.arange(3)
        bar2 = [x + bar_width for x in bar1]
        bar3 = [x + bar_width for x in bar2]
        bar4 = [x + bar_width for x in bar3]

        add_bar_plot_to_tb(
            vals_list=[pred_success, cur_success, mean_success, total], bars=[bar1, bar2, bar3, bar4],
            bar_labels=['pred', 'bl_cur', 'bl_mean', 'total'], colors=['peachpuff', 'lavender', 'skyblue', 'wheat'],
            bar_width=bar_width, figsize=(5 * 3, fig_height), text_fontsize=8, ylim=[], axis_labels=['', 'Count'],
            title='Contact', tag=f'{mode}_contact/count', epoch=epoch, writer=writer, use_float_text=False,
            xticks=['pos', 'neg', 'combined'], xticks_bar_width_mult=1.5
        )


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class EpochLossMonitor(object):
    def __init__(self, monitor_contact=False):
        self.monitor_contact = monitor_contact
        self.total_loss = AverageMeter()
        self.total_unweighted_loss = AverageMeter()
        self.xy_loss = AverageMeter()
        self.depth_loss = AverageMeter()
        self.ori_loss = AverageMeter()
        self.contact_loss = AverageMeter()
        self.contact_acc = AverageMeter()

    def update(self, loss_stats, n=1):
        self.total_loss.update(loss_stats.total_loss, n)
        self.total_unweighted_loss.update(loss_stats.total_unweighted_loss, n)
        self.xy_loss.update(loss_stats.xy_loss, n)
        self.depth_loss.update(loss_stats.depth_loss, n)
        self.ori_loss.update(loss_stats.ori_loss, n)
        if self.monitor_contact:
            self.contact_loss.update(loss_stats.contact_loss, n)
            self.contact_acc.update(loss_stats.contact_acc, n)


if __name__ == '__main__':
    test_path_conversion = False
    test_pose_to_joint_z = False
    test_metric = False
    test_bar_plot = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_visualizer = False
    no_future_info = True
    log_tb = False

    if test_path_conversion:
        # test path conversion
        mocap_path = '/home/junyao/Datasets/something_something_processed/' \
                     'push_left_right/train/mocap_output/0/mocap/frame10_prediction_result.pkl'
        rendered_path = mocap_path_to_rendered_path(mocap_path)
        print(rendered_path)

        # test bbox normalization
        future_hand_pose_path = '/home/junyao/Datasets/something_something_processed/' \
                                'push_left_right/valid/mocap_output/108375/mocap/frame31_prediction_result.pkl'
        hand= 'left_hand'
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        img_path = future_hand_info['image_path']
        if img_path[:8] == '/scratch':
            img_path = '/home' + img_path[8:]
        img = cv2.imread(img_path)
        future_hand_bbox = future_hand_info['hand_bbox_list'][0][hand]
        img_size = (img.shape[1], img.shape[0])
        norm_future_hand_bbox = normalize_bbox(future_hand_bbox, img_size)
        unnorm_future_hand_bbox = unnormalize_bbox(norm_future_hand_bbox, img_size)
        print(f'Original bbox: {future_hand_bbox}')
        print(f'Normalized bbox: {norm_future_hand_bbox}')
        print(f'Unnormalized bbox: {unnorm_future_hand_bbox}')

    if test_pose_to_joint_z:
        # test pose to joint z
        run_on_cv_server = True
        frankmocap_path = '/home/junyao/LfHV/frankmocap'
        r3m_path = '/home/junyao/LfHV/r3m'
        sys.path.insert(1, frankmocap_path)
        os.chdir(frankmocap_path)
        device = 'cuda'
        from handmocap.hand_mocap_api import HandMocap

        print('Loading frank mocap hand module.')
        checkpoint_hand = join(
            frankmocap_path,
            'extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
        )
        smpl_dir = join(frankmocap_path, 'extra_data/smpl')
        hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device=device, batch_size=2)
        os.chdir(r3m_path)

        current_hand_pose_path = '/home/junyao/Datasets/something_something_processed/' \
                                 'push_left_right/valid/mocap_output/80757/mocap/frame24_prediction_result.pkl'
        future_hand_pose_path = '/home/junyao/Datasets/something_something_processed/' \
                                'push_left_right/valid/mocap_output/80757/mocap/frame29_prediction_result.pkl'

        hand = ['right_hand', 'right_hand']
        pose_path = [current_hand_pose_path, future_hand_pose_path]

        print('Preprocessing hand data from pkl.')
        with open(current_hand_pose_path, 'rb') as f:
            current_hand_info = pickle.load(f)
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)

        current_hand_pose = current_hand_info['pred_output_list'][0]['right_hand']['pred_hand_pose']
        future_hand_pose = future_hand_info['pred_output_list'][0]['right_hand']['pred_hand_pose']
        pose = np.vstack([current_hand_pose, future_hand_pose])
        pose = torch.Tensor(pose).to(device)

        current_hand_shape = current_hand_info['pred_output_list'][0]['right_hand']['pred_hand_betas']
        future_hand_shape = future_hand_info['pred_output_list'][0]['right_hand']['pred_hand_betas']
        shape = np.vstack([current_hand_shape, future_hand_shape])
        shape = torch.Tensor(shape).to(device)

        current_cam = current_hand_info['pred_output_list'][0]['right_hand']['pred_camera']
        future_cam = future_hand_info['pred_output_list'][0]['right_hand']['pred_camera']
        cam = np.vstack([current_cam, future_cam])
        cam = torch.Tensor(cam).to(device)

        current_hand_bbox = current_hand_info['hand_bbox_list'][0]['right_hand']
        current_image_path = current_hand_info['image_path']
        if current_image_path[:8] == '/scratch' and run_on_cv_server:
            current_image_path = '/home' + current_image_path[8:]
        current_image = cv2.imread(current_image_path)
        current_hand_bbox = normalize_bbox(current_hand_bbox, (current_image.shape[1], current_image.shape[0]))
        current_img_shape = current_image.shape
        future_hand_bbox = future_hand_info['hand_bbox_list'][0]['right_hand']
        future_image_path = future_hand_info['image_path']
        if future_image_path[:8] == '/scratch' and run_on_cv_server:
            future_image_path = '/home' + future_image_path[8:]
        future_image = cv2.imread(future_image_path)
        future_hand_bbox = normalize_bbox(future_hand_bbox, (future_image.shape[1], future_image.shape[0]))
        future_img_shape = future_image.shape
        hand_bbox = np.vstack([current_hand_bbox, future_hand_bbox])
        hand_bbox = torch.Tensor(hand_bbox).to(device)
        img_shape = np.vstack([current_img_shape, future_img_shape])
        img_shape = torch.Tensor(img_shape).to(device).round().int()

        joint_z = pose_to_joint_depth(
            hand_mocap, hand, pose, hand_bbox, cam, img_shape, device,
            shape=None, shape_path=pose_path
        )

        gt_current_joint_z = current_hand_info['pred_output_list'][0]['right_hand']['pred_joints_img'][:, 2]
        gt_future_joint_z = future_hand_info['pred_output_list'][0]['right_hand']['pred_joints_img'][:, 2]
        gt_joint_z = np.vstack([gt_current_joint_z, gt_future_joint_z])
        gt_joint_z = torch.Tensor(gt_joint_z).to(device)

        print(f'joint_z: \n{joint_z}')
        print(f'gt_joint_z: \n{gt_joint_z}')

    if test_metric:
        dataset_dir = '/home/junyao/Datasets/something_something_hand_demos'
        task = 'move_away'
        vid_num = '1'
        hand = 'left_hand'

        current_bbox_json = join(dataset_dir, task, 'bbs_json', vid_num, 'frame10.json')
        current_bbox_dict = json.load(open(current_bbox_json))
        current_bbox = current_bbox_dict['hand_bbox_list'][0][hand]

        future_bbox_json = join(dataset_dir, task, 'bbs_json', vid_num, 'frame20.json')
        future_bbox_dict = json.load(open(future_bbox_json))
        future_bbox = future_bbox_dict['hand_bbox_list'][0][hand]

        passed_metric = evaluate_metric(task_name=task, current_bbox=current_bbox, pred_bbox=future_bbox)
        print(f'Passed metric? {passed_metric}.')

    if test_bar_plot:
        torch.manual_seed(43)
        size = 128
        plot_rate = True
        plot_count = True

        current_x, pred_x = torch.rand(size).cuda(), torch.rand(size).cuda()
        current_y, pred_y = torch.rand(size).cuda(), torch.rand(size).cuda()
        current_depth, pred_depth = torch.rand(size).cuda(), torch.rand(size).cuda()
        ALL_TASKS.remove('push_slightly')
        task_names = ALL_TASKS
        # task_names = CV_TASKS
        task_nums = torch.randint(low=0, high=len(task_names), size=(size,))
        task = torch.nn.functional.one_hot(task_nums)

        metric_stats = evaluate_transferable_metric_batch(
            task_names, task, device,
            current_x, pred_x,
            current_y, pred_y,
            current_depth, pred_depth,
        )

        if plot_rate:
            plt.figure(figsize=(10, 7))
            success_rate = [v['success'] / v['total'] for v in metric_stats.values()]
            plt.bar(list(metric_stats.keys()), success_rate, width=0.7, color='skyblue')
            for i, r in enumerate(success_rate):
                plt.text(i, r / 2, f'{r:.2f}', ha='center', fontsize=15)
            plt.xlabel('Tasks', fontweight='bold', size=13)
            plt.ylabel('Success rate', fontweight='bold', size=13)
            plt.tight_layout()
            plt.savefig('success_rate_bar_plot.png')
            plt.show()
            plt.close()

        if plot_count:
            bar_width = 0.35
            plt.figure(figsize=(10, 7))

            success = [v['success'] for v in metric_stats.values()]
            total = [v['total'] for v in metric_stats.values()]

            bar1 = np.arange(len(metric_stats))
            bar2 = [x + bar_width for x in bar1]

            plt.bar(bar1, success, color='peachpuff', width=bar_width, label='success')
            plt.bar(bar2, total, color='lavender', width=bar_width, label='total')
            for b1, s, b2, t in zip(bar1, success, bar2, total):
                plt.text(b1, s / 2, s, ha='center', fontsize=8)
                plt.text(b2, t / 2, t, ha='center', fontsize=8)

            plt.xlabel('Tasks', fontweight='bold', size=13)
            plt.ylabel('Count', fontweight='bold', size=15)
            plt.xticks([r + bar_width / 2 for r in bar1], list(metric_stats.keys()))

            plt.legend()
            plt.tight_layout()
            plt.savefig('count_bar_plot.png')
            plt.show()
            plt.close()

    pass
