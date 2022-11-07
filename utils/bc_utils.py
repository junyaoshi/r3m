import json
import pickle
import sys
import time
from os.path import join
import os
import os.path as osp
from pprint import pprint
from collections import namedtuple

import numpy as np
import cv2
import torch

from bc_models.resnet import EndtoEndNet, PartiallyTransferableNet
from utils.depth_utils import scaling_factor_depth


CV_TASKS = [
    'push_left',
    'push_right',
    'move_down',
    'move_up',
]
CLUSTER_TASKS = [
    'move_away',
    'move_towards',
    'move_down',
    'move_up',
    'pull_left',
    'pull_right',
    'push_left',
    'push_right'
]


def cv_task_to_cluster_task(cv_task):
    """Convert one-hot cv task to one-hot cluster task"""
    task_name = CV_TASKS[torch.argmax(cv_task)]
    cluster_task_idx = CLUSTER_TASKS.index(task_name)
    cluster_task = torch.zeros(len(CLUSTER_TASKS))
    cluster_task[cluster_task_idx] = 1
    return cluster_task


def cluster_task_to_cv_task(cluster_task):
    """Convert one-hot cv task to one-hot cluster task"""
    task_name = CLUSTER_TASKS[torch.argmax(cluster_task)]
    cv_task_idx = CV_TASKS.index(task_name)
    cv_task = torch.zeros(len(CV_TASKS))
    cv_task[cv_task_idx] = 1
    return cv_task


def determine_which_hand(hand_info):
    left_hand_exists = len(hand_info['pred_output_list'][0]['left_hand']) > 0
    right_hand_exists = len(hand_info['pred_output_list'][0]['right_hand']) > 0
    if left_hand_exists and not right_hand_exists:
        return 'left_hand'
    if right_hand_exists and not left_hand_exists:
        return 'right_hand'
    if left_hand_exists and right_hand_exists:
        # select the hand with the bigger bounding box
        left_hand_bbox = hand_info['hand_bbox_list'][0]['left_hand']
        *_, lw, lh = left_hand_bbox
        right_hand_bbox = hand_info['hand_bbox_list'][0]['right_hand']
        *_, rw, rh = right_hand_bbox
        if lw * lh >= rw * rh:
            return 'left_hand'
        else:
            return 'right_hand'
    else:
        raise ValueError('No hand detected!')


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def xywh_to_xyxy(xywh_bbox):
    x, y, w, h = xywh_bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return np.array([x1, y1, x2, y2])


def normalize_bbox(unnormalized_bbox, img_size):
    img_x, img_y = img_size
    img_x, img_y = float(img_x), float(img_y)

    bbox_0 = unnormalized_bbox[0] / img_x
    bbox_1 = unnormalized_bbox[1] / img_y
    bbox_2 = unnormalized_bbox[2] / img_x
    bbox_3 = unnormalized_bbox[3] / img_y

    return np.array([bbox_0, bbox_1, bbox_2, bbox_3])


def unnormalize_bbox(normalized_bbox, img_size):
    img_x, img_y = img_size

    bbox_0 = int(round(normalized_bbox[0] * img_x))
    bbox_1 = int(round(normalized_bbox[1] * img_y))
    bbox_2 = int(round(normalized_bbox[2] * img_x))
    bbox_3 = int(round(normalized_bbox[3] * img_y))

    return np.array([bbox_0, bbox_1, bbox_2, bbox_3])


def unnormalize_bbox_batch(normalized_bbox, img_size):
    img_x, img_y = img_size

    bbox_0 = torch.mul(normalized_bbox[:, 0], img_x).round().int()
    bbox_1 = torch.mul(normalized_bbox[:, 1], img_y).round().int()
    bbox_2 = torch.mul(normalized_bbox[:, 2], img_x).round().int()
    bbox_3 = torch.mul(normalized_bbox[:, 3], img_y).round().int()

    return torch.stack([bbox_0, bbox_1, bbox_2, bbox_3]).swapaxes(0, 1)


def mocap_path_to_rendered_path(mocap_path):
    path_split = mocap_path.split('/')
    path_split[-2] = 'rendered'
    path_split[-1] = path_split[-1].split('_')[0] + '.jpg'
    return join(*path_split)


def _convert_smpl_to_bbox(data_z, scale):
    resnet_input_size_half = 224 * 0.5
    data_z = torch.mul(data_z, scale.unsqueeze(1)) # apply scaling
    data_z *= resnet_input_size_half  # 112 is originated from hrm's input size (224,24)
    return data_z


def _convert_bbox_to_oriIm(data_z, img_shape, hand_bbox, final_size=224):
    ori_height, ori_width = img_shape[:, 0], img_shape[:, 1]
    hand_bbox = unnormalize_bbox_batch(hand_bbox, (ori_width, ori_height))
    hand_bbox[:, 2] = torch.min(hand_bbox[:, 2], ori_width - hand_bbox[:, 0])
    hand_bbox[:, 3] = torch.min(hand_bbox[:, 3], ori_height - hand_bbox[:, 1])

    min_x, min_y = hand_bbox[:, 0], hand_bbox[:, 1]
    width, height = hand_bbox[:, 2], hand_bbox[:, 3]
    max_x = min_x + width
    max_y = min_y + height

    for i in range(width.size(0)):
        if width[i] > height[i]:
            margin = torch.div((width[i] - height[i]), 2, rounding_mode='floor')
            min_y[i] = torch.max(min_y[i] - margin, torch.zeros_like(margin))
            max_y[i] = torch.min(max_y[i] + margin, ori_height[i])
        else:
            margin = torch.div((height[i] - width[i]), 2, rounding_mode='floor')
            min_x[i] = torch.max(min_x[i] - margin, torch.zeros_like(margin))
            max_x[i] = torch.min(max_x[i] + margin, ori_width[i])

    margin = (0.3 * (max_y - min_y)).round().int()  # if use loose crop, change 0.3 to 1.0
    min_y = torch.max(min_y - margin, torch.zeros_like(margin))
    max_y = torch.min(max_y + margin, ori_height)
    min_x = torch.max(min_x - margin, torch.zeros_like(margin))
    max_x = torch.min(max_x + margin, ori_width)

    new_size = torch.max(max_x - min_x, max_y - min_y)
    bbox_ratio = final_size / new_size
    data_z = torch.div(data_z, bbox_ratio.unsqueeze(1))

    return data_z


def pose_to_joint_depth(
        hand_mocap,
        hand, pose, bbox, cam, img_shape,
        device,
        shape, shape_path=None
):
    assert not (shape is None and shape_path is None)
    if shape is None:
        shape = []
        for i, p in enumerate(shape_path):
            with open(p, 'rb') as f:
                hand_info = pickle.load(f)
            hand_shape = hand_info['pred_output_list'][0][hand[i]]['pred_hand_betas']
            shape.append(hand_shape)
        shape = np.vstack(shape)
        shape = torch.Tensor(shape).to(device)

    _, joints_smplcoord = hand_mocap.model_regressor.get_smplx_output(pose, shape)
    joints_smplcoord_z = joints_smplcoord[:, :, 2]
    cam_scale = cam[:, 0]
    joints_bboxcoord_z = _convert_smpl_to_bbox(joints_smplcoord_z, cam_scale)  # SMPL space -> bbox space
    joints_imgcoord_z = _convert_bbox_to_oriIm(joints_bboxcoord_z, img_shape, bbox)
    return joints_imgcoord_z


def process_mocap_pred(mocap_pred_path, hand, mocap_pred=None, depth_descriptor='scaling_factor'):
    assert mocap_pred_path is not None or mocap_pred is not None
    if mocap_pred is None:
        with open(mocap_pred_path, 'rb') as f:
            mocap_pred = pickle.load(f)
    hand_info = mocap_pred

    hand_pose = hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
    unnormalized_hand_bbox = hand_info['hand_bbox_list'][0][hand]
    camera = hand_info['pred_output_list'][0][hand]['pred_camera']
    img_shape = hand_info['image_shape']
    hand_bbox = normalize_bbox(unnormalized_hand_bbox, (img_shape[1], img_shape[0]))
    hand_shape = hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)
    wrist_3d = hand_info['pred_output_list'][0][hand]['pred_joints_img'][0]
    wrist_depth_real = -999

    hand_depth_estimate = None
    if depth_descriptor == 'wrist_img_z':
        hand_depth_estimate = wrist_3d[2]
    elif depth_descriptor == 'bbox_size':
        *_, w, h = unnormalized_hand_bbox
        bbox_size = w * h
        hand_depth_estimate = 1. / bbox_size
    elif depth_descriptor == 'scaling_factor':
        cam_scale = camera[0]
        hand_boxScale_o2n = hand_info['pred_output_list'][0][hand]['bbox_scale_ratio']
        scaling_factor = cam_scale / hand_boxScale_o2n
        hand_depth_estimate = 1. / scaling_factor
    elif depth_descriptor == 'normalized_bbox_size':
        *_, w, h = hand_bbox
        normalized_bbox_size = w * h
        hand_depth_estimate = 1. / normalized_bbox_size

    return (
        hand_bbox,
        camera,
        img_shape,
        hand_depth_estimate,
        wrist_depth_real,
        hand_pose,
        hand_shape
    )


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl = pickle.load(f)
    return pkl


def zscore_normalize(data, norm_params):
    if isinstance(data, torch.Tensor):
        device = data.device
        if isinstance(norm_params['mean'], np.ndarray):
            mean = torch.from_numpy(norm_params['mean']).to(device)
            std = torch.from_numpy(norm_params['std']).to(device)
        else:
            mean = torch.tensor(norm_params['mean']).to(device)
            std = torch.tensor(norm_params['std']).to(device)
    else:
        mean = norm_params['mean']
        std = norm_params['std']
    return (data - mean) / std


def zscore_unnormalize(data, norm_params):
    if isinstance(data, torch.Tensor):
        device = data.device
        if isinstance(norm_params['mean'], np.ndarray):
            mean = torch.from_numpy(norm_params['mean']).to(device)
            std = torch.from_numpy(norm_params['std']).to(device)
        else:
            mean = torch.tensor(norm_params['mean']).to(device)
            std = torch.tensor(norm_params['std']).to(device)
    else:
        mean = norm_params['mean']
        std = norm_params['std']
    return data * std + mean


def process_transferable_mocap_pred(
        mocap_pred_path, hand,
        depth_norm_params, ori_norm_params,
        mocap_pred=None,
        depth_descriptor='scaling_factor'
):
    assert mocap_pred_path is not None or mocap_pred is not None
    if mocap_pred is None:
        with open(mocap_pred_path, 'rb') as f:
            mocap_pred = pickle.load(f)
    hand_info = mocap_pred

    hand_pose = hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
    unnormalized_hand_bbox = hand_info['hand_bbox_list'][0][hand]
    camera = hand_info['pred_output_list'][0][hand]['pred_camera']
    img_shape = hand_info['image_shape']
    img_x, img_y = img_shape[1], img_shape[0]
    hand_bbox = normalize_bbox(unnormalized_hand_bbox, (img_x, img_y))
    wrist_3d = hand_info['pred_output_list'][0][hand]['pred_joints_img'][0]
    contact = hand_info['contact_filtered']

    wrist_coord = wrist_3d[:2]
    wrist_x_float, wrist_y_float = wrist_coord

    if depth_descriptor == 'wrist_img_z':
        hand_depth_estimate = wrist_3d[2]
    elif depth_descriptor == 'bbox_size':
        *_, w, h = unnormalized_hand_bbox
        bbox_size = w * h
        hand_depth_estimate = 1. / bbox_size
    elif depth_descriptor == 'scaling_factor':
        cam_scale = camera[0]
        hand_boxScale_o2n = hand_info['pred_output_list'][0][hand]['bbox_scale_ratio']
        hand_depth_estimate = scaling_factor_depth(cam_scale, hand_boxScale_o2n)
    elif depth_descriptor == 'normalized_bbox_size':
        *_, w, h = hand_bbox
        normalized_bbox_size = w * h
        hand_depth_estimate = 1. / normalized_bbox_size
    else:
        raise ValueError(f'Invalid depth descriptor: {depth_descriptor}')

    wrist_x_normalized = wrist_x_float / float(img_x)
    wrist_y_normalized = wrist_y_float / float(img_y)
    hand_depth_normalized = zscore_normalize(hand_depth_estimate, depth_norm_params)
    wrist_orientation_normalized = zscore_normalize(hand_pose[:3], ori_norm_params)

    info = namedtuple('info', [
        'wrist_x_normalized',
        'wrist_y_normalized',
        'hand_depth_normalized',
        'wrist_orientation',
        'contact',
        'img_shape'
    ])

    return info(
        wrist_x_normalized=wrist_x_normalized,
        wrist_y_normalized=wrist_y_normalized,
        hand_depth_normalized=hand_depth_normalized,
        wrist_orientation=wrist_orientation_normalized,
        contact=contact,
        img_shape=img_shape
    )


def load_eval_bc_model_and_args(eval_args, device):
    print('loading the model at:')
    print(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']
    pprint(f'loaded train args: \n{args}')

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


def estimate_depth(eval_args, args, hand_bbox):
    pred_hand_depth_estimate = None
    if eval_args.depth_descriptor == 'wrist_img_z':
        raise NotImplementedError
    elif eval_args.depth_descriptor == 'bbox_size':
        raise NotImplementedError
    elif eval_args.depth_descriptor == 'scaling_factor':
        raise NotImplementedError
    elif eval_args.depth_descriptor == 'normalized_bbox_size':
        assert args.predict_hand_bbox
        # *_, w, h = pred_hand_bbox[0].cpu().detach().numpy()
        *_, w, h = hand_bbox
        pred_normalized_bbox_size = w * h
        pred_hand_depth_estimate = 1. / pred_normalized_bbox_size

    return pred_hand_depth_estimate


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


class AvgrageMeter(object):
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
        import matplotlib.pyplot as plt
        torch.manual_seed(43)
        size = 128
        plot_rate = True
        plot_count = True

        current_x, pred_x = torch.rand(size).cuda(), torch.rand(size).cuda()
        current_y, pred_y = torch.rand(size).cuda(), torch.rand(size).cuda()
        current_depth, pred_depth = torch.rand(size).cuda(), torch.rand(size).cuda()
        CLUSTER_TASKS.remove('push_slightly')
        task_names = CLUSTER_TASKS
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
