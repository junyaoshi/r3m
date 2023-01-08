import pickle
from collections import namedtuple
from os.path import join

import numpy as np
import torch

HALF_TASKS = [
    'push_left',
    'push_right',
    'move_down',
    'move_up',
]

ALL_TASKS = [
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
    task_name = HALF_TASKS[torch.argmax(cv_task)]
    cluster_task_idx = ALL_TASKS.index(task_name)
    cluster_task = torch.zeros(len(ALL_TASKS))
    cluster_task[cluster_task_idx] = 1
    return cluster_task


def cluster_task_to_cv_task(cluster_task):
    """Convert one-hot cv task to one-hot cluster task"""
    task_name = ALL_TASKS[torch.argmax(cluster_task)]
    cv_task_idx = HALF_TASKS.index(task_name)
    cv_task = torch.zeros(len(HALF_TASKS))
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
        depth_descriptor='scaling_factor',
        depth_real_path=None
):
    # process mocap prediction for AgentTransferable torch dataset
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

    wrist_depth_real = -999  # value for invalid depth due to out of bound wrist joint
    if depth_real_path is not None:
        if (0 <= wrist_x_float < img_x) and (0 <= wrist_y_float < img_y):
            wrist_coord = wrist_coord.round().astype(np.int16)
            wrist_x, wrist_y = wrist_coord
            if wrist_x != img_x and wrist_y != img_y:
                depth_real = np.load(depth_real_path)
                wrist_depth_real = depth_real[wrist_y, wrist_x].astype(np.int16)

    info = namedtuple('info', [
        'wrist_x_normalized',
        'wrist_y_normalized',
        'hand_depth_normalized',
        'hand_depth_real',
        'wrist_orientation',
        'contact',
        'img_shape'
    ])

    return info(
        wrist_x_normalized=wrist_x_normalized,
        wrist_y_normalized=wrist_y_normalized,
        hand_depth_normalized=hand_depth_normalized,
        hand_depth_real=wrist_depth_real,
        wrist_orientation=wrist_orientation_normalized,
        contact=contact,
        img_shape=img_shape
    )


def check_contact_sequence(contact_sequence):
    """Check if contact sequence is valid"""
    valid = True
    contact_keys, contact_vals = list(contact_sequence.keys()), list(contact_sequence.values())
    first_contact_frame, last_contact_frame = -1, -1
    for first_contact_idx, frame in enumerate(contact_keys):
        if contact_sequence[frame] == 1:
            first_contact_frame = frame
            break
    for last_contact_idx, frame in reversed(list(enumerate(contact_keys))):
        if contact_sequence[frame] == 1:
            last_contact_frame = frame
            break
    if first_contact_frame == -1 or last_contact_frame == -1:
        valid = False
    else:
        assert first_contact_frame <= last_contact_frame
        for idx in range(first_contact_idx, last_contact_idx):
            if contact_vals[idx] == 0:
                valid = False

    return valid, first_contact_frame, last_contact_frame


def filter_frames_by_stage(frames, first_contact_frame, last_contact_frame, stage):
    """Filter given frames in a video by the specified stage"""
    assert stage in ['pre', 'during']
    frames_filtered = []

    for frame in frames:
        if stage == 'pre' and frame < first_contact_frame:
            frames_filtered.append(frame)
        elif stage == 'during' and first_contact_frame <= frame <= last_contact_frame:
            frames_filtered.append(frame)

    return frames_filtered

# Depth related functions
def scaling_factor_depth(cam_scale, hand_boxScale_o2n):
    scaling_factor = cam_scale / hand_boxScale_o2n
    depth = 1. / scaling_factor
    return depth


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


def get_sum_val(cur_v, pred_v, args):
    return cur_v + pred_v if args.pred_residual else pred_v


def get_target_val(next_v, cur_v, args):
    return next_v - cur_v if args.pred_residual else next_v


def get_res_val(pred_v, cur_v, args):
    return pred_v if args.pred_residual else pred_v - cur_v
