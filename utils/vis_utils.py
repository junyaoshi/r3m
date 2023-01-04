import pickle
import os.path as osp
import os
from copy import deepcopy
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.bc_utils import (
    evaluate_metric
)
from utils.data_utils import normalize_bbox, unnormalize_bbox, load_pkl, zscore_normalize, zscore_unnormalize, \
    scaling_factor_depth


def load_img_from_hand_info(hand_info, robot_demos, run_on_cv_server):
    img_path = hand_info['image_path']
    if img_path[:8] == '/scratch' and run_on_cv_server:
        img_path = '/home' + img_path[8:]

    # temporary solution: fix the path in a hard-coding way
    if not osp.exists(img_path):
        # change 'Datasets' to 'WidowX_Datasets'
        img_path_splits = img_path.split('/')
        img_path_splits[3] = 'WidowX_Datasets'
        img_path = '/'.join(img_path_splits)
    if not osp.exists(img_path):
        # change 'WidowX_Datasets' to 'Franka_Datasets'
        img_path_splits = img_path.split('/')
        img_path_splits[3] = 'WidowX_Datasets'
        img_path = '/'.join(img_path_splits)
    assert osp.exists(img_path), f'Image path does not exist: {img_path}'

    if robot_demos:
        # replace frames with robot_frames
        img_path = '/' + osp.join(*list(map(lambda x: x.replace('frames', 'robot_frames'), img_path.split('/'))))
    return cv2.imread(img_path)

def render_bbox_and_hand_pose(
        visualizer, img_original_bgr, hand_bbox_list, mesh_list, use_visualizer
):
    from renderer.image_utils import draw_hand_bbox
    res_img = img_original_bgr.copy()
    res_img = draw_hand_bbox(res_img, hand_bbox_list)
    res_img = visualizer.render_pred_verts(res_img, mesh_list) if use_visualizer else np.zeros_like(res_img)
    return res_img

def extract_hand_bbox_and_mesh_list(hand_info, hand):
    from ss_utils.filter_utils import extract_pred_mesh_list
    other_hand = 'left_hand' if hand == 'right_hand' else 'right_hand'
    hand_bbox_list = hand_info['hand_bbox_list']
    hand_bbox_list[0][other_hand] = None
    hand_info['pred_output_list'][0][other_hand] = None
    mesh_list = extract_pred_mesh_list(hand_info)
    return hand_bbox_list, mesh_list

def generate_single_visualization(
        current_hand_pose_path,
        future_hand_pose_path,
        future_cam,
        hand,
        pred_hand_bbox,
        pred_hand_pose,
        pred_hand_shape,
        task_names,
        task,
        visualizer,
        hand_mocap,
        use_visualizer,
        run_on_cv_server,
        current_hand_info=None,
        current_img=None,
        future_img=None,
        original_task=False,
        robot_demos=False,
        log_depth=False,
        log_metric=False,
        current_depth=None,
        future_depth=None,
        pred_depth=None,
        pre_interaction=False,
        object_label=None
):

    from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm

    if current_hand_info is None:
        assert current_hand_pose_path is not None
        with open(current_hand_pose_path, 'rb') as f:
            current_hand_info = pickle.load(f)
    current_hand_bbox_list, current_mesh_list = extract_hand_bbox_and_mesh_list(
        current_hand_info, hand
    )
    if current_img is None:
        current_img = load_img_from_hand_info(current_hand_info, robot_demos, run_on_cv_server)
    current_rendered_img = render_bbox_and_hand_pose(
        visualizer, current_img, current_hand_bbox_list, current_mesh_list, use_visualizer
    )

    if pre_interaction:
        assert object_label is not None

    img_h, img_w, _ = current_img.shape
    pred_hand_bbox = unnormalize_bbox(pred_hand_bbox.detach().cpu().numpy(), (img_w, img_h))
    pred_hand_bbox[2] = min(pred_hand_bbox[2], img_w - pred_hand_bbox[0])
    pred_hand_bbox[3] = min(pred_hand_bbox[3], img_h - pred_hand_bbox[1])
    pred_hand_bbox_list = deepcopy(current_hand_bbox_list)
    pred_hand_bbox_list[0][hand] = pred_hand_bbox

    #  get predicted smpl verts and joints,
    if hand == 'left_hand':
        pred_hand_pose[1::3] *= -1
        pred_hand_pose[2::3] *= -1
    pred_verts, _ = hand_mocap.model_regressor.get_smplx_output(
        pred_hand_pose.unsqueeze(0),
        pred_hand_shape.unsqueeze(0)
    )

    # Convert vertices into bbox & image space
    pred_verts_origin = pred_verts.detach().cpu().numpy()[:, hand_mocap.model_regressor.right_hand_verts_idx, :][0]
    if hand == 'left_hand':
        pred_verts_origin[:, 0] *= -1
    cam_scale = future_cam[0]
    cam_trans = future_cam[1:]
    vert_smplcoord = pred_verts_origin.copy()
    vert_bboxcoord = convert_smpl_to_bbox(
        vert_smplcoord, cam_scale, cam_trans, bAppTransFirst=True
    )  # SMPL space -> bbox space
    *_, bbox_scale_ratio, bbox_processed = hand_mocap.process_hand_bbox(
        current_img, pred_hand_bbox, hand, add_margin=True
    )
    bbox_top_left = np.array(bbox_processed[:2])
    vert_imgcoord = convert_bbox_to_oriIm(
        vert_bboxcoord, bbox_scale_ratio, bbox_top_left,
        img_w, img_h
    )
    pred_hand_info = deepcopy(current_hand_info)
    pred_hand_info['pred_output_list'][0][hand]['pred_vertices_img'] = vert_imgcoord

    _, pred_mesh_list = extract_hand_bbox_and_mesh_list(
        pred_hand_info, hand
    )

    has_future_label = future_hand_pose_path is not None
    if has_future_label:
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        future_hand_bbox_list, future_mesh_list = extract_hand_bbox_and_mesh_list(
            future_hand_info, hand
        )
        if future_img is None:
            future_img = load_img_from_hand_info(future_hand_info, robot_demos, run_on_cv_server)
        future_rendered_img = render_bbox_and_hand_pose(
            visualizer, future_img, future_hand_bbox_list, future_mesh_list, use_visualizer
        )
        pred_rendered_img = render_bbox_and_hand_pose(
            visualizer, future_img,
            pred_hand_bbox_list, pred_mesh_list, use_visualizer
        )
        vis_img_bgr = np.vstack([current_rendered_img, pred_rendered_img, future_rendered_img]).astype(np.uint8)
    else:
        pred_rendered_img = render_bbox_and_hand_pose(
            visualizer, current_img,
            pred_hand_bbox_list, pred_mesh_list, use_visualizer
        )
        vis_img_bgr = np.vstack([current_rendered_img, pred_rendered_img]).astype(np.uint8)


    white_y = 50
    next_line = 40
    if log_depth:
        assert current_depth is not None and pred_depth is not None
        if has_future_label:
            assert future_depth is not None
            white_y += 150
        else:
            white_y += 100
    if log_metric:
        white_y += 50
    white = np.zeros((white_y, vis_img_bgr.shape[1], 3), np.uint8)
    white[:] = (255, 255, 255)
    vis_img_bgr = cv2.vconcat((white, vis_img_bgr))
    font = cv2.FONT_HERSHEY_COMPLEX
    task_name = task_names[np.argmax(task)]
    task_str = "Original Task" if original_task else "Task"
    task_desc_str = f'{task_str}: {task_name}'
    cv2.putText(vis_img_bgr, task_desc_str, (30, next_line), font, 0.6, (0, 0, 0), 2, 0)
    next_line += 50
    if log_depth:
        current_depth_str = f'Current Depth: {current_depth:.4f}'
        cv2.putText(vis_img_bgr, current_depth_str, (30, next_line), font, 0.6, (0, 0, 0), 2, 0)
        next_line += 50
        pred_depth_str = f'Predicted Depth: {pred_depth:.4f}'
        cv2.putText(vis_img_bgr, pred_depth_str, (30, next_line), font, 0.6, (0, 0, 0), 2, 0)
        next_line += 50
        if has_future_label:
            future_depth_str = f'Future Depth: {future_depth:.4f}'
            cv2.putText(vis_img_bgr, future_depth_str, (30, 190), font, 0.6, (0, 0, 0), 2, 0)
            next_line += 50
    passed_metric = None
    if log_metric:
        passed_metric = evaluate_metric(
            task_name=task_name,
            current_bbox=current_hand_bbox_list[0][hand],
            pred_bbox=pred_hand_bbox,
            current_depth=current_depth,
            pred_depth=pred_depth
        )
        metric_str = f'Passed metric: {passed_metric}'
        cv2.putText(vis_img_bgr, metric_str, (30, next_line), font, 0.6, (0, 0, 0), 2, 0)

    vis_img = cv2.cvtColor(vis_img_bgr, cv2.COLOR_BGR2RGB)
    return vis_img.astype(np.uint8), passed_metric


def generate_pred_hand_info(
        current_hand_info,
        current_hand_bbox_list,
        current_img,
        current_depth,
        pred_delta_x,
        pred_delta_y,
        pred_delta_depth,
        pred_delta_ori,
        depth_norm_params,
        ori_norm_params,
        hand,
        hand_mocap,
        visualizer,
        use_visualizer,
        device
):
    # process current hand info
    current_hand_bbox = current_hand_bbox_list[0][hand]
    cur_bbox_x, cur_bbox_y, cur_bbox_w, cur_bbox_h = current_hand_bbox  # xywh format
    current_hand_pose = torch.from_numpy(current_hand_info['pred_output_list'][0][hand]['pred_hand_pose']).to(device)
    current_ori = current_hand_pose[0, :3]
    current_hand_shape = torch.from_numpy(current_hand_info['pred_output_list'][0][hand]['pred_hand_betas']).to(device)
    current_wrist_x, current_wrist_y = current_hand_info['pred_output_list'][0][hand]['pred_joints_img'][0][:2]
    current_camera = current_hand_info['pred_output_list'][0][hand]['pred_camera']
    cam_scale = current_camera[0]
    cam_trans = current_camera[1:]

    # process image shape info
    img_h, img_w, _ = current_img.shape
    img_center_x, img_center_y = img_w / 2, img_h / 2

    # predicted hand info
    pred_hand_bbox_list = deepcopy(current_hand_bbox_list)
    pred_ori_normalized = zscore_normalize(current_ori, ori_norm_params) + pred_delta_ori
    pred_ori = zscore_unnormalize(pred_ori_normalized, ori_norm_params)
    pred_hand_pose = deepcopy(current_hand_pose)
    pred_hand_pose[0, :3] = pred_ori

    # process and clip predicted dx and dy
    pred_delta_x *= img_w
    pred_delta_y *= img_h
    pred_wrist_x, pred_wrist_y = current_wrist_x + pred_delta_x, current_wrist_y + pred_delta_y
    delta_x_min, delta_x_max = -cur_bbox_x, img_w - (cur_bbox_x + cur_bbox_w)
    delta_y_min, delta_y_max = -cur_bbox_y, img_h - (cur_bbox_y + cur_bbox_h)
    pred_delta_x_clipped = np.clip(pred_delta_x, delta_x_min, delta_x_max)
    pred_delta_y_clipped = np.clip(pred_delta_y, delta_y_min, delta_y_max)
    pred_bbox_x_clipped = cur_bbox_x + pred_delta_x_clipped
    pred_bbox_y_clipped = cur_bbox_y + pred_delta_y_clipped

    # process and clip predicted depth
    pred_depth_normalized = zscore_normalize(current_depth, depth_norm_params) + pred_delta_depth
    pred_depth = zscore_unnormalize(pred_depth_normalized, depth_norm_params)
    bbox_scale = current_depth / pred_depth if pred_depth > 0 else np.inf  # larger depth -> smaller bbox
    pred_center_x, pred_center_y = pred_bbox_x_clipped + cur_bbox_w / 2, pred_bbox_y_clipped + cur_bbox_h / 2

    max_w_scale = (img_w - pred_center_x) / (cur_bbox_w / 2) if pred_center_x > img_center_x \
        else pred_center_x / (cur_bbox_w / 2)
    max_h_scale = (img_h - pred_center_y) / (cur_bbox_h / 2) if pred_center_y > img_center_y \
        else pred_center_y / (cur_bbox_h / 2)
    max_bbox_scale = min(max_w_scale, max_h_scale)
    max_bbox_scale = min(max_bbox_scale, 50 * current_depth)  # so that pred_depth is lowered bounded by 0.01
    bbox_scale_clipped = min(bbox_scale, max_bbox_scale)
    pred_depth_clipped = current_depth / bbox_scale_clipped

    pred_bbox_w_clipped, pred_bbox_h_clipped = cur_bbox_w * bbox_scale_clipped, cur_bbox_h * bbox_scale_clipped
    pred_bbox_x_clipped -= (pred_bbox_w_clipped - cur_bbox_w) / 2
    pred_bbox_y_clipped -= (pred_bbox_h_clipped - cur_bbox_h) / 2

    # processed and clipped hand bbox
    pred_hand_bbox = np.array([pred_bbox_x_clipped, pred_bbox_y_clipped, pred_bbox_w_clipped, pred_bbox_h_clipped])
    pred_hand_bbox_list[0][hand] = pred_hand_bbox

    # get predicted smpl verts and joints,
    if hand == 'left_hand':
        pred_hand_pose[:, 1::3] *= -1
        pred_hand_pose[:, 2::3] *= -1
    pred_verts, pred_joints = hand_mocap.model_regressor.get_smplx_output(
        pred_hand_pose,
        current_hand_shape
    )
    pred_joints = pred_joints.detach().cpu().numpy()[0]

    # Convert joints and vertices into bbox & image space, and generate predicted mesh
    from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
    pred_verts_origin = pred_verts.detach().cpu().numpy()[:, hand_mocap.model_regressor.right_hand_verts_idx, :][0]
    if hand == 'left_hand':
        pred_verts_origin[:, 0] *= -1
        pred_joints[:, 0] *= -1
    vert_smplcoord = pred_verts_origin.copy()
    joints_smplcoord = pred_joints.copy()

    vert_bboxcoord = convert_smpl_to_bbox(vert_smplcoord, cam_scale, cam_trans, bAppTransFirst=True)
    joints_bboxcoord = convert_smpl_to_bbox(joints_smplcoord, cam_scale, cam_trans, bAppTransFirst=True)

    *_, pred_bbox_scale_ratio, pred_bbox_processed = hand_mocap.process_hand_bbox(
        current_img, pred_hand_bbox, hand, add_margin=True
    )
    pred_bbox_top_left = np.array(pred_bbox_processed[:2])
    vert_imgcoord = convert_bbox_to_oriIm(
        vert_bboxcoord, pred_bbox_scale_ratio, pred_bbox_top_left,
        img_w, img_h
    )
    joints_imgcoord = convert_bbox_to_oriIm(
        joints_bboxcoord, pred_bbox_scale_ratio, pred_bbox_top_left,
        img_w, img_h
    )
    vis_wrist_x, vis_wrist_y = joints_imgcoord[0][:2]

    pred_hand_info = deepcopy(current_hand_info)
    pred_hand_info['pred_output_list'][0][hand]['pred_vertices_img'] = vert_imgcoord
    _, pred_mesh_list = extract_hand_bbox_and_mesh_list(pred_hand_info, hand)
    pred_rendered_img = render_bbox_and_hand_pose(
        visualizer, current_img,
        pred_hand_bbox_list, pred_mesh_list, use_visualizer
    )

    return(
        pred_wrist_x, pred_wrist_y, pred_depth, pred_depth_clipped, pred_ori, pred_rendered_img,
        vis_wrist_x, vis_wrist_y,
    )


def generate_transferable_visualization(
        current_hand_pose_path,
        future_hand_pose_path,
        run_on_cv_server,
        hand,
        pred_delta_x,
        pred_delta_y,
        pred_delta_depth,
        pred_delta_ori,  # delta wrist joint rotation, needs to be on Cuda
        pred_contact,
        depth_norm_params,
        ori_norm_params,
        task_name,
        visualizer,
        use_visualizer,
        hand_mocap,
        device,
        current_hand_info=None,
        current_img=None,
        future_img=None,
        original_task=False,
        vis_robot=False,
        log_metric=False,
        passed_metric=False,
        current_depth=None,
        future_depth=None,
        vis_groundtruth=True
):
    # vis for current frame
    if current_hand_info is None:
        assert current_hand_pose_path is not None
        with open(current_hand_pose_path, 'rb') as f:
            current_hand_info = pickle.load(f)
    current_hand_bbox_list, current_mesh_list = extract_hand_bbox_and_mesh_list(
        hand_info=current_hand_info,
        hand=hand
    )
    if current_img is None:
        current_img = load_img_from_hand_info(
            hand_info=current_hand_info,
            robot_demos=vis_robot,
            run_on_cv_server=run_on_cv_server
        )
    current_rendered_img = render_bbox_and_hand_pose(
        visualizer=visualizer,
        img_original_bgr=current_img,
        hand_bbox_list=current_hand_bbox_list,
        mesh_list=current_mesh_list,
        use_visualizer=use_visualizer
    )

    # process current hand info
    img_h, img_w, _ = current_img.shape
    current_hand_pose = torch.from_numpy(current_hand_info['pred_output_list'][0][hand]['pred_hand_pose']).to(device)
    current_ori = current_hand_pose[0, :3]
    current_camera = current_hand_info['pred_output_list'][0][hand]['pred_camera']
    cam_scale = current_camera[0]
    current_contact = current_hand_info['contact_filtered']
    current_wrist_x, current_wrist_y = current_hand_info['pred_output_list'][0][hand]['pred_joints_img'][0][:2]
    if current_depth is None:
        current_bbox_scale_ratio = current_hand_info['pred_output_list'][0][hand]['bbox_scale_ratio']
        current_depth = scaling_factor_depth(cam_scale, current_bbox_scale_ratio)

    pred_hand_tuple = generate_pred_hand_info(
        current_hand_info,
        current_hand_bbox_list,
        current_img,
        current_depth,
        pred_delta_x,
        pred_delta_y,
        pred_delta_depth,
        pred_delta_ori,
        depth_norm_params,
        ori_norm_params,
        hand,
        hand_mocap,
        visualizer,
        use_visualizer,
        device
    )
    (
        pred_wrist_x, pred_wrist_y, pred_depth, pred_depth_clipped, pred_ori, pred_rendered_img,
        vis_wrist_x, vis_wrist_y,
    ) = pred_hand_tuple

    # processing if there is future label
    has_future_label = future_hand_pose_path is not None
    if vis_groundtruth:
        assert has_future_label, "Needs future label to visualize ground truth"
    next_bbox_x, next_bbox_y, next_contact = None, None, None
    future_wrist_x, future_wrist_y, future_ori = None, None, None
    gt_wrist_x, gt_wrist_y, gt_depth_clipped = None, None, None
    if has_future_label:
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        future_hand_bbox_list, future_mesh_list = extract_hand_bbox_and_mesh_list(future_hand_info, hand)
        future_hand_bbox = future_hand_bbox_list[0][hand]
        next_bbox_x, next_bbox_y, *_ = future_hand_bbox  # xywh format
        future_ori = future_hand_info['pred_output_list'][0][hand]['pred_hand_pose'][0, :3]
        next_contact = future_hand_info['contact_filtered']
        future_wrist_x, future_wrist_y = future_hand_info['pred_output_list'][0][hand]['pred_joints_img'][0][:2]
        if future_img is None:
            future_img = load_img_from_hand_info(
                hand_info=future_hand_info,
                robot_demos=vis_robot,
                run_on_cv_server=run_on_cv_server
            )
        future_rendered_img = render_bbox_and_hand_pose(
            visualizer, future_img, future_hand_bbox_list, future_mesh_list, use_visualizer
        )
        if vis_groundtruth:
            if future_depth is None:
                future_cam_scale = future_hand_info['pred_output_list'][0][hand]['pred_camera'][0]
                future_bbox_scale_ratio = future_hand_info['pred_output_list'][0][hand]['bbox_scale_ratio']
                future_depth = scaling_factor_depth(future_cam_scale, future_bbox_scale_ratio)
            gt_delta_x = future_wrist_x / float(img_w) - current_wrist_x / float(img_w)
            gt_delta_y = future_wrist_y / float(img_h) - current_wrist_y / float(img_h)
            gt_delta_depth = zscore_normalize(future_depth, depth_norm_params) - \
                             zscore_normalize(current_depth, depth_norm_params)
            gt_delta_ori = zscore_normalize(torch.from_numpy(future_ori).to(device), ori_norm_params) - \
                           zscore_normalize(current_ori, ori_norm_params)

            gt_hand_tuple = generate_pred_hand_info(
                current_hand_info,
                current_hand_bbox_list,
                current_img,
                current_depth,
                gt_delta_x,
                gt_delta_y,
                gt_delta_depth,
                gt_delta_ori,
                depth_norm_params,
                ori_norm_params,
                hand,
                hand_mocap,
                visualizer,
                use_visualizer,
                device
            )
            *_, gt_depth_clipped, _, gt_rendered_img, gt_wrist_x, gt_wrist_y = gt_hand_tuple
            vis_img_bgr = np.vstack(
                [current_rendered_img, pred_rendered_img, gt_rendered_img, future_rendered_img]
            ).astype(np.uint8)
        else:
            vis_img_bgr = np.vstack(
                [current_rendered_img, pred_rendered_img, future_rendered_img]
            ).astype(np.uint8)
    else:
        vis_img_bgr = np.vstack(
            [current_rendered_img, pred_rendered_img]
        ).astype(np.uint8)

    # initialize vis header
    white_top_height = 500
    white_left_width = 110
    next_line = 40
    desc_left_align = 20
    header_left_align = desc_left_align
    text_size = 0.6
    color = (0, 0, 0)
    thickness = 2
    if has_future_label:
        white_top_height += 50
    if log_metric:
        white_top_height += 50
    white_top = np.zeros((white_top_height, vis_img_bgr.shape[1], 3), np.uint8)
    white_top[:] = (255, 255, 255)
    vis_img_bgr = cv2.vconcat((white_top, vis_img_bgr))
    white_left = np.zeros((vis_img_bgr.shape[0], white_left_width, 3), np.uint8)
    white_left[:] = (255, 255, 255)
    vis_img_bgr = cv2.hconcat((white_left, vis_img_bgr))

    # add task string to vis header
    font = cv2.FONT_HERSHEY_COMPLEX
    task_str = "Original Task" if original_task else "Task"
    task_desc_str = f'{task_str}: {task_name}'
    cv2.putText(vis_img_bgr, task_desc_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50

    # add xy string to vis header
    cur_next_x_str = f'x cur: {int(current_wrist_x)}'
    if has_future_label:
        cur_next_x_str += f', x next: {int(future_wrist_x)}'
        if vis_groundtruth:
            cur_next_x_str += f', x gt: {int(gt_wrist_x)}'
    cv2.putText(vis_img_bgr, cur_next_x_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50
    pred_x_str = f'x pred: {int(pred_wrist_x)}, x vis: {int(vis_wrist_x)}'
    cv2.putText(vis_img_bgr, pred_x_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50

    cur_next_y_str = f'y cur: {int(current_wrist_y)}'
    if has_future_label:
        cur_next_y_str += f', y next: {int(future_wrist_y)}'
        if vis_groundtruth:
            cur_next_y_str += f', y gt: {int(gt_wrist_y)}'
    cv2.putText(vis_img_bgr, cur_next_y_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50
    pred_y_str = f'y pred: {int(pred_wrist_y)}, y vis: {int(vis_wrist_y)}'
    cv2.putText(vis_img_bgr, pred_y_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50

    # add depth string to vis header
    cur_next_depth_str = f'z cur: {current_depth:.2f}'
    if has_future_label:
        cur_next_depth_str += f', z next: {future_depth:.2f}'
        if vis_groundtruth:
            cur_next_depth_str += f', z gt: {gt_depth_clipped:.2f}'
    cv2.putText(vis_img_bgr, cur_next_depth_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50
    pred_depth_str = f'z pred: {pred_depth:.2f}, z vis: {pred_depth_clipped:.2f}'
    cv2.putText(vis_img_bgr, pred_depth_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50

    # add orientation string to vis header
    current_ori1, current_ori2, current_ori3 = current_ori
    cur_ori_str = f'ori cur: {current_ori1:.2f}, {current_ori2:.2f}, {current_ori3:.2f}'
    cv2.putText(vis_img_bgr, cur_ori_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50
    pred_ori1, pred_ori2, pred_ori3 = pred_ori
    pred_ori_str = f'ori pred: {pred_ori1:.2f}, {pred_ori2:.2f}, {pred_ori3:.2f}'
    cv2.putText(vis_img_bgr, pred_ori_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50
    if has_future_label:
        future_ori1, future_ori2, future_ori3 = future_ori
        future_ori_str = f'ori next: {future_ori1:.2f}, {future_ori2:.2f}, {future_ori3:.2f}'
        cv2.putText(vis_img_bgr, future_ori_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
        next_line += 50

    # add contact string to vis header
    contact_str = f'Contact: cur {current_contact}'
    if has_future_label:
        contact_str += f', next {next_contact}'
    contact_str += f', pred {int(pred_contact)}'
    cv2.putText(vis_img_bgr, contact_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50

    # add metric string to vis header
    if log_metric:
        metric_str = f'Passed metric: {passed_metric}'
        cv2.putText(vis_img_bgr, metric_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
        next_line += 50

    # add image description to blank space on the left
    cv2.putText(vis_img_bgr, 'current', (desc_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += img_h
    cv2.putText(vis_img_bgr, 'pred', (desc_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += img_h
    if has_future_label:
        if vis_groundtruth:
            cv2.putText(vis_img_bgr, 'gt', (desc_left_align, next_line), font, text_size, color, thickness, 0)
            next_line += img_h
        cv2.putText(vis_img_bgr, 'future', (desc_left_align, next_line), font, text_size, color, thickness, 0)
        next_line += img_h

    vis_img = cv2.cvtColor(vis_img_bgr, cv2.COLOR_BGR2RGB)
    return vis_img.astype(np.uint8)


if __name__ == '__main__':
    test_vis= False
    test_transferable_vis = True
    test_robot_vis = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_visualizer = True
    has_future_label = True
    log_tb = False

    if test_vis:
        # test visualization
        frankmocap_path = '/home/junyao/LfHV/frankmocap'
        r3m_path = '/home/junyao/LfHV/r3m'
        sys.path.insert(1, frankmocap_path)
        os.chdir(frankmocap_path)
        from handmocap.hand_mocap_api import HandMocap

        if use_visualizer:
            print('Loading opengl visualizer.')
            from renderer.visualizer import Visualizer

            visualizer = Visualizer('opengl')
        else:
            visualizer = None
        print('Loading frank mocap hand module.')
        checkpoint_hand = osp.join(
            frankmocap_path,
            'extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
        )
        smpl_dir = osp.join(frankmocap_path, 'extra_data/smpl')
        hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device='cuda')
        os.chdir(r3m_path)

        current_hand_pose_path = '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output' \
                                 '/220825/mocap/frame24_prediction_result.pkl'
        future_hand_pose_path = '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output' \
                                 '/220825/mocap/frame34_prediction_result.pkl'
        hand = 'right_hand'

        print('Preprocessing hand data from pkl.')
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        future_image_path = future_hand_info['image_path']
        if future_image_path[:8] == '/scratch':
            future_image_path = '/home' + future_image_path[8:]
        future_image = cv2.imread(future_image_path)
        future_hand_pose = future_hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        future_hand_bbox = normalize_bbox(
            future_hand_info['hand_bbox_list'][0][hand],
            (future_image.shape[1], future_image.shape[0])
        )
        future_camera = future_hand_info['pred_output_list'][0][hand]['pred_camera']
        future_joint_depth = future_hand_info['pred_output_list'][0][hand]['pred_joints_img'][:, 2]
        future_hand_shape = future_hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)

        task_names = [
            'move_away',
            'move_towards',
            'move_down',
            'move_up',
            'pull_left',
            'pull_right',
            'push_left',
            'push_right',
            'push_slightly',
        ]
        task = np.zeros(len(task_names))
        task[2] = 1

        print('Begin visualization.')
        vis_img = generate_single_visualization(
            current_hand_pose_path=current_hand_pose_path,
            future_hand_pose_path=future_hand_pose_path if has_future_label else None,
            future_cam=future_camera,
            hand=hand,
            pred_hand_bbox=torch.from_numpy(future_hand_bbox).to(device),
            pred_hand_pose=torch.from_numpy(future_hand_pose).to(device),
            pred_hand_shape=torch.from_numpy(future_hand_shape).to(device),
            task_names=task_names,
            task=task,
            visualizer=visualizer,
            hand_mocap=hand_mocap,
            use_visualizer=use_visualizer,
            run_on_cv_server=True,
            log_depth=True,
            log_metric=True,
            current_depth=1,
            future_depth=future_joint_depth.mean(),
            pred_depth=1
        )

        plt_path = 'vis_img.png'
        plt.figure()
        plt.imshow(vis_img)
        plt.savefig('vis_img.png')
        # plt.show()
        plt.close()
        print(f'Saved visualization image to {plt_path}.')

        if log_tb:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir='tb_vis_test')
            writer.add_scalar('loss', 3, 1)
            writer.add_image(f'vis_images', vis_img, 1, dataformats='HWC')
            print(f'Saved tensorboard visualization image to tb_vis_test.')

    if test_transferable_vis:
        use_visualizer = True

        # test visualization
        frankmocap_path = '/home/junyao/LfHV/frankmocap'
        r3m_path = '/home/junyao/LfHV/r3m'
        sys.path.insert(1, frankmocap_path)
        os.chdir(frankmocap_path)
        from handmocap.hand_mocap_api import HandMocap

        if use_visualizer:
            print('Loading opengl visualizer.')
            from renderer.visualizer import Visualizer

            visualizer = Visualizer('opengl')
        else:
            visualizer = None
        print('Loading frank mocap hand module.')
        checkpoint_hand = osp.join(
            frankmocap_path,
            'extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
        )
        smpl_dir = osp.join(frankmocap_path, 'extra_data/smpl')
        hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device='cuda')
        os.chdir(r3m_path)

        depth_norm_params_path = '/home/junyao/LfHV/frankmocap/ss_utils/depth_normalization_params.pkl'
        ori_norm_params_path = '/home/junyao/LfHV/frankmocap/ss_utils/ori_normalization_params.pkl'
        depth_descriptor = 'scaling_factor'
        depth_norm_params = load_pkl(depth_norm_params_path)[depth_descriptor]
        ori_norm_params = load_pkl(ori_norm_params_path)

        # # right hand test data
        # current_hand_pose_path = '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output' \
        #                          '/192254/mocap/frame40_prediction_result.pkl'
        # future_hand_pose_path = '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output' \
        #                          '/192254/mocap/frame50_prediction_result.pkl'
        # hand = 'right_hand'

        # left hand test data
        current_hand_pose_path = '/home/junyao/Datasets/something_something_processed/push_right/valid/mocap_output' \
                                 '/209007/mocap/frame40_prediction_result.pkl'
        future_hand_pose_path = '/home/junyao/Datasets/something_something_processed/push_right/valid/mocap_output' \
                                 '/209007/mocap/frame50_prediction_result.pkl'
        hand = 'left_hand'

        print('Preprocessing hand data from pkl.')
        with open(current_hand_pose_path, 'rb') as f:
            current_hand_info = pickle.load(f)
        img_h, img_w = current_hand_info['image_shape']
        curret_cam_scale = current_hand_info['pred_output_list'][0][hand]['pred_camera'][0]
        curret_hand_boxScale_o2n = current_hand_info['pred_output_list'][0][hand]['bbox_scale_ratio']
        current_depth = scaling_factor_depth(curret_cam_scale, curret_hand_boxScale_o2n)
        current_ori = current_hand_info['pred_output_list'][0][hand]['pred_hand_pose'][0, :3]
        current_wrist_x, current_wrist_y = current_hand_info['pred_output_list'][0][hand]['pred_joints_img'][0][:2]
        current_wrist_x, current_wrist_y = current_wrist_x / float(img_w), current_wrist_y / float(img_h)

        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        future_camera = future_hand_info['pred_output_list'][0][hand]['pred_camera']
        future_joint_depth = future_hand_info['pred_output_list'][0][hand]['pred_joints_img'][:, 2]
        future_hand_shape = future_hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)
        future_cam_scale = future_hand_info['pred_output_list'][0][hand]['pred_camera'][0]
        future_hand_boxScale_o2n = future_hand_info['pred_output_list'][0][hand]['bbox_scale_ratio']
        future_depth = scaling_factor_depth(future_cam_scale, future_hand_boxScale_o2n)
        future_ori = future_hand_info['pred_output_list'][0][hand]['pred_hand_pose'][0, :3]
        future_wrist_x, future_wrist_y = future_hand_info['pred_output_list'][0][hand]['pred_joints_img'][0][:2]
        future_wrist_x, future_wrist_y = future_wrist_x / float(img_w), future_wrist_y / float(img_h)

        delta_x = future_wrist_x - current_wrist_x
        delta_y = future_wrist_y - current_wrist_y
        delta_depth = zscore_normalize(future_depth, depth_norm_params) - \
                      zscore_normalize(current_depth, depth_norm_params)
        delta_ori = zscore_normalize(future_ori, ori_norm_params) - \
                    zscore_normalize(current_ori, ori_norm_params)
        # delta_ori = np.array([100, -100, 1000])
        delta_ori = torch.from_numpy(delta_ori).to(device)
        contact = future_hand_info['contact_filtered']

        task_name = 'move_away'

        print('Begin visualization.')
        vis_img = generate_transferable_visualization(
            current_hand_pose_path=current_hand_pose_path,
            future_hand_pose_path=future_hand_pose_path if has_future_label else None,
            run_on_cv_server=True,
            hand=hand,
            pred_delta_x=delta_x,
            pred_delta_y=delta_y,
            # pred_delta_x=1,
            # pred_delta_y=-1,
            pred_delta_depth=delta_depth,
            pred_delta_ori=delta_ori,
            depth_norm_params=depth_norm_params,
            ori_norm_params=ori_norm_params,
            pred_contact=contact,
            task_name=task_name,
            visualizer=visualizer,
            use_visualizer=use_visualizer,
            hand_mocap=hand_mocap,
            device=device,
            log_metric=True,
            vis_groundtruth=True
        )

        plt_path = 'vis_img.png'
        plt.figure(dpi=500)
        plt.imshow(vis_img)
        plt.savefig('vis_img.png')
        # plt.show()
        plt.close()
        print(f'Saved visualization image to {plt_path}.')

        if log_tb:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir='tb_vis_test')
            writer.add_scalar('loss', 3, 1)
            writer.add_image(f'vis_images', vis_img, 1, dataformats='HWC')
            print(f'Saved tensorboard visualization image to tb_vis_test.')

    if test_robot_vis:
        # test visualization
        frankmocap_path = '/home/junyao/LfHV/frankmocap'
        r3m_path = '/home/junyao/LfHV/r3m'
        sys.path.insert(1, frankmocap_path)
        os.chdir(frankmocap_path)
        from handmocap.hand_mocap_api import HandMocap

        if use_visualizer:
            print('Loading opengl visualizer.')
            from renderer.visualizer import Visualizer

            visualizer = Visualizer('opengl')
        else:
            visualizer = None
        print('Loading frank mocap hand module.')
        checkpoint_hand = osp.join(
            frankmocap_path,
            'extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
        )
        smpl_dir = osp.join(frankmocap_path, 'extra_data/smpl')
        hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device='cuda')
        os.chdir(r3m_path)

        current_hand_pose_path = '/home/junyao/Datasets/something_something_robot_demos/' \
                                 'push_right/mocap_output/1/mocap/frame2_prediction_result.pkl'
        future_hand_pose_path = '/home/junyao/Datasets/something_something_robot_demos/' \
                                'push_right/mocap_output/1/mocap/frame3_prediction_result.pkl'
        hand = 'left_hand'

        print('Preprocessing hand data from pkl.')
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        future_image_path = future_hand_info['image_path']
        if future_image_path[:8] == '/scratch':
            future_image_path = '/home' + future_image_path[8:]
        future_image = cv2.imread(future_image_path)
        future_hand_pose = future_hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        future_hand_bbox = normalize_bbox(
            future_hand_info['hand_bbox_list'][0][hand],
            (future_image.shape[1], future_image.shape[0])
        )
        future_camera = future_hand_info['pred_output_list'][0][hand]['pred_camera']
        future_joint_depth = future_hand_info['pred_output_list'][0][hand]['pred_joints_img'][:, 2]
        future_hand_shape = future_hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)

        task_names = [
            'move_away',
            'move_towards',
            'move_down',
            'move_up',
            'pull_left',
            'pull_right',
            'push_left',
            'push_right',
            'push_slightly',
            'push_left_right'
        ]
        task = np.zeros(len(task_names))
        task[-1] = 1

        print('Begin visualization.')
        vis_img = generate_single_visualization(
            current_hand_pose_path=current_hand_pose_path,
            future_hand_pose_path=future_hand_pose_path if has_future_label else None,
            future_cam=future_camera,
            hand=hand,
            pred_hand_bbox=torch.from_numpy(future_hand_bbox).to(device),
            pred_hand_pose=torch.from_numpy(future_hand_pose).to(device),
            pred_hand_shape=torch.from_numpy(future_hand_shape).to(device),
            task_names=task_names,
            task=task,
            visualizer=visualizer,
            hand_mocap=hand_mocap,
            use_visualizer=use_visualizer,
            run_on_cv_server=True,
            robot_demos=True,
            log_depth=True,
            current_depth=1,
            future_depth=future_joint_depth.mean(),
            pred_depth=1
        )

        plt_path = 'vis_img.png'
        plt.figure()
        plt.imshow(vis_img)
        plt.savefig('vis_img.png')
        # plt.show()
        plt.close()
        print(f'Saved visualization image to {plt_path}.')

        if log_tb:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir='tb_vis_test')
            writer.add_scalar('loss', 3, 1)
            writer.add_image(f'vis_images', vis_img, 1, dataformats='HWC')
            print(f'Saved tensorboard visualization image to tb_vis_test.')
    pass