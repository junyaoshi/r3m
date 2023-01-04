import os.path as osp
import argparse
import sys
import time
from copy import deepcopy
import os
from pprint import pprint

import numpy as np
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.save_r3m_for_ss_frame import setup_r3m, save_r3m
from utils.bc_utils import (
    load_eval_bc_model_and_args, generate_single_visualization
)
from utils.data_utils import CLUSTER_TASKS, cluster_task_to_cv_task, determine_which_hand, process_mocap_pred, \
    estimate_depth


def parse_eval_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Training BC network.')

    # data
    parser.add_argument('--collect_paired_data', action='store_true', default=False,
                        help='if true, pipeline uses paired data collection mode')
    parser.add_argument('--save_bbox_json', action='store_true', default=False,
                        help='if true, extracted bounding box will be saved to json')
    parser.add_argument('--save_mocap_pkl', action='store_true', default=False,
                        help='if true, extracted hand pose will be saved to pkl')
    parser.add_argument('--save_mocap_vis', action='store_true', default=False,
                        help='if true, hand pose visualization will be saved to jpg')
    parser.add_argument('--save_r3m', action='store_true', default=False,
                        help='if true, r3m embedding will be saved to pkl')
    parser.add_argument("--renderer_type", type=str, default="opengl",
                        choices=['opengl', 'opendr', 'pytorch3d', 'None'],
                        help="type of renderer to use")
    parser.add_argument('--image_path', type=str, default='/home/junyao/test/frames/frame3.jpg',
                        help='location of test frame')
    parser.add_argument('--robot_image_path', type=str, default='/home/junyao/test/robot_frames/frame3.jpg',
                        help='location of test robot frame')
    parser.add_argument('--json_path', type=str, default='/home/junyao/test/bbs_json/frame3.json',
                        help='location of test json')
    parser.add_argument('--mocap_dir', type=str, default='/home/junyao/test',
                        help='location of test mocap dir')
    parser.add_argument('--r3m_embedding_path', type=str, default='/home/junyao/test/r3m/frame3_r3m.pkl',
                        help='location for saving r3m embedding')
    parser.add_argument('--iou_dir', type=str, default='/home/junyao/test',
                        help='location of iou filtering dir, set to empty string to skip saving IoU json')
    parser.add_argument('--iou_thresh', type=float, default=0.6,
                        help='IoU threshold for filtering frames. Set to 0 to skip filtering.')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='if true, will enable verbose printing')

    # paths
    parser.add_argument('--frankmocap_path', type=str, default='/home/junyao/LfHV/frankmocap',
                        help='location of FrankMocap')
    parser.add_argument('--hod_path', type=str, default='/home/junyao/LfHV/hand_object_detector',
                        help='location of hand_object_detector')
    parser.add_argument('--r3m_path', type=str, default='/home/junyao/LfHV/r3m',
                        help='location of R3M')
    parser.add_argument('--root', type=str, default='/home/junyao/LfHV/r3m/eval_checkpoints',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='debug',
                        help='id used for storing intermediate results')
    parser.add_argument('--checkpoint', type=str,
                        default="/home/junyao/LfHV/r3m/checkpoints/transfer/"
                                "cluster_model=transfer_blocks=4_net=residual_time=20_"
                                "lr=0.0004_lambdas=[1,5,0]_batch=64_date=06261730/checkpoint_0150.pt",
                        help='location of the checkpoint to load for evaluation')
    parser.add_argument('--conda_root', dest='conda_root', type=str, default='/home/junyao/anaconda3',
                        help='root directory of conda')

    # eval
    parser.add_argument('--log_depth', action='store_true',
                        help='if true, log depth evaluation in visualization')
    parser.add_argument('--depth_descriptor', type=str, default='normalized_bbox_size',
                        help='which descriptor to use for hand depth estimation',
                        choices=['wrist_img_z', 'bbox_size', 'scaling_factor', 'normalized_bbox_size'])
    parser.add_argument('--log_metric', action='store_true',
                        help='if true, log metric evaluation in visualization')

    eval_args = parser.parse_args()
    return eval_args


def main(eval_args):
    if eval_args.collect_paired_data:
        assert eval_args.robot_image_path
    eval_args.save = osp.join(eval_args.root, eval_args.save)
    pprint(f'eval args: \n{eval_args}')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = device == "cuda"
    print(f'Device: {device}.')

    # set up hod
    if eval_args.hod_path not in sys.path:
        sys.path.insert(1, eval_args.hod_path)
    from demo_ss_frame import set_up_bbox_extractor, extract_bbox_from_frame
    fasterRCNN = set_up_bbox_extractor(
        cfg_file=osp.join(eval_args.hod_path, 'cfgs/res101.yml'),
        cuda=cuda,
        load_dir=osp.join(eval_args.hod_path, 'models')
    )

    # set up frankmokcap
    if eval_args.frankmocap_path not in sys.path:
        sys.path.insert(1, eval_args.frankmocap_path)
    os.chdir(eval_args.frankmocap_path)
    from demo.demo_handmocap_frame import setup_handmocap, setup_visualizer, run_frame_hand_mocap
    from ss_utils.filter_utils import filter_singe_data_by_IoU_threshold
    hand_mocap = setup_handmocap(
        frankmocap_dir=eval_args.frankmocap_path,
        checkpoint_hand_relative=osp.join(
            eval_args.frankmocap_path, 'extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
        ),
        smpl_dir_relative=osp.join(
            eval_args.frankmocap_path, 'extra_data/smpl/'
        )
    )
    visualizer = setup_visualizer(eval_args.renderer_type)
    os.chdir(eval_args.r3m_path)

    # set up r3m
    r3m, transforms = setup_r3m(device)
    eval_args.save = osp.join(eval_args.root, eval_args.save)

    # set up BC
    writer = SummaryWriter(log_dir=eval_args.save, flush_secs=60)
    model, args = load_eval_bc_model_and_args(eval_args, device)
    eval_task_names = CLUSTER_TASKS
    print(f'Evaluation task names: {eval_task_names}')
    metric_stats = {t: {'hand_total': 0, 'hand_success': 0} for t in eval_task_names}
    if eval_args.collect_paired_data:
        for t in eval_task_names:
            metric_stats[t]['robot_total'] = 0
            metric_stats[t]['robot_success'] = 0

    start = time.time()
    print()
    # extract bbox
    bbox_dict, image_bgr, bbox_success = extract_bbox_from_frame(
        fasterRCNN,
        eval_args.image_path,
        json_path=eval_args.json_path if eval_args.save_bbox_json else None,
        cuda=cuda,
        verbose=eval_args.verbose
    )
    robot_image_bgr = None
    if eval_args.collect_paired_data:
        robot_image_bgr = cv2.imread(eval_args.robot_image_path)

    if bbox_success:
        print('Successfully extracted bounding box. Proceeding with next step.')
    else:
        print('Failed to extract bounding box. Skipping this frame.')
        return

    # Hand pose detection
    mocap_pred, mocap_success = run_frame_hand_mocap(
        input_path=None,
        out_dir=eval_args.mocap_dir,
        bbox_dict=bbox_dict,
        img_original_bgr=image_bgr,
        hand_mocap=hand_mocap,
        visualizer=visualizer,
        save_pred_pkl=eval_args.save_mocap_pkl,
        save_pred_vis=eval_args.save_mocap_vis,
        verbose=eval_args.verbose
    )

    if mocap_success:
        print('Successfully extracted hand pose. Proceeding with next step.')
    else:
        print('Failed to extract hand pose. Skipping this frame.')
        return

    # filter data by IoU threshold
    if eval_args.iou_dir:
        iou_json_path = osp.join(eval_args.iou_dir, f'IoU_{eval_args.iou_thresh}.json')
        print(f'Saving IoU json to: {iou_json_path}.')
    else:
        iou_json_path = None
    if eval_args.iou_thresh == 0:
        print('IoU threshold is set to 0. Skip filtering data by IoU...')
    else:
        IoU, filter_success = filter_singe_data_by_IoU_threshold(
            mocap_path=None,
            IoU_thresh=eval_args.iou_thresh,
            json_path=iou_json_path,
            vid_num=0,
            frame_num=0,
            mocap_pred=mocap_pred,
            verbose=eval_args.verbose
        )

        if filter_success:
            print(f'Extracted hand pose passed IoU threshold filtering with IoU: {IoU:.3f}. '
                  'Proceeding with next step.')
        else:
            print(f'Extracted hand pose did not pass IoU threshold filtering with IoU: {IoU:.3f}. '
                  'Skipping this frame.')
            return

    # get r3m embedding
    hand_r3m_embedding = save_r3m(
        r3m=r3m,
        transforms=transforms,
        frame_path=None,
        r3m_embedding_path=None,
        device=device,
        frame_bgr=image_bgr
    )
    robot_r3m_embedding = None
    if eval_args.collect_paired_data:
        robot_r3m_embedding = save_r3m(
            r3m=r3m,
            transforms=transforms,
            frame_path=None,
            r3m_embedding_path=None,
            device=device,
            frame_bgr=robot_image_bgr
        )
    r3m_embedding = robot_r3m_embedding if eval_args.collect_paired_data else hand_r3m_embedding

    # process data for model evaluation
    t0 = time.time()
    hand = determine_which_hand(hand_info=mocap_pred)
    (
        current_hand_bbox,
        current_camera,
        current_img_shape,
        current_hand_depth_estimate,
        current_wrist_depth_real,
        current_hand_pose,
        current_hand_shape
    ) = process_mocap_pred(
        mocap_pred_path=None,
        hand=hand,
        mocap_pred=mocap_pred,
        depth_descriptor=eval_args.depth_descriptor
    )
    future_camera = deepcopy(current_camera)

    # visualize model output conditioning on different task inputs
    vis_imgs = []
    for i, task_name in enumerate(eval_task_names):
        task = torch.zeros(1, len(eval_task_names))
        task[0, i] = 1
        task_input = task.clone()
        if args.run_on_cv_server and not eval_args.run_on_cv_server:
            task_input = cluster_task_to_cv_task(task.squeeze()).unsqueeze(0)

        input = torch.cat((
            r3m_embedding.cpu(),
            task_input,
            torch.from_numpy(current_hand_bbox).unsqueeze(0),
            torch.from_numpy(current_hand_pose).unsqueeze(0),
            torch.from_numpy(current_hand_shape).unsqueeze(0),
            torch.from_numpy(current_camera).unsqueeze(0),
            torch.from_numpy(future_camera).unsqueeze(0)
        ), dim=1).to(device).float()

        with torch.no_grad():
            output = model(input)
            pred_hand_bbox = torch.sigmoid(
                output[:, :args.hand_bbox_dim]
            )  # force positive values for bbox output
            pred_hand_pose = output[:, args.hand_bbox_dim:(args.hand_bbox_dim + args.hand_pose_dim)]
            pred_hand_shape = output[:, (args.hand_bbox_dim + args.hand_pose_dim):]

        pred_hand_depth_estimate = None
        if eval_args.log_depth:
            pred_hand_depth_estimate = estimate_depth(
                eval_args=eval_args,
                args=args,
                hand_bbox=pred_hand_bbox[0].cpu().detach().numpy()
            )

        vis_img, passed_metric = generate_single_visualization(
            current_hand_pose_path=None,
            current_hand_info=mocap_pred,
            future_hand_pose_path=None,
            future_cam=future_camera,
            hand=hand,
            pred_hand_bbox=pred_hand_bbox[0]
            if args.predict_hand_bbox else torch.from_numpy(current_hand_bbox).to(device),
            pred_hand_pose=pred_hand_pose[0]
            if args.predict_hand_pose else torch.from_numpy(current_hand_pose).to(device),
            pred_hand_shape=pred_hand_shape[0]
            if args.predict_hand_shape else torch.from_numpy(current_hand_shape).to(device),
            task_names=eval_task_names,
            task=task[0],
            visualizer=visualizer,
            hand_mocap=hand_mocap,
            use_visualizer=visualizer is not None,
            run_on_cv_server=False,
            robot_demos=eval_args.collect_paired_data,
            current_img=robot_image_bgr if eval_args.collect_paired_data else image_bgr,
            log_depth=eval_args.log_depth,
            log_metric=eval_args.log_metric,
            current_depth=current_hand_depth_estimate,
            pred_depth=pred_hand_depth_estimate
        )
        vis_imgs.append(vis_img)
        if eval_args.log_metric:
            metric_stats[task_name]['robot_total' if eval_args.collect_paired_data else 'hand_total'] += 1
            if passed_metric:
                metric_stats[task_name]['robot_success' if eval_args.collect_paired_data else 'hand_success'] += 1

        if eval_args.collect_paired_data:
            hand_input = torch.cat((
                hand_r3m_embedding.cpu(),
                task_input,
                torch.from_numpy(current_hand_bbox).unsqueeze(0),
                torch.from_numpy(current_hand_pose).unsqueeze(0),
                torch.from_numpy(current_hand_shape).unsqueeze(0),
                torch.from_numpy(current_camera).unsqueeze(0),
                torch.from_numpy(future_camera).unsqueeze(0)
            ), dim=1).to(device).float()

            with torch.no_grad():
                output = model(hand_input)
                pred_hand_bbox = torch.sigmoid(
                    output[:, :args.hand_bbox_dim]
                )  # force positive values for bbox output
                pred_hand_pose = output[:, args.hand_bbox_dim:(args.hand_bbox_dim + args.hand_pose_dim)]
                pred_hand_shape = output[:, (args.hand_bbox_dim + args.hand_pose_dim):]

            hand_vis_img, hand_passed_metric = generate_single_visualization(
                current_hand_pose_path=None,
                current_hand_info=mocap_pred,
                future_hand_pose_path=None,
                future_cam=future_camera,
                hand=hand,
                pred_hand_bbox=pred_hand_bbox[0]
                if args.predict_hand_bbox else torch.from_numpy(current_hand_bbox).to(device),
                pred_hand_pose=pred_hand_pose[0]
                if args.predict_hand_pose else torch.from_numpy(current_hand_pose).to(device),
                pred_hand_shape=pred_hand_shape[0]
                if args.predict_hand_shape else torch.from_numpy(current_hand_shape).to(device),
                task_names=eval_task_names,
                task=task[0],
                visualizer=visualizer,
                hand_mocap=hand_mocap,
                use_visualizer=visualizer is not None,
                run_on_cv_server=False,
                robot_demos=False,
                current_img=image_bgr,
                log_depth=eval_args.log_depth,
                log_metric=eval_args.log_metric,
                current_depth=current_hand_depth_estimate,
                pred_depth=pred_hand_depth_estimate
            )
            vis_imgs.append(hand_vis_img)
            if eval_args.log_metric:
                metric_stats[task_name]['hand_total'] += 1
                if passed_metric:
                    metric_stats[task_name]['hand_success'] += 1

    final_vis_img = np.hstack(vis_imgs)
    writer.add_image(f'vis_tasks_vid0/frame0', final_vis_img, dataformats='HWC')
    if eval_args.log_metric:
        all_tasks_hand_total, all_tasks_hand_success = 0, 0
        all_tasks_robot_total, all_tasks_robot_success = 0, 0
        for t_name, t_metric in metric_stats.items():
            task_hand_total = t_metric['hand_total']
            task_hand_success = t_metric['hand_success']
            writer.add_scalar(f'{t_name}_metric_stats/hand_total', task_hand_total)
            writer.add_scalar(f'{t_name}_metric_stats/hand_success', task_hand_success)
            all_tasks_hand_total += task_hand_total
            all_tasks_hand_success += task_hand_success
            if eval_args.collect_paired_data:
                task_robot_total = t_metric['robot_total']
                task_robot_success = t_metric['robot_success']
                writer.add_scalar(f'{t_name}_metric_stats/robot_total', task_robot_total)
                writer.add_scalar(f'{t_name}_metric_stats/robot_success', task_robot_success)
                all_tasks_robot_total += task_robot_total
                all_tasks_robot_success += task_robot_success
        writer.add_scalar('overall_metric_stats/hand_total', all_tasks_robot_total)
        writer.add_scalar('overall_metric_stats/hand_success', all_tasks_robot_success)
        if eval_args.collect_paired_data:
            writer.add_scalar('overall_metric_stats/robot_total', all_tasks_robot_total)
            writer.add_scalar('overall_metric_stats/robot_success', all_tasks_robot_success)

    time.sleep(0.5)
    t1 = time.time()
    print(f'Evaluated data for model task conditioning in {t1 - t0:.3f} seconds.')
    end = time.time()
    print(f'\nDone with online model evaluation. Time elapsed: {end - start:.3f} seconds.')


if __name__ == '__main__':
    eval_args = parse_eval_args()
    main(eval_args)
