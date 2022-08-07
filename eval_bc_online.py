import os
import os.path as osp
import argparse
import time
import sys
from pprint import pprint
from copy import deepcopy

import pyrealsense2 as rs
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from save_r3m_for_ss_frame import setup_r3m, save_r3m
from bc_utils import (
    determine_which_hand, process_mocap_pred, cluster_task_to_cv_task,
    load_eval_bc_model_and_args, generate_single_visualization,
    estimate_depth, CLUSTER_TASKS
)


def parse_eval_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Training BC network.')

    # data
    parser.add_argument('--collect_paired_data', action='store_true', default=False,
                        help='if true, pipeline uses paired data collection mode')
    parser.add_argument('--data_save_dir', type=str, required=True,
                        help='location where real-time data will be saved')
    parser.add_argument('--video_index', type=int, default=1,
                        help='starting video index; press n to increment by 1 during collection')
    parser.add_argument('--hand_frame_index', type=int, default=0,
                        help='starting hand frame index; will increment by 1 each time data is recorded')
    parser.add_argument('--robot_frame_index', type=int, default=0,
                        help='starting robot frame index; will increment by 1 each time data is recorded')
    parser.add_argument('--save_bbox_json', action='store_true', default=False,
                        help='if true, extracted bounding box will be saved to json')
    parser.add_argument('--save_mocap_pkl', action='store_true', default=False,
                        help='if true, extracted hand pose will be saved to pkl')
    parser.add_argument('--save_mocap_vis', action='store_true', default=False,
                        help='if true, hand pose visualization will be saved to jpg')
    parser.add_argument('--save_iou', action='store_true', default=False,
                        help='if true, iou filtering results will be saved to json')
    parser.add_argument('--save_r3m', action='store_true', default=False,
                        help='if true, r3m embedding will be saved to pkl')
    parser.add_argument("--renderer_type", type=str, default="opengl",
                        choices=['opengl', 'opendr', 'None'],
                        help="type of renderer to use")
    parser.add_argument('--iou_thresh', type=float, default=0.6,
                        help='IoU threshold for filtering frames. Set to 0 to skip filtering.')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='if true, will enable verbose printing')

    # Intel RealSense
    parser.add_argument('--device_id', type=str, default='913522070437',
                        help='location of the checkpoint to load for evaluation')

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



def setup_realsense_pipeline(args):
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_device(args.device_id)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    print(f'RealSense camera stream started on device: {args.device_id}.')

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f'Depth Scale is: {depth_scale:.4f}')

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align


def main(eval_args):
    eval_args.save = osp.join(eval_args.root, eval_args.save)
    pprint(f'eval args: \n{eval_args}')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = device == 'cuda'
    print(f'Device: {device}.')
    pipeline, align = setup_realsense_pipeline(eval_args)

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

    frames_save_dir = osp.join(eval_args.data_save_dir, 'frames')
    depths_save_dir = osp.join(eval_args.data_save_dir, 'depths')
    robot_frames_save_dir = osp.join(eval_args.data_save_dir, 'robot_frames')
    robot_depths_save_dir = osp.join(eval_args.data_save_dir, 'robot_depths')
    r3m_save_dir = osp.join(eval_args.data_save_dir, 'r3m')
    print(f'Begin capturing frames of video {eval_args.video_index}.'
          f'\nStarting frame index: {eval_args.hand_frame_index} for hand; {eval_args.robot_frame_index} for robot.'
          f'\nSaving the frames under {eval_args.data_save_dir}.')

    video_index = eval_args.video_index
    hand_frame_index = eval_args.hand_frame_index
    robot_frame_index = eval_args.robot_frame_index
    frames_buffer = []  # format: (video_index, frame_index)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()  # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow('Online Data Stream', cv2.WINDOW_NORMAL)
            cv2.imshow('Online Data Stream', images)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                if eval_args.collect_paired_data:
                    assert hand_frame_index == robot_frame_index, \
                        'Collecting paired data but number of hand and robot frames are not equal!'
                print(f'Video {video_index} frame capturing stopped. Frames elapsed: {hand_frame_index}.')
                break
            elif key == ord('p'):  # hand frame capturing
                video_frames_save_dir = osp.join(frames_save_dir, str(video_index))
                video_depths_save_dir = osp.join(depths_save_dir, str(video_index))
                os.makedirs(video_frames_save_dir, exist_ok=True)
                os.makedirs(video_depths_save_dir, exist_ok=True)
                print(f'Saving hand frame {hand_frame_index} under {eval_args.data_save_dir} video {video_index}.')
                cv2.imwrite(osp.join(video_frames_save_dir, f'frame{hand_frame_index}.jpg'), color_image)
                np.save(osp.join(video_depths_save_dir, f'frame{hand_frame_index}.npy'), depth_image)
                frames_buffer.append((video_index, hand_frame_index))
                hand_frame_index += 1
            elif key == ord('l'):  # robot frame capturing
                video_robot_frames_save_dir = osp.join(robot_frames_save_dir, str(video_index))
                video_robot_depths_save_dir = osp.join(robot_depths_save_dir, str(video_index))
                os.makedirs(video_robot_frames_save_dir, exist_ok=True)
                os.makedirs(video_robot_depths_save_dir, exist_ok=True)
                print(f'Saving robot frame {robot_frame_index} under {eval_args.data_save_dir} video {video_index}.')
                cv2.imwrite(osp.join(video_robot_frames_save_dir, f'frame{robot_frame_index}.jpg'), color_image)
                np.save(osp.join(video_robot_depths_save_dir, f'frame{robot_frame_index}.npy'), depth_image)
                robot_frame_index += 1
            elif key == ord('n'):  # stop current video, start next video
                if eval_args.collect_paired_data and hand_frame_index != robot_frame_index:
                    print('Collecting paired data but number of hand and robot frames are not equal!')
                    print('Continue with frame capturing data collection.')
                    continue
                print(f'Video {video_index} frame capturing stopped. Frames elapsed: {hand_frame_index}.')
                video_index += 1
                hand_frame_index = 0
                robot_frame_index = 0
                print(f'\nBegin capturing frames of video {video_index}. '
                      f'\nSaving the frames under {eval_args.data_save_dir}.')
            elif key == ord('m'):  # evaluate model
                if eval_args.collect_paired_data and hand_frame_index != robot_frame_index:
                    print('Collecting paired data but number of hand and robot frames are not equal!')
                    print('Continue with frame capturing data collection.')
                    continue
                if not frames_buffer:
                    print('No frames in the buffer. Continue with frame capturing data collection.')
                    continue

                evaluated_frames = 0
                for v_idx, f_idx in tqdm(
                        frames_buffer, desc=f'Evaluating mode on {len(frames_buffer)} frames in the buffer...'
                ):
                    start = time.time()
                    print(f'\nBegin evaluation of video {v_idx} frame {f_idx}.')

                    # hand bbox detection
                    image_path = osp.join(frames_save_dir, str(v_idx), f'frame{f_idx}.jpg')
                    json_path = osp.join(eval_args.data_save_dir, 'bbs_json', str(v_idx), f'frame{f_idx}.json')
                    bbox_dict, image_bgr, bbox_success = extract_bbox_from_frame(
                        fasterRCNN,
                        image_path,
                        json_path=json_path if eval_args.save_bbox_json else None,
                        cuda=cuda,
                        verbose=eval_args.verbose
                    )

                    if bbox_success:
                        print('Successfully extracted bounding box. Proceeding with next step.')
                    else:
                        print('Failed to extract bounding box. Skipping this frame.')
                        continue

                    # Hand pose detection
                    mocap_dir = osp.join(eval_args.data_save_dir, 'mocap_output', str(v_idx))
                    mocap_pred, mocap_success = run_frame_hand_mocap(
                        input_path=None,
                        out_dir=mocap_dir,
                        bbox_dict=bbox_dict,
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
                        continue

                    # filter data by IoU threshold
                    if eval_args.save_iou:
                        iou_json_path = osp.join(eval_args.data_save_dir, f'IoU_{eval_args.iou_thresh}.json')
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
                            vid_num=v_idx,
                            frame_num=f_idx,
                            mocap_pred=mocap_pred,
                            verbose=eval_args.verbose
                        )

                        if filter_success:
                            print(f'Extracted hand pose passed IoU threshold filtering with IoU: {IoU:.3f}. '
                                  'Proceeding with next step.')
                        else:
                            print(f'Extracted hand pose did not pass IoU threshold filtering with IoU: {IoU:.3f}. '
                                  'Skipping this frame.')
                            continue

                    # get r3m embedding
                    r3m_embedding_path = osp.join(r3m_save_dir, str(v_idx), f'frame{f_idx}_r3m.pkl')
                    hand_r3m_embedding = save_r3m(
                        r3m=r3m,
                        transforms=transforms,
                        frame_path=None,
                        r3m_embedding_path=r3m_embedding_path if eval_args.save_r3m else None,
                        device=device,
                        frame_bgr=image_bgr
                    )
                    r3m_embedding = hand_r3m_embedding

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

                        vis_img = generate_single_visualization(
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
                            current_img=image_bgr,
                            log_depth=eval_args.log_depth,
                            log_metric=eval_args.log_metric,
                            current_depth=current_hand_depth_estimate,
                            pred_depth=pred_hand_depth_estimate
                        )
                        vis_imgs.append(vis_img)
                    final_vis_img = np.hstack(vis_imgs)
                    writer.add_image(f'vis_tasks_vid{v_idx}/frame{f_idx}', final_vis_img, dataformats='HWC')
                    t1 = time.time()
                    print(f'Evaluated video {v_idx} frame {f_idx} for model task conditioning '
                          f'in {t1 - t0:.3f} seconds.')

                    end = time.time()
                    print(f'\nDone with online model evaluation for video {v_idx} frame {f_idx}. '
                          f'Time elapsed: {end - start:.3f} seconds.')
                    evaluated_frames += 1

                print(f'Done with online model evaluation for '
                      f'{evaluated_frames}/{len(frames_buffer)} frames in the buffer.')
                frames_buffer = []

    finally:
        pipeline.stop()


if __name__ == '__main__':
    eval_args = parse_eval_args()
    main(eval_args)
