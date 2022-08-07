import json
import math
import os
import time
from os.path import join
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import pickle
import cv2
import multiprocessing as mp

import torch
from torch.utils.data import Dataset

from bc_utils import determine_which_hand, normalize_bbox, process_mocap_pred, CLUSTER_TASKS

class SomethingSomethingR3M(Dataset):
    def __init__(self, task_names, data_home_dir,
                 iou_thresh=0.7, time_interval=5, train=True, debug=False,
                 run_on_cv_server=False, num_cpus=4, depth_descriptor='scaling_factor'):
        """
        Set num_cpus=1 to disable multiprocessing
        """
        print(f'Initializing dataset for tasks: \n{task_names}. \nSplit: {"train" if train else "valid"}')
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.data_home_dir = data_home_dir
        self.iou_thresh = iou_thresh
        self.time_interval = time_interval
        self.train = train
        self.debug = debug
        self.run_on_cv_server = run_on_cv_server
        self.depth_descriptor = depth_descriptor
        self.task_dict = {task_name: int(i) for i, task_name in enumerate(task_names)}

        (
            self.r3m_paths, self.tasks, self.hands,
            self.current_hand_pose_paths, self.future_hand_pose_paths
        ) = self.fetch_data(num_cpus=num_cpus)

        print(f'Dataset has {len(self)} data.')

    def fetch_data(self, num_cpus):
        if self.debug:
            self.task_names = [self.task_names[0]]
            (
                r3m_paths, tasks, hands,
                current_hand_pose_paths, future_hand_pose_paths
            ) = self.single_process_fetch_data(self.task_names)
        elif num_cpus == 1 or num_cpus == 0:
            (
                r3m_paths, tasks, hands,
                current_hand_pose_paths, future_hand_pose_paths
            ) = self.single_process_fetch_data(self.task_names)
        else:
            splits, n_tasks_left, n_cpus_left = [0], self.num_tasks, num_cpus
            while n_tasks_left:
                tasks_assigned = math.ceil(n_tasks_left / n_cpus_left)
                n_tasks_left -= tasks_assigned
                n_cpus_left -= 1
                last_task = splits[-1]
                splits.append(last_task + tasks_assigned)
            args_list = [self.task_names[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]
            print(f'Number of processes: {num_cpus}')
            print(f'Splits for multiprocessing: \n{args_list}')

            # multiprocessing (num_cpus processes)
            pool = mp.Pool(num_cpus)
            with pool as p:
                r = list(tqdm(p.imap(self.single_process_fetch_data, args_list), total=num_cpus))
            r3m_paths, tasks, hands = [], [], []
            current_hand_pose_paths, future_hand_pose_paths = [], []
            for process_r in r:
                (
                    p_r3m_paths, p_tasks, p_hands,
                    p_current_hand_pose_paths, p_future_hand_pose_paths
                ) = process_r
                r3m_paths += p_r3m_paths
                tasks += p_tasks
                hands += p_hands
                current_hand_pose_paths += p_current_hand_pose_paths
                future_hand_pose_paths += p_future_hand_pose_paths

        return r3m_paths, tasks, hands, current_hand_pose_paths, future_hand_pose_paths

    def single_process_fetch_data(self, task_names):
        r3m_paths, tasks, hands = [], [], []
        current_hand_pose_paths, future_hand_pose_paths = [], []
        for task_name in task_names:
            print(f'Processing task: {task_name}.')
            split = 'train' if self.train else 'valid'
            split_dir = join(self.data_home_dir, task_name, split)
            r3m_dir = join(split_dir, 'r3m')
            iou_json_path = join(split_dir, f'IoU_{self.iou_thresh}.json')
            with open(iou_json_path, 'r') as f:
                json_dict = json.load(f)
                f.close()

            for vid_num in tqdm(json_dict, desc='Going through videos...'):
                r3m_vid_dir = join(r3m_dir, vid_num)
                mocap_vid_dir = join(split_dir, 'mocap_output', vid_num, 'mocap')
                for current_frame_num in json_dict[vid_num]:
                    # check if future frame exists
                    future_frame_num = str(int(current_frame_num) + self.time_interval)
                    if future_frame_num not in json_dict[vid_num]:
                        continue

                    # check if current and future frames have the same hand
                    current_hand_pose_path = join(mocap_vid_dir, f'frame{current_frame_num}_prediction_result.pkl')
                    future_hand_pose_path = join(mocap_vid_dir, f'frame{future_frame_num}_prediction_result.pkl')
                    with open(current_hand_pose_path, 'rb') as f:
                        current_hand_info = pickle.load(f)
                    current_hand = determine_which_hand(current_hand_info)
                    with open(future_hand_pose_path, 'rb') as f:
                        future_hand_info = pickle.load(f)
                    future_hand = determine_which_hand(future_hand_info)
                    if current_hand != future_hand:
                        continue
                    task = np.zeros(self.num_tasks)
                    task[self.task_dict[task_name]] = 1

                    hands.append(current_hand)
                    r3m_paths.append(join(r3m_vid_dir, f'frame{current_frame_num}_r3m.pkl'))
                    tasks.append(task)
                    current_hand_pose_paths.append(current_hand_pose_path)
                    future_hand_pose_paths.append(future_hand_pose_path)

                if self.debug and len(r3m_paths) > 600:
                    break

        return r3m_paths, tasks, hands, current_hand_pose_paths, future_hand_pose_paths


    def __len__(self):
        return len(self.r3m_paths)

    def _extract_hand_info(self, hand_pose_path, hand):
        return process_mocap_pred(
            mocap_pred_path=hand_pose_path,
            hand=hand,
            mocap_pred=None,
            depth_descriptor=self.depth_descriptor
        )

    def __getitem__(self, idx):
        hand_r3m_path = self.r3m_paths[idx]
        task = self.tasks[idx]
        current_hand_pose_path = self.current_hand_pose_paths[idx]
        future_hand_pose_path = self.future_hand_pose_paths[idx]
        hand = self.hands[idx]

        with open(hand_r3m_path, 'rb') as f:
            hand_r3m_embedding = pickle.load(f)
        robot_r3m_embedding = torch.zeros_like(hand_r3m_embedding)

        # with open(current_hand_pose_path, 'rb') as f:
        #     current_hand_info = pickle.load(f)
        # current_hand_pose = current_hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        # current_hand_bbox = current_hand_info['hand_bbox_list'][0][hand]
        # current_camera = current_hand_info['pred_output_list'][0][hand]['pred_camera']
        # current_img_shape = current_hand_info['image_shape']
        # current_hand_bbox = normalize_bbox(current_hand_bbox, (current_img_shape[1], current_img_shape[0]))
        # current_joint_depth = current_hand_info['pred_output_list'][0][hand]['pred_joints_img'][:, 2]
        # current_hand_shape = current_hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)
        #
        # with open(future_hand_pose_path, 'rb') as f:
        #     future_hand_info = pickle.load(f)
        # future_hand_pose = future_hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        # future_hand_bbox = future_hand_info['hand_bbox_list'][0][hand]
        # future_camera = future_hand_info['pred_output_list'][0][hand]['pred_camera']
        # future_img_shape = future_hand_info['image_shape']
        # future_hand_bbox = normalize_bbox(future_hand_bbox, (future_img_shape[1], future_img_shape[0]))
        # future_joint_depth = future_hand_info['pred_output_list'][0][hand]['pred_joints_img'][:, 2]
        # future_hand_shape = future_hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)
        #
        # current_wrist_depth_real, future_wrist_depth_real = 0, 0

        (
            current_hand_bbox,
            current_camera,
            current_img_shape,
            current_hand_depth_estimate,
            current_wrist_depth_real,
            current_hand_pose,
            current_hand_shape
        ) = self._extract_hand_info(
            current_hand_pose_path,
            hand
        )

        (
            future_hand_bbox,
            future_camera,
            future_img_shape,
            future_hand_depth_estimate,
            future_wrist_depth_real,
            future_hand_pose,
            future_hand_shape
        ) = self._extract_hand_info(
            future_hand_pose_path,
            hand
        )

        return (
            hand_r3m_embedding.squeeze().to(torch.device('cpu')),
            robot_r3m_embedding.squeeze().to(torch.device('cpu')),
            task, hand,
            current_hand_bbox, future_hand_bbox,
            current_camera, future_camera,
            current_img_shape, future_img_shape,
            current_hand_depth_estimate, future_hand_depth_estimate,
            current_wrist_depth_real, future_wrist_depth_real,
            current_hand_pose, future_hand_pose,
            current_hand_shape, future_hand_shape,
            current_hand_pose_path, future_hand_pose_path
        )


class SomethingSomethingDemosR3M(Dataset):
    def __init__(
            self, task_names, data_home_dir,
            iou_thresh=0.7, time_interval=5,
            debug=False, run_on_cv_server=True,
            demo_type='hand', depth_descriptor='scaling_factor'
    ):
        print(f'Initializing {demo_type} demos dataset for tasks: \n{task_names}.')
        assert demo_type in ['hand', 'robot', 'same_hand'], f'Invalid demo type: {demo_type}'
        valid_depth_descriptors = ['wrist_img_z', 'bbox_size', 'scaling_factor', 'normalized_bbox_size']
        assert depth_descriptor in valid_depth_descriptors, f'Invalid depth descriptor: {depth_descriptor}.'

        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.data_home_dir = data_home_dir
        self.iou_thresh = iou_thresh
        self.time_interval = time_interval
        if demo_type in ['robot', 'same_hand']:
            assert self.time_interval == 1
        self.debug = debug
        self.run_on_cv_server = run_on_cv_server
        self.demo_type = demo_type
        self.depth_descriptor = depth_descriptor
        self.task_dict = {task_name: int(i) for i, task_name in enumerate(task_names)}
        (
            self.r3m_paths, self.tasks, self.hands,
            self.current_hand_pose_paths, self.future_hand_pose_paths,
            self.current_depth_paths, self.future_depth_paths
        ) = self.fetch_data()

        print(f'Dataset has {len(self)} data.')

    def __len__(self):
        return len(self.r3m_paths)

    def fetch_data(self):
        r3m_paths, tasks, hands = [], [], []
        current_hand_pose_paths, future_hand_pose_paths = [], []
        current_depth_paths, future_depth_paths = [], []
        dataset_task_names = os.listdir(self.data_home_dir)
        for task_name in dataset_task_names:
            print(f'Processing task: {task_name}.')
            split_dir = join(self.data_home_dir, task_name)
            r3m_dir = join(split_dir, 'r3m')
            depths_dir = join(split_dir, 'depths')
            iou_json_path = join(split_dir, f'IoU_{self.iou_thresh}.json')
            with open(iou_json_path, 'r') as f:
                json_dict = json.load(f)
                f.close()

            for vid_num in tqdm(json_dict, desc='Going through videos...'):
                r3m_vid_dir = join(r3m_dir, vid_num)
                mocap_vid_dir = join(split_dir, 'mocap_output', vid_num, 'mocap')
                depths_vid_dir = join(depths_dir, vid_num)
                frame_nums = sorted(json_dict[vid_num], key=int)
                for current_frame_num in frame_nums:
                    # check if future frame exists
                    future_frame_num = str(int(current_frame_num) + self.time_interval)
                    if future_frame_num not in json_dict[vid_num] and self.demo_type != 'same_hand':
                        continue

                    # check if current and future frames have the same hand
                    current_hand_pose_path = join(mocap_vid_dir, f'frame{current_frame_num}_prediction_result.pkl')
                    with open(current_hand_pose_path, 'rb') as f:
                        current_hand_info = pickle.load(f)
                    current_hand = determine_which_hand(current_hand_info)
                    current_depth_path = join(depths_vid_dir, f'frame{current_frame_num}.npy')
                    if self.demo_type != 'same_hand':
                        future_hand_pose_path = join(mocap_vid_dir, f'frame{future_frame_num}_prediction_result.pkl')
                        with open(future_hand_pose_path, 'rb') as f:
                            future_hand_info = pickle.load(f)
                        future_hand = determine_which_hand(future_hand_info)
                        if current_hand != future_hand:
                            continue
                        future_depth_path = join(depths_vid_dir, f'frame{future_frame_num}.npy')
                        future_hand_pose_paths.append(future_hand_pose_path)
                        future_depth_paths.append(future_depth_path)

                    task = np.zeros(self.num_tasks)
                    task[self.task_dict[task_name]] = 1
                    hands.append(current_hand)
                    r3m_paths.append(join(r3m_vid_dir, f'frame{current_frame_num}_r3m.pkl'))
                    tasks.append(task)
                    current_hand_pose_paths.append(current_hand_pose_path)
                    current_depth_paths.append(current_depth_path)

                if self.debug and len(r3m_paths) > 50:
                    break

        return (
            r3m_paths, tasks, hands,
            current_hand_pose_paths, future_hand_pose_paths,
            current_depth_paths, future_depth_paths
        )

    def _extract_hand_info(self, hand_pose_path, depth_path, hand):
        with open(hand_pose_path, 'rb') as f:
            hand_info = pickle.load(f)
        hand_pose = hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        unnormalized_hand_bbox = hand_info['hand_bbox_list'][0][hand]
        camera = hand_info['pred_output_list'][0][hand]['pred_camera']
        img_shape = hand_info['image_shape']
        hand_bbox = normalize_bbox(unnormalized_hand_bbox, (img_shape[1], img_shape[0]))
        hand_shape = hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)
        wrist_3d = hand_info['pred_output_list'][0][hand]['pred_joints_img'][0]
        wrist_coord = wrist_3d[:2]

        ymax, xmax = img_shape
        wrist_x_float, wrist_y_float = wrist_coord
        wrist_depth_real = -999  # value for invalid depth due to out of bound wrist joint
        if (0 <= wrist_x_float < xmax) and (0 <= wrist_y_float < ymax):
            wrist_coord = wrist_coord.round().astype(np.int16)
            wrist_x, wrist_y = wrist_coord
            if wrist_x != xmax and wrist_y != ymax:
                depth_real = np.load(depth_path)
                wrist_depth_real = depth_real[wrist_y, wrist_x].astype(np.int16)

        hand_depth_estimate = None
        if self.depth_descriptor == 'wrist_img_z':
            hand_depth_estimate = wrist_3d[2]
        elif self.depth_descriptor == 'bbox_size':
            *_, w, h = unnormalized_hand_bbox
            bbox_size = w * h
            hand_depth_estimate = 1. / bbox_size
        elif self.depth_descriptor == 'scaling_factor':
            cam_scale = camera[0]
            hand_boxScale_o2n = hand_info['pred_output_list'][0][hand]['bbox_scale_ratio']
            scaling_factor = cam_scale / hand_boxScale_o2n
            hand_depth_estimate = 1. / scaling_factor
        elif self.depth_descriptor == 'normalized_bbox_size':
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


    def __getitem__(self, idx):
        hand_r3m_path = self.r3m_paths[idx]
        task = self.tasks[idx]
        current_hand_pose_path = self.current_hand_pose_paths[idx]
        current_depth_path = self.current_depth_paths[idx]
        hand = self.hands[idx]

        with open(hand_r3m_path, 'rb') as f:
            hand_r3m_embedding = pickle.load(f)
        if self.demo_type == 'robot':
            robot_r3m_path = '/' + join(*hand_r3m_path.split('/')[:-3], 'robot_r3m', *hand_r3m_path.split('/')[-2:])
            with open(robot_r3m_path, 'rb') as f:
                robot_r3m_embedding = pickle.load(f)
        else:
            robot_r3m_embedding = torch.zeros_like(hand_r3m_embedding)

        (
            current_hand_bbox,
            current_camera,
            current_img_shape,
            current_hand_depth_estimate,
            current_wrist_depth_real,
            current_hand_pose,
            current_hand_shape
        ) = self._extract_hand_info(
            current_hand_pose_path,
            current_depth_path,
            hand
        )

        if self.demo_type == 'same_hand':
            future_hand_pose_path = ''
            future_hand_bbox = np.zeros_like(current_hand_bbox)
            future_camera = deepcopy(current_camera)
            future_img_shape = deepcopy(current_img_shape)
            future_hand_depth_estimate = -999
            future_wrist_depth_real = -999
            future_hand_pose = np.zeros_like(current_hand_pose)
            future_hand_shape = np.zeros_like(current_hand_shape)
        else:
            future_hand_pose_path = self.future_hand_pose_paths[idx]
            future_depth_path = self.future_depth_paths[idx]
            (
                future_hand_bbox,
                future_camera,
                future_img_shape,
                future_hand_depth_estimate,
                future_wrist_depth_real,
                future_hand_pose,
                future_hand_shape
            ) = self._extract_hand_info(
                future_hand_pose_path,
                future_depth_path,
                hand
            )

        return (
            hand_r3m_embedding.squeeze().to(torch.device('cpu')),
            robot_r3m_embedding.squeeze().to(torch.device('cpu')),
            task, hand,
            current_hand_bbox, future_hand_bbox,
            current_camera, future_camera,
            current_img_shape, future_img_shape,
            current_hand_depth_estimate, future_hand_depth_estimate,
            current_wrist_depth_real, future_wrist_depth_real,
            current_hand_pose, future_hand_pose,
            current_hand_shape, future_hand_shape,
            current_hand_pose_path, future_hand_pose_path
        )


if __name__ == '__main__':
    debug = False
    run_on_cv_server = True
    num_cpus = 4
    batch_size = 4
    time_interval = 20
    test_ss_r3m = False
    test_ss_hand_demos_r3m = False
    test_ss_robot_demos_r3m = False
    test_ss_same_hand_demos_r3m = True

    if test_ss_r3m:
        # test SomethingSomethingR3M
        if run_on_cv_server:
            task_names = [
                'push_left',
                'push_right',
                'move_down',
                'move_up',
            ]
            data_home_dir = '/home/junyao/Datasets/something_something_processed'
        else:
            task_names = [
                'move_away',
                'move_towards',
                'move_down',
                'move_up',
                'pull_left',
                'pull_right',
                'push_left',
                'push_right',
                'push_slightly'
            ]
            data_home_dir = '/scratch/junyao/Datasets/something_something_processed'

        start = time.time()
        valid_data = SomethingSomethingR3M(
            task_names, data_home_dir=data_home_dir, train=False,
            debug=debug, run_on_cv_server=run_on_cv_server, num_cpus=num_cpus
        )
        end = time.time()
        print(f'Loaded valid data. Time: {end - start}')
        print(f'Number of valid data: {len(valid_data)}')

        print('Creating data loaders...')
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=True,
            num_workers=0 if debug else num_cpus, drop_last=True)

        print('Creating data loaders: done')
        for step, data in enumerate(valid_queue):
            (
                hand_r3m_embedding, robot_r3m_embedding,
                task, hand,
                current_hand_bbox, future_hand_bbox,
                current_camera, future_camera,
                current_img_shape, future_img_shape,
                current_hand_depth_estimate, future_hand_depth_estimate,
                current_wrist_depth_real, future_wrist_depth_real,
                current_hand_pose, future_hand_pose,
                current_hand_shape, future_hand_shape,
                current_hand_pose_path, future_hand_pose_path
            ) = data
            print(f'robot_r3m_embedding: \n{robot_r3m_embedding[0]}')
            print(f'task: {task}')
            print(f'hand: {hand}')
            if step == 3:
                break

    if test_ss_hand_demos_r3m:
        # test SomethingSomethingHandDemosR3M
        task_names = [
            'move_away',
            'move_towards',
            'move_down',
            'move_up',
            'pull_left',
            'pull_right',
            'push_left',
            'push_right',
            'push_slightly'
        ]
        data_home_dir = '/home/junyao/Datasets/something_something_hand_demos'

        start = time.time()
        data = SomethingSomethingDemosR3M(
            task_names, data_home_dir=data_home_dir, time_interval=20,
            debug=debug, run_on_cv_server=run_on_cv_server, demo_type='hand'
        )
        end = time.time()
        print(f'Loaded data. Time: {end - start}')
        print(f'Number of data: {len(data)}')

        print('Creating data loaders...')
        queue = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True,
            num_workers=0 if debug else 1, drop_last=True
        )

        print('Creating data loaders: done')
        for step, data in enumerate(queue):
            (
                hand_r3m_embedding, robot_r3m_embedding,
                task, hand,
                current_hand_bbox, future_hand_bbox,
                current_camera, future_camera,
                current_img_shape, future_img_shape,
                current_hand_depth_estimate, future_hand_depth_estimate,
                current_wrist_depth_real, future_wrist_depth_real,
                current_hand_pose, future_hand_pose,
                current_hand_shape, future_hand_shape,
                current_hand_pose_path, future_hand_pose_path
            ) = data
            print(f'robot_r3m_embedding: \n{robot_r3m_embedding[0]}')
            print(f'task: \n{task}')
            print(f'hand: \n{hand}')
            if step == 3:
                break

    if test_ss_robot_demos_r3m:
        # test SomethingSomethingHandDemosR3M
        task_names = [
            'move_away',
            'move_towards',
            'move_down',
            'move_up',
            'pull_left',
            'pull_right',
            'push_left',
            'push_right',
            'push_slightly'
        ]
        data_home_dir = '/home/junyao/Datasets/something_something_robot_demos'

        start = time.time()
        data = SomethingSomethingDemosR3M(
            task_names, data_home_dir=data_home_dir, time_interval=1,
            debug=debug, run_on_cv_server=run_on_cv_server, demo_type='robot'
        )
        end = time.time()
        print(f'Loaded data. Time: {end - start}')
        print(f'Number of data: {len(data)}')

        print('Creating data loaders...')
        queue = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True,
            num_workers=0 if debug else 1, drop_last=True
        )

        print('Creating data loaders: done')
        for step, data in enumerate(queue):
            (
                hand_r3m_embedding, robot_r3m_embedding,
                task, hand,
                current_hand_bbox, future_hand_bbox,
                current_camera, future_camera,
                current_img_shape, future_img_shape,
                current_hand_depth_estimate, future_hand_depth_estimate,
                current_wrist_depth_real, future_wrist_depth_real,
                current_hand_pose, future_hand_pose,
                current_hand_shape, future_hand_shape,
                current_hand_pose_path, future_hand_pose_path
            ) = data
            print(f'robot_r3m_embedding: \n{robot_r3m_embedding[0]}')
            print(f'task: \n{task}')
            print(f'hand: \n{hand}')
            if step == 3:
                break

    if test_ss_same_hand_demos_r3m:
        data_home_dir = '/home/junyao/Datasets/something_something_hand_demos_same_hand'

        start = time.time()
        data = SomethingSomethingDemosR3M(
            CLUSTER_TASKS, data_home_dir=data_home_dir, time_interval=1,
            debug=debug, run_on_cv_server=run_on_cv_server, demo_type='same_hand'
        )
        end = time.time()
        print(f'Loaded data. Time: {end - start}')

        print('Creating data loaders...')
        queue = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True,
            num_workers=0 if debug else 1, drop_last=True
        )

        print('Creating data loaders: done')
        for step, data in enumerate(queue):
            (
                hand_r3m_embedding, robot_r3m_embedding,
                task, hand,
                current_hand_bbox, future_hand_bbox,
                current_camera, future_camera,
                current_img_shape, future_img_shape,
                current_hand_depth_estimate, future_hand_depth_estimate,
                current_wrist_depth_real, future_wrist_depth_real,
                current_hand_pose, future_hand_pose,
                current_hand_shape, future_hand_shape,
                current_hand_pose_path, future_hand_pose_path
            ) = data
            print(f'robot_r3m_embedding: \n{robot_r3m_embedding[0]}')
            print(f'task: \n{task}')
            print(f'hand: \n{hand}')
            if step == 3:
                break
