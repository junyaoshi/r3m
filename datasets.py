import json
import math
import time
from os.path import join

import numpy as np
from tqdm import tqdm
import pickle
import cv2
import multiprocessing as mp

import torch
from torch.utils.data import Dataset

from bc_utils import determine_which_hand, normalize_bbox

class SomethingSomethingR3M(Dataset):
    def __init__(self, task_names, data_home_dir,
                 iou_thresh=0.7, time_interval=5, train=True, debug=False, run_on_cv_server=False, num_cpus=4):
        """
        Set num_cpus=1 to disable multiprocessing
        """
        print(f'Initializing dataset for tasks: {task_names}. \nSplit: {"train" if train else "valid"}')
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.data_home_dir = data_home_dir
        self.iou_thresh = iou_thresh
        self.time_interval = time_interval
        self.train = train
        self.debug = debug
        self.run_on_cv_server = run_on_cv_server
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

    def __getitem__(self, idx):
        r3m_path = self.r3m_paths[idx]
        task = self.tasks[idx]
        current_hand_pose_path = self.current_hand_pose_paths[idx]
        future_hand_pose_path = self.future_hand_pose_paths[idx]
        hand = self.hands[idx]

        with open(r3m_path, 'rb') as f:
            r3m_embedding = pickle.load(f)

        with open(current_hand_pose_path, 'rb') as f:
            current_hand_info = pickle.load(f)
        current_hand_pose = current_hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        current_hand_bbox = current_hand_info['hand_bbox_list'][0][hand]
        current_camera = current_hand_info['pred_output_list'][0][hand]['pred_camera']
        current_img_shape = current_hand_info['image_shape']
        current_hand_bbox = normalize_bbox(current_hand_bbox, (current_img_shape[1], current_img_shape[0]))
        current_hand_shape = current_hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)

        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        future_hand_pose = future_hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        future_hand_bbox = future_hand_info['hand_bbox_list'][0][hand]
        future_camera = future_hand_info['pred_output_list'][0][hand]['pred_camera']
        future_img_shape = future_hand_info['image_shape']
        future_hand_bbox = normalize_bbox(future_hand_bbox, (future_img_shape[1], future_img_shape[0]))
        future_joint_depth = future_hand_info['pred_output_list'][0][hand]['pred_joints_img'][:, 2]
        future_hand_shape = future_hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)

        return (
            r3m_embedding.squeeze().to(torch.device('cpu')), task, hand,
            current_hand_bbox, future_hand_bbox,
            current_camera, future_camera,
            future_img_shape, future_joint_depth,
            current_hand_pose, future_hand_pose,
            current_hand_shape, future_hand_shape,
            current_hand_pose_path, future_hand_pose_path
        )

if __name__ == '__main__':
    debug = False
    run_on_cv_server = False
    num_cpus = 4
    batch_size = 4

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
            # 'move_towards',
            # 'move_down',
            # 'move_up',
            # 'pull_left',
            # 'pull_right',
            # 'push_left',
            # 'push_right',
            # 'push_slightly'
        ]
        data_home_dir = '/scratch/junyao/Datasets/something_something_processed'
    # start = time.time()
    # train_data = SomethingSomethingR3M(
    #     task_names, data_home_dir=data_home_dir, train=True,
    #     debug=debug, run_on_cv_server=run_on_cv_server, num_cpus=num_cpus
    # )
    # end = time.time()
    # print(f'Loaded train data. Time: {end - start}')
    # print(f'Number of train data: {len(train_data)}')

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
            r3m_embedding, task, hand,
            current_hand_bbox, future_hand_bbox,
            current_camera, future_camera,
            future_img_shape, future_joint_depth,
            current_hand_pose, future_hand_pose,
            current_hand_shape, future_hand_shape,
            current_hand_pose_path, future_hand_pose_path
        ) = data
        print(f'future_img_shape: \n{future_img_shape}')
        print(f'current_hand_shape: \n{current_hand_shape}')
        print(f'future_hand_shape: \n{future_hand_shape}')
        if step == 3:
            break
