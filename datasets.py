import json
import os
from os.path import join
from tqdm import tqdm
import pickle
import cv2

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from bc_utils import determine_which_hand, xywh_to_xyxy, normalize_bbox


DATA_HOME_DIR = '/home/junyao/Datasets/something_something_processed'

class SomethingSomethingR3M(Dataset):
    def __init__(self, task_names, data_home_dir,
                 iou_thresh=0.7, time_interval=5, train=True, debug=False, run_on_cv_server=False):
        print(f'Initializing dataset for tasks: {task_names}. \nSplit: {"train" if train else "valid"}')
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.task_dict = {task_name: int(i) for i, task_name in enumerate(task_names)}
        self.run_on_cv_server = run_on_cv_server

        self.r3m_paths, self.tasks, self.hands  = [], [], []
        self.current_hand_pose_paths, self.future_hand_pose_paths = [], []

        done = False
        for task_name in task_names:
            print(f'Processing task: {task_name}.')
            split = 'train' if train else 'valid'
            split_dir = join(data_home_dir, task_name, split)
            r3m_dir = join(split_dir, 'r3m')
            iou_json_path = join(split_dir, f'IoU_{iou_thresh}.json')
            with open(iou_json_path, 'r') as f:
                json_dict = json.load(f)
                f.close()

            for vid_num in tqdm(json_dict, desc='Going through videos: '):
                r3m_vid_dir = join(r3m_dir, vid_num)
                mocap_vid_dir = join(split_dir, 'mocap_output', vid_num, 'mocap')
                for current_frame_num in json_dict[vid_num]:
                    # check if future frame exists
                    future_frame_num = str(int(current_frame_num) + time_interval)
                    if future_frame_num not in json_dict[vid_num]:
                        continue

                    # check if current and future frames have the same hand
                    current_hand_pose_path = join(mocap_vid_dir, f'frame{current_frame_num}_prediction_result.pkl')
                    future_hand_pose_path  = join(mocap_vid_dir, f'frame{future_frame_num}_prediction_result.pkl')
                    with open(current_hand_pose_path, 'rb') as f:
                        current_hand_info = pickle.load(f)
                    current_hand = determine_which_hand(current_hand_info)
                    with open(future_hand_pose_path, 'rb') as f:
                        future_hand_info = pickle.load(f)
                    future_hand = determine_which_hand(future_hand_info)
                    if current_hand != future_hand:
                        continue

                    self.hands.append(current_hand)
                    self.r3m_paths.append(join(r3m_vid_dir, f'frame{current_frame_num}_r3m.pkl'))
                    self.tasks.append(F.one_hot(
                        torch.Tensor([self.task_dict[task_name]]).to(torch.int64),
                        num_classes=self.num_tasks
                    ).squeeze(0))
                    self.current_hand_pose_paths.append(current_hand_pose_path)
                    self.future_hand_pose_paths.append(future_hand_pose_path)

                if debug and len(self.r3m_paths) > 300:
                    done = True
                    break

            if done: break

        print(f'Dataset has {len(self)} data.')

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
        current_image_path = current_hand_info['image_path']
        if current_image_path[:8] == '/scratch' and self.run_on_cv_server:
            current_image_path = '/home' + current_image_path[8:]
        current_image = cv2.imread(current_image_path)
        current_hand_bbox = normalize_bbox(current_hand_bbox, (current_image.shape[1], current_image.shape[0]))

        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        future_hand_pose = future_hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        future_hand_bbox = future_hand_info['hand_bbox_list'][0][hand]
        future_camera = future_hand_info['pred_output_list'][0][hand]['pred_camera']
        future_image_path = future_hand_info['image_path']
        if future_image_path[:8] == '/scratch' and self.run_on_cv_server:
            future_image_path = '/home' + future_image_path[8:]
        future_image = cv2.imread(future_image_path)
        future_hand_bbox = normalize_bbox(future_hand_bbox, (future_image.shape[1], future_image.shape[0]))

        return r3m_embedding.squeeze().to(torch.device('cpu')), task, hand, \
               current_hand_bbox, future_hand_bbox, \
               current_camera, future_camera, \
               current_hand_pose, future_hand_pose, \
               current_hand_pose_path, future_hand_pose_path

if __name__ == '__main__':
    debug = False
    task_names = ['push_left_right']
    train_data = SomethingSomethingR3M(
        task_names, data_home_dir=DATA_HOME_DIR, train=True, debug=debug, run_on_cv_server=True
    )
    print(f'Number of train data: {len(train_data)}')
    valid_data = SomethingSomethingR3M(
        task_names, data_home_dir=DATA_HOME_DIR, train=False, debug=debug, run_on_cv_server=True
    )
    print(f'Number of valid data: {len(valid_data)}')

    print('Creating data loaders...')
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=10, shuffle=True,
        num_workers=0 if debug else 8, drop_last=True)

    print('Creating data loaders: done')
    for step, data in enumerate(valid_queue):
        r3m_embedding, task, hand, \
        current_hand_bbox, future_hand_bbox, \
        current_camera, future_camera, \
        current_hand_pose, future_hand_pose, \
        current_hand_pose_path, future_hand_pose_path = data
        if step == 3:
            break
