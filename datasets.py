import json
import os
from os.path import join
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


DATA_HOME_DIR = '/home/junyao/Datasets/something_something_processed'

class SomethingSomethingR3M(Dataset):
    def __init__(self, task_names, iou_thresh=0.7, time_interval=5, train=True, debug=False):
        print(f'Initializing dataset for {task_names} tasks. Train split: {train}')
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.task_dict = {task_name: int(i) for i, task_name in enumerate(task_names)}

        self.r3m_paths, self.tasks  = [], []
        self.current_handpose_paths, self.target_handpose_paths = [], []

        for task_name in task_names:
            print(f'Processing task: {task_name}.')
            split = 'train' if train else 'valid'
            split_dir = join(DATA_HOME_DIR, task_name, split)
            r3m_dir = join(split_dir, 'r3m')
            iou_json_path = join(split_dir, f'IoU_{iou_thresh}.json')
            with open(iou_json_path, 'r') as f:
                json_dict = json.load(f)
                f.close()

            for vid_num in tqdm(json_dict, desc='Going through videos: '):
                r3m_vid_dir = join(r3m_dir, vid_num)
                mocap_vid_dir = join(split_dir, 'mocap_output', vid_num, 'mocap')
                for current_frame_num in json_dict[vid_num]:
                    # check if target frame exists
                    target_frame_num = str(int(current_frame_num) + time_interval)
                    if target_frame_num not in json_dict[vid_num]:
                        continue

                    self.r3m_paths.append(join(r3m_vid_dir, f'frame{current_frame_num}_r3m.pkl'))
                    self.tasks.append(F.one_hot(
                        torch.Tensor([self.task_dict[task_name]]).to(torch.int64),
                        num_classes=self.num_tasks
                    ))
                    self.current_handpose_paths.append(join(
                        mocap_vid_dir,
                        f'frame{current_frame_num}_prediction_result.pkl'
                    ))
                    self.target_handpose_paths.append(join(
                        mocap_vid_dir,
                        f'frame{target_frame_num}_prediction_result.pkl'
                    ))

        if debug:
            self.r3m_paths, self.tasks = self.r3m_paths[:50], self.tasks[:50]
            self.current_handpose_paths = self.current_handpose_paths[:50]
            self.target_handpose_paths = self.target_handpose_paths[:50]

        print(f'Dataset has {len(self)} data.')

    def __len__(self):
        return len(self.r3m_paths)

    def __getitem__(self, idx):
        r3m_path = self.r3m_paths[idx]
        task = self.tasks[idx]
        current_handpose_path = self.current_handpose_paths[idx]
        target_handpose_path = self.target_handpose_paths[idx]

        with open(r3m_path, 'rb') as f:
            r3m_embedding = pickle.load(f)
        f.close()

        with open(current_handpose_path, 'rb') as f:
            current_hand_info = pickle.load(f)
        current_hand = current_hand_info['pred_output_list'][0]['left_hand'] \
            if len(current_hand_info['pred_output_list'][0]['left_hand']) > 0 \
            else current_hand_info['pred_output_list'][0]['right_hand']
        current_joints_smpl = current_hand['pred_joints_smpl'].reshape(63)
        f.close()

        with open(target_handpose_path, 'rb') as f:
            target_hand_info = pickle.load(f)
        target_hand = target_hand_info['pred_output_list'][0]['left_hand'] \
            if len(target_hand_info['pred_output_list'][0]['left_hand']) > 0 \
            else target_hand_info['pred_output_list'][0]['right_hand']
        target_joints_smpl = target_hand['pred_joints_smpl'].reshape(63)
        f.close()

        delta_joints_smpl = target_joints_smpl - current_joints_smpl

        return r3m_embedding, task, 500. * delta_joints_smpl

if __name__ == '__main__':
    debug = False
    task_names = ['push_left_right']
    train_data = SomethingSomethingR3M(task_names, train=True, debug=debug)
    valid_data = SomethingSomethingR3M(task_names, train=False, debug=debug)

    print('Creating data loaders...')
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=10, shuffle=True,
        num_workers=0 if debug else 8, drop_last=True)

    print('Creating data loaders: done')
    for step, data in enumerate(valid_queue):
        r3m_embedding, task, delta_joints_smpl = data
        if step == 3:
            break
