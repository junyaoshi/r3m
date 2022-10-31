import json
import os
import os.path as osp
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

lr_dict = {'0': 'same', '1': 'left', '2': 'right'}
tb_dict = {'0': 'same', '1': 'top', '2': 'bottom'}
at_dict = {'0': 'same', '1': 'away', '2': 'towards'}

def label_data(data_home_dir, label_path):
    print(f'Begin labeling data at: {data_home_dir}.')
    frames_dir = osp.join(data_home_dir, 'frames')

    if osp.exists(label_path):
        print(f'Loading existing data file at: {label_path}')
        with open(label_path, 'r') as f:
            labels = json.load(f)
        vid_frame_num_pairs = []
        for vid_num, frame_names in labels.items():
            for frame_name in frame_names:
                vid_frame_num_pairs.append((vid_num, frame_name))
    else:
        # get a list of all frames
        vid_nums = [vid_num for vid_num in os.listdir(frames_dir)]
        labels = {vid_num: {} for vid_num in os.listdir(frames_dir)}
        vid_frame_num_pairs = []
        for vid_num in vid_nums:
            vid_dir = osp.join(frames_dir, vid_num)
            for fname in os.listdir(vid_dir):
                frame_name = fname.split('.')[0]
                vid_frame_num_pairs.append((vid_num, frame_name))
                labels[vid_num][frame_name] = {}

    # debugging
    vid_frame_num_pairs = vid_frame_num_pairs[:5]

    n_skipped, n_total = 0, len(vid_frame_num_pairs)
    print(f'There are a total of {n_total} frames.\n')

    save_prompt_freq = 10
    for (i, (vid_num, frame_name)) in tqdm(enumerate(vid_frame_num_pairs), desc='Labeling frames'):
        if labels[vid_num][frame_name]:
            n_skipped += 1
            continue
        frame_path = osp.join(frames_dir, vid_num, f'{frame_name}.jpg')
        frame = cv2.imread(frame_path)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()
        valid_xyz, xyz = False, ''
        while not valid_xyz:
            xyz = input('\nRelative to the hand, the object is '
                        '\nabout same(0)/left(1)/right(2) along x axis, '
                        '\nabout same(0)/top(1)/bottom(2) along y axis, '
                        '\nabout same(0)/away(1)/towards(2) along z axis?\n')
            xyz = str(xyz)
            if len(xyz) == 3 and all(a in ['0', '1', '2'] for a in xyz):
                valid_xyz = True
            else:
                print(f'{xyz} is an invalid answer, please try again.')
        print(f'Answer: {lr_dict[xyz[0]]}; {tb_dict[xyz[1]]}; {at_dict[xyz[2]]}')

        labels[vid_num][frame_name]['frame_path'] = frame_path
        labels[vid_num][frame_name]['left_right'] = xyz[0]
        labels[vid_num][frame_name]['top_bottom'] = xyz[1]
        labels[vid_num][frame_name]['away_towards'] = xyz[2]

        if (i + 1) % save_prompt_freq == 0:
            valid_answer, answer = False, ''
            while not valid_answer:
                answer = input(f'\nReached step {i + 1}/{n_total}'
                               '\nDo you want to save current labeling progress? '
                               'Answer Yes(1)/No(0).\n')
                answer = str(answer)
                if answer in ['0', '1']:
                    valid_answer = True
                else:
                    print(f'{answer} is an invalid answer, please try again.')
            if answer == '0':
                print('Skip saving current labeling progress.')
            else:
                with open(label_path, 'w') as f:
                    json.dump(labels, f)
                print(f'Saved current labeling progress at: {label_path}')

    with open(label_path, 'w') as f:
        json.dump(labels, f)

    print(f'\nDone with labeling data at: {data_home_dir}')
    print(f'skipped/total: {n_skipped}/{n_total}')
    print(f'Saved labeled data at: {label_path}')

if __name__ == '__main__':
    label_data(
        data_home_dir='/home/junyao/Franka_Datasets/something_something_hand_demos/pre_interaction',
        label_path='/home/junyao/Franka_Datasets/something_something_hand_demos/'
                   'pre_interaction/pre_interaction_labels.json'
    )
