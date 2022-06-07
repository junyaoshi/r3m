import os
from os.path import join
import json
from PIL import Image
from tqdm import tqdm
import pickle

import torch
import torchvision.transforms as T

from r3m import load_r3m

DATA_HOME_DIR = '/scratch/junyao/Datasets/something_something_processed'
SPLITS = ['train', 'valid']
TASK_NAMES = ['pull_left', 'pull_right', 'push_right']

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

## DEFINE PREPROCESSING
transforms = T.Compose([T.Resize((224, 224)), T.ToTensor()]) # ToTensor() divides by 255

for task_name in TASK_NAMES:
    task_data_dir = join(DATA_HOME_DIR, task_name)
    for split in SPLITS:
        split_data_dir = join(task_data_dir, split)
        print(f'Saving R3M embeddings for data at f{split_data_dir}')
        json_path = join(split_data_dir, 'IoU_0.7.json')
        with open(json_path, 'r') as f:
            json_dict = json.load(f)

        frames_dir = join(split_data_dir, 'frames')
        r3m_dir = join(split_data_dir, 'r3m')
        for vid_num in tqdm(
            json_dict,
            desc='Going through videos in this split...'
        ):
            frames_vid_dir = join(frames_dir, vid_num)
            r3m_vid_dir = join(r3m_dir, vid_num)
            os.makedirs(r3m_vid_dir, exist_ok=True)
            for frame_num in json_dict[vid_num]:
                frame_img_path = join(frames_vid_dir, f'frame{frame_num}.jpg')
                image = Image.open(frame_img_path)
                preprocessed_image = transforms(image).reshape(-1, 3, 224, 224)
                preprocessed_image.to(device)
                with torch.no_grad():
                    frame_r3m = r3m(preprocessed_image * 255.0)  ## R3M expects image input to be [0-255]
                frame_r3m_path = join(r3m_vid_dir, f'frame{frame_num}_r3m.pkl')
                with open(frame_r3m_path, 'wb') as f:
                    pickle.dump(frame_r3m, f)
