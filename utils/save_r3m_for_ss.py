import os
from os.path import join
import json
from PIL import Image
from tqdm import tqdm
import pickle
import argparse

import torch
import torchvision.transforms as T

from r3m import load_r3m

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Save r3m embeddings for images')
    parser.add_argument('--input_dir', type=str,
                        help='directory to data (parent directory of "frames" and "IoU_xx.json")',
                        default='/home/junyao/Datasets/something_something_hand_demos')
    parser.add_argument('--iou_thresh', dest='iou_thresh', type=float, required=True,
                        help='threshold for filtering data with hand and mesh bbox IoU',
                        default=0.7)
    parser.add_argument('--robot_demos', action='store_true',
                        help='if true, use something-something robot demo dataset')
    args = parser.parse_args()
    return args


def main(args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    r3m = load_r3m("resnet50") # resnet18, resnet34
    r3m.eval()
    r3m.to(device)

    ## DEFINE PREPROCESSING
    transforms = T.Compose([T.Resize((224, 224)), T.ToTensor()]) # ToTensor() divides by 255

    print(f'\nSaving R3M embeddings for {"robot" if args.robot_demos else "hand"} data at f{args.input_dir}')
    json_path = join(args.input_dir, f'IoU_{args.iou_thresh}.json')
    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    if args.robot_demos:
        frames_dir = join(args.input_dir, 'robot_frames')
        r3m_dir = join(args.input_dir, 'robot_r3m')
    else:
        frames_dir = join(args.input_dir, 'frames')
        r3m_dir = join(args.input_dir, 'r3m')

    n_processed, n_skipped = 0, 0
    for vid_num in tqdm(
        json_dict,
        desc=f'Going through {len(json_dict)} videos...'
    ):
        frames_vid_dir = join(frames_dir, vid_num)
        r3m_vid_dir = join(r3m_dir, vid_num)
        os.makedirs(r3m_vid_dir, exist_ok=True)
        for frame_num in json_dict[vid_num]:
            frame_img_path = join(frames_vid_dir, f'frame{frame_num}.jpg')
            frame_r3m_path = join(r3m_vid_dir, f'frame{frame_num}_r3m.pkl')
            if os.path.exists(frame_r3m_path):
                n_skipped += 1
                continue
            image = Image.open(frame_img_path)
            preprocessed_image = transforms(image).reshape(-1, 3, 224, 224)
            preprocessed_image.to(device)
            with torch.no_grad():
                frame_r3m = r3m(preprocessed_image * 255.0)  ## R3M expects image input to be [0-255]
            with open(frame_r3m_path, 'wb') as f:
                pickle.dump(frame_r3m, f)
            n_processed += 1

    print(f'Saved R3M embeddings. Total selected frames: {n_processed + n_skipped}; '
          f'Processed frames: {n_processed}; Skipped frames: {n_skipped}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
