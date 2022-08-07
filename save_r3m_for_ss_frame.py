import os
import pprint
import time

import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import pickle

from r3m import load_r3m


def setup_r3m(device):
    r3m = load_r3m("resnet50")  # resnet18, resnet34
    r3m.eval()
    r3m.to(device)

    ## DEFINE PREPROCESSING
    transforms = T.Compose([T.Resize((224, 224)), T.ToTensor()])  # ToTensor() divides by 255
    return r3m, transforms


def save_r3m(r3m, transforms, frame_path, r3m_embedding_path, device, frame_bgr=None, verbose=True):
    if verbose:
        print(f'Processing r3m embedding from: {frame_path if frame_bgr is None else "given frame image"}.')
    t0 = time.time()
    assert frame_path is not None or frame_bgr is not None
    if frame_bgr is None:
        frame_bgr = cv2.imread(frame_path)
    image_rgb_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    preprocessed_image = transforms(image_rgb_pil).reshape(-1, 3, 224, 224)
    preprocessed_image.to(device)
    with torch.no_grad():
        r3m_embedding = r3m(preprocessed_image * 255.0)  ## R3M expects image input to be [0-255]

    if r3m_embedding_path is not None:
        os.makedirs(os.path.dirname(r3m_embedding_path), exist_ok=True)
        with open(r3m_embedding_path, 'wb') as f:
            pickle.dump(r3m_embedding, f)
        if verbose:
            print(f'Saved r3m embedding to: {r3m_embedding_path} ')

    t1 = time.time()
    if verbose:
        print(f'Processed r3m embedding in {t1 - t0:.4f} seconds.')

    return r3m_embedding


if __name__ == '__main__':
    device = 'cuda'
    frame_bgr = cv2.imread('/home/junyao/test/frame10.jpg')

    r3m, transforms = setup_r3m(device)
    r3m_embedding = save_r3m(
        r3m=r3m,
        transforms=transforms,
        frame_path=None,
        r3m_embedding_path='/home/junyao/test/frame10_r3m.pkl',
        device=device,
        frame_bgr=frame_bgr
    )

    pprint.pprint(r3m_embedding)