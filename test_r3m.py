import os
import time

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from os.path import join

from r3m import load_r3m

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

## DEFINE PREPROCESSING
transforms = T.Compose([T.Resize((224, 224)), T.ToTensor()]) # ToTensor() divides by 255

## ENCODE IMAGE
img_dir = '/home/junyao/Datasets/something_something_processed/push_left_right/train/frames/0'
imgs_paths = [join(img_dir, p) for p in os.listdir(img_dir)]
start = time.time()
for imgs_path in imgs_paths:
    image = Image.open(imgs_path)
    preprocessed_image = transforms(image).reshape(-1, 3, 224, 224)
    preprocessed_image.to(device)
    with torch.no_grad():
      embedding = r3m(preprocessed_image * 255.0) ## R3M expects image input to be [0-255]
    # print(embedding.shape) # [1, 2048]
end = time.time()
print(f'10 runs of R3M took {end - start} seconds')