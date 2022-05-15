import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.io import read_image

from pathlib import Path
from PIL import Image, ImageDraw


import torchvision.transforms.functional as F


def show(imgs):
    if not isinstance(imgs, list):
        fix, axs = plt.subplots(ncols=1, squeeze=False)
        img = imgs.detach()
        img = F.to_pil_image(img)
        axs[0, 0].imshow(np.asarray(img))
        axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    else:
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def load_annotations(annotation_file):
    annotations = {}
    with open(annotation_file, "r") as read_file:
        data = json.load(read_file)
        for d in data.values():
            annotations[d['filename']]=d['regions']
    return annotations

def show_annotations(annotations):
    for an in annotations:
        car_img_path = Path(f'train/{an}')
        if car_img_path.is_file():
            car_img = read_image(str(car_img_path))
            show(car_img)

            mask_img = return_mask(car_img, annotations[an])
            show(mask_img)

def return_mask(img, regions):
    mask = torch.zeros(img.shape)
    mask = F.to_pil_image(mask)
    img1 = ImageDraw.Draw(mask)
    for r in regions:
        img1.polygon(list(zip(regions[r]['shape_attributes']['all_points_x'], regions[r]['shape_attributes']['all_points_y'])), fill ="white", outline ="white")
    img = F.to_tensor(mask)
    return img

if __name__ == "__main__":
    annotation_file = 'train/via_region_data.json'
    annotations = load_annotations(annotation_file)
    show_annotations(annotations)
