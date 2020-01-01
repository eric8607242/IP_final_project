import os
from os.path import isfile, join
import argparse

import PIL.Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from utils.dataset import VertebraDataset
from utils.util import get_loggers, load_weight
from utils.loss import dice_loss
from model.resunet import ResNetUNet

from config import CONFIG


def demo(demo_path, checkpoint_path):
    if CONFIG["CUDA"]:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG["ngpu"] > 0) else "cpu")
    else:
        device = torch.device("cpu")

    demo_data = get_demo_data(demo_path)


    model = ResNetUNet(CONFIG["dataloading"]["classes"])
    model = model.to(device)
    model = load_weight(model, checkpoint_path)

    for image in demo_data:
        image = image.to(device, non_blocking=True)

        out = model(image)

    criterion = dice_loss

def get_demo_data(demo_path):
    images = []
    if isfile(demo_path):
        images.extend([_image_preprocess(demo_path)])
    else:
        images.extend([_image_preprocess(join(demo_path, f)) for f in os.listdir(demo_path) if isfile(join(demo_path, f))])

    return images

def _image_preprocess(self, image_path):
    image = PIL.Image.open(image_path)
    image = image.convert("L")

    image = TF.to_tensor(image)
    image[:, :100] = 0
    image[:, -100:] = 0

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, required=True, help="path to the checkpoint")
    parser.add_argument("--demo-image", type=str, required=True, help="path to demo")
    args = parser.parse_args()

    demo(args.demo_image, args.checkpoint_path)

