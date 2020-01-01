import os
from os.path import isfile, join
import argparse

import numpy as np
import cv2
import PIL.Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from utils.dataset import VertebraDataset
from utils.util import get_loggers, load_weight, save_image
from utils.loss import dice_loss
from model.resunet import ResNetUNet

from config import CONFIG


def demo(demo_path, checkpoint_path, output_path):
    if CONFIG["CUDA"]:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG["ngpu"] > 0) else "cpu")
    else:
        device = torch.device("cpu")

    demo_data = get_demo_data(demo_path)


    model = ResNetUNet(CONFIG["dataloading"]["classes"])
    model = model.to(device)
    model = nn.DataParallel(model, [0])
    load_weight(model, checkpoint_path)

    for index, image in enumerate(demo_data):
        image_cuda = image.to(device, non_blocking=True)

        image_cuda[0, 0, :, :125] = 0
        image_cuda[0, 0, :, -125:] = 0
        out = model(image_cuda)

        process_output(image, out, join(output_path, str(index)+".png"))

    criterion = dice_loss

def process_output(image, output_tensor, output_path):
    image = image[0].clone().detach().cpu()*255
    output_tensor = output_tensor[0].clone().detach().cpu()*255

    output_array = np.array(output_tensor, np.uint8)
    _, output_array = cv2.threshold(output_array, 200, 255, cv2.THRESH_BINARY)


    image = np.array(image, np.uint8)
    image = np.transpose(image, (1, 2, 0))

    _, output_cnts, _ = cv2.findContours(output_array[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.drawContours(image, output_cnts, -1, (0,0,255), 3)

    cv2.imwrite(output_path, image)


def get_demo_data(demo_path):
    images = []
    if isfile(demo_path):
        images.extend([_image_preprocess(demo_path)])
    else:
        images.extend([_image_preprocess(join(demo_path, f)) for f in os.listdir(demo_path) if isfile(join(demo_path, f))])

    return images

def _image_preprocess(image_path):
    image = PIL.Image.open(image_path)
    image = image.convert("L")

    image = TF.to_tensor(image)
    image = image.unsqueeze(0)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, required=True, help="path to the checkpoint")
    parser.add_argument("--demo-image", type=str, required=True, help="path to demo")
    parser.add_argument("--output-path", type=str, required=True, help="path to output folder")
    args = parser.parse_args()

    demo(args.demo_image, args.checkpoint_path, args.output_path)

