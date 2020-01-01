import random

from os import listdir
from os.path import isfile, join

import cv2
import PIL.Image
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from config import CONFIG


class VertebraDataset(Dataset):
    def __init__(self, data_path, input_size=224):
        self.classes = CONFIG["dataloading"]["classes"]
        self.input_size = input_size

        self.images, self.labels = self._load_image_label(data_path)

    def _load_image_label(self, paths):
        images = []
        labels = []

        paths = paths if isinstance(paths, list) else paths
        for path in paths:
            images_path = join(path, "image")
            labels_path = join(path, "label")

            images.extend([self._image_preprocess(join(images_path, f)) for f in listdir(images_path) if isfile(join(images_path, f))])
            labels.extend([self._image_preprocess(join(labels_path, f)) for f in listdir(labels_path) if isfile(join(labels_path, f))])

        return images, labels

    def process_label(self, label):
        """
        process label image to 16 image
        """
        label = np.array(label, np.uint8)
        _, label_cnts, _ = cv2.findContours(label ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        process_label = np.zeros((*label.shape, self.classes), dtype="f")
        for i, c in enumerate(label_cnts):
            process_label_axis = process_label[:, :, i]
            process_label_axis = process_label_axis.copy()

            process_label_axis = cv2.fillPoly(process_label_axis, pts =[c], color=(255))
            process_label[:, :, i] = process_label_axis

        return process_label


        
    def _image_preprocess(self, image_path):
        image = PIL.Image.open(image_path)
        image = image.convert("L")

        return image


    def _transform(self, image, label):
        image = TF.to_tensor(image)
        label = TF.to_tensor(label)


        image[0, :, :125] = 0
        image[0, :, -125:] = 0

        return image, label

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image, label = self._transform(image, label)

        return image, label


        
