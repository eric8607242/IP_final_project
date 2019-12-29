from os import listdir
from os.path import isfile, join

import PIL.Image
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class VertebraDataset(Dataset):
    def __init__(self, data_path, input_size=224):
        self.input_size = input_size
        self.images, self.labels = self._load_image_label(data_path)


    def _load_image_label(self, paths):
        images = []
        labels = []

        paths = paths if isinstance(paths, list) else paths
        for path in paths:
            images_path = join(path, "image")
            labels_path = join(path, "label")

            images.append([self._image_preprocess(join(images_path, f)) for f in listdir(images_path) if isfile(join(images_path, f))])
            labels.append([self._image_preprocess(join(labels_path, f)) for f in listdir(labels_path) if isfile(join(labels_path, f))])

        return images, labels
        
    def _image_preprocess(self, image_path):
        image = PIL.Image.open(image_path)
        image = image.resize((self.input_size, self.input_size), PIL.Image.ANTIALIAS)

        image = np.array(image)
        image = image[np.newaxis, ...]

        return image


    def _transform(self, image, label):
        #resize = transforms.Resize(size=(r_seed, r_seed))
        #image = resize(image)
        #landmark = resize(landmark)
        
        #pixel = np.array(image)
        
        #pad = transforms.Pad(p_value, (pixel[0, 0, 0], pixel[0, 0, 1], pixel[0, 0, 2]))
        #image = pad(image)
        #landmark = pad(landmark)

        #i, j, h, w = transforms.RandomCrop.get_params(
        #        image, output_size=(256, 256))
        #image = TF.crop(image, i, j, h, w)
        #landmark = TF.crop(landmark, i, j, h, w)

        if random.random() > 0.5:
            image = TF.hflip(image)
            landmark = TF.hflip(landmark)

        #resize = transforms.Resize(size=(self.input_size, self.input_size))
        #image = resize(image)
        #label = resize(label)

        image = TF.to_tensor(image)
        label = TF.to_tensor(label)

        return image, landmark
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image, label = self._transform(image, label)

        return image, label


        
