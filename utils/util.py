import logging

import PIL.Image as Image

from config import CONFIG

class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum/self.cnt

    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return self.__str__()

def get_loggers(file_path):
    logger = logging.getLogger("dpnet")
    log_formats = "%(asctime)s | %(message)s"
    formatter = logging.Formatter(log_formats, datefmt="%m/%d %I:%M:%S %[")

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def save_image(image, image_path):
    image = image.clone().detach().cpu()
    image = (image.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    image = Image.fromarray(image[:, :, 0], mode="L")
    image.save(image_path)
