import time

import torch
import torch.nn as nn

from utils.util import AverageMeter, save_image

from config import CONFIG

class Trainer:
    def __init__(self, criterion, optimizer, scheduler, logger, device):
        self.logger = logger
        self.device = device

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.losses = AverageMeter()

        self.epochs = CONFIG["train_settings"]["epochs"]
        self.print_freq = CONFIG["train_settings"]["print_freq"]

    def train_loop(self, train_loader, val_loader, model):
        for epoch in range(self.epochs):
            self.logger.info("Start to train for epoch %d" % (epoch))
            self._training_step(model, train_loader, epoch, info_for_logger="_training_step_")
            self.scheduler.step()


    def _training_step(self, model, loader, epoch, info_for_logger=""):
        model = model.train()
        start_time = time.time()

        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            N = X.shape[0]

            self.optimizer.zero_grad()

            outs = model(X)
            loss = self.criterion(outs, y)
            loss.backward()
            self.optimizer.step()

            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")


    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
        self.losses.update(loss.item(), N)

        if (step > 1 and step % self.print_freq==0) or step == len_loader-1:
            save_image(outs[0], "./pred{}.png".format(epoch))
            save_image(y[0], "./label{}.png".format(epoch))
            self.logger.info(val_or_train+
                    "[{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f}".format(
                        epoch+1, self.epochs, step, len_loader-1, self.losses.get_avg()))
