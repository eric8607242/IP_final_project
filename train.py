import torch
import torch.nn as nn

from utils.trainer import Trainer
from utils.dataset import VertebraDataset
from model.resunet import ResNetUNet
from utils.util import get_loggers
from utils.loss import dice_loss

from config import CONFIG

def train():
    if CONFIG["CUDA"]:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG["ngpu"] > 0) else "cpu")
    else:
        device = torch.device("cpu")

    train_dataset = VertebraDataset(CONFIG["dataloading"]["train_dataset_path"])
    train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=CONFIG["dataloading"]["batch_size"], num_workers=CONFIG["dataloading"]["num_workers"]
            )

    val_dataset = VertebraDataset(CONFIG["dataloading"]["val_dataset_path"])
    val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=CONFIG["dataloading"]["batch_size"], num_workers=CONFIG["dataloading"]["num_workers"]
            )

    model = ResNetUNet(CONFIG["dataloading"]["classes"])
    model = model.to(device)
    if (device.type == "cuda") and (CONFIG["ngpu"] > 1):
        model = nn.DataParallel(model, list(range(CONFIG["ngpu"])))

    criterion = dice_loss

    optimizer = torch.optim.Adam(params=model.parameters(), 
                                 lr=CONFIG["optimizer"]["lr"],
                                 weight_decay=CONFIG["optimizer"]["weigt_decay"])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=CONFIG["optimizer"]["step_size"],
                                                gamma=CONFIG["optimizer"]["scheduler_gamma"])

    logger = get_loggers(CONFIG["logging"]["path_to_log_file"])
    trainer = Trainer(criterion, optimizer, scheduler, logger, device)
    trainer.train_loop(train_loader, val_loader, model)


if __name__ == "__main__":
    train()

