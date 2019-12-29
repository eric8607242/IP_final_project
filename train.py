import torch

from utils.trainer import Trainer
from utils.dataset import VertebraDataset
from utils.util import get_loggers

from config import CONFIG

def train():
    if CONFIG["CUDA"]:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG["ngpu"] > 0) else "cpu")
    else:
        device = torch.device("cpu")

    dataset = VertebraDataset(CONFIG["dataloading"]["dataset_path"])
    train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=CONFIG["dataloading"]["batch_size"], num_workers=CONFIG["dataloading"]["num_workers"]
            )

    criterion = None
    optimizer = None
    scheduler = None

    logger = get_loggers(CONFIG["logging"]["path_to_log_file"])
    trainer = Trainer(criterion, optimizer, scheduler, logger, device)

if __name__ == "__main__":
    train()

