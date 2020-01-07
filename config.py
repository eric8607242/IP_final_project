import numpy as np

CONFIG = {
    "CUDA" : True,
    "ngpu" : 4,
    "dataloading" : {
        "classes" : 1,
        "batch_size" : 8,
        "train_dataset_path" : ["./data/f02/", "./data/f03/"],
        "val_dataset_path" : ["./data/f01/"],
        "num_workers" : 0,
    },
    "logging" : {
        "path_to_log_file" : "./logs/logger/"
    },
    "optimizer" : {
        "lr" : 1e-3,
        "momentum" : 0.9,
        "weigt_decay" : 2e-5,
        "step_size" : 150,
        "scheduler_gamma" : 0.1
    },
    "train_settings" : {
        "epochs" : 400,
        "print_freq" : 1,
        "path_to_save_model" : "./logs/best.pth",
        "path_to_info" : ""
    }
}
