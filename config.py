import numpy as np

CONFIG = {
    "CUDA" : True,
    "ngpu" : 4,
    "dataloading" : {
        "batch_size" : 128,
        "dataset_path" : ["./data/f01/", "./data/f02/"],
        "num_workers" : 4,
    },
    "logging" : {
        "path_to_log_file" : "./logs/logger/"
    },
    "optimizer" : {
        "lr" : 1e-1,
        "momentum" : 0.9,
        "weigt_decay" : 0,
        "step_size" : 10,
        "scheduler_gamma" : 0.1
    },
    "train_settings" : {
        "epochs" : 150,
        "print_freq" : 50,
        "path_to_save_model" : "",
        "path_to_info" : ""
    }
}
