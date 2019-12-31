import numpy as np

CONFIG = {
    "CUDA" : True,
    "ngpu" : 4,
    "dataloading" : {
        "classes" : 1,
        "batch_size" : 8,
        "dataset_path" : ["./data/f01/", "./data/f02/"],
        "num_workers" : 0,
    },
    "logging" : {
        "path_to_log_file" : "./logs/logger/"
    },
    "optimizer" : {
        "lr" : 3e-4,
        "momentum" : 0.9,
        "weigt_decay" : 0,
        "step_size" : 50,
        "scheduler_gamma" : 0.1
    },
    "train_settings" : {
        "epochs" : 200,
        "print_freq" : 1,
        "path_to_save_model" : "",
        "path_to_info" : ""
    }
}
