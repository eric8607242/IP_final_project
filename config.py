import numpy as np

CONFIG = {
    "CUDA" : True,
    "ngpu" : 4,
    "dataloading" : {
        "batch_size" : 128,
        "dataset_path" : "",
        "num_workers" : 4,
    },
    "optimizer" : {
        "lr" : 1e-1,
        "momentum" : 0.9,
        "weigt_decay" : 0
    },
    "train_settings" : {
        "epochs" : 150,
        "print_freq" : 50,
        "path_to_save_model" : "",
        "path_to_info" : ""
    }
}
