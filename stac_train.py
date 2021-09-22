# import
from pathlib import Path
import numpy as np
import pandas as pd
import random
random.seed(9) # set a seed for reproducibility
import utils
import torch

from networks.STAC_model import FloodModel
import loss
# process data
DATA_PATH = Path("training_data")
train_metadata = pd.read_csv(
    DATA_PATH / "flood-training-metadata.csv", parse_dates=["scene_start"]
)
# Sample 3 random floods for validation set
flood_ids = train_metadata.flood_id.unique().tolist()
val_flood_ids = random.sample(flood_ids, 3)
# Split data in two sets
val = train_metadata[train_metadata.flood_id.isin(val_flood_ids)]
train = train_metadata[~train_metadata.flood_id.isin(val_flood_ids)]
# Separate features from labels
val_x = utils.get_paths_by_chip(val)
val_y = val[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)
train_x = utils.get_paths_by_chip(train)
train_y = train[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)

# set-up model
hparams = {
    # Required hparams
    "x_train": train_x,
    "x_val": val_x,
    "y_train": train_y,
    "y_val": val_y,
    # Optional hparams
    "backbone": "resnet34",
    "weights": "imagenet",
    "lr": 1e-3,
    "min_epochs": 20,
    "max_epochs": 1000,
    "patience": 4,
    "batch_size": 32,
    "num_workers": 88888888,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": "model-outputs",
    "log_path": "tensorboard_logs",
    "gpu": torch.cuda.is_available(),
}

flood_model = FloodModel(hparams=hparams)

# run model
flood_model.fit()
# results
print(f'Best IOU score is : {flood_model.trainer_params["callbacks"][0].best_model_score}')
# save the weights to submitssion file
weight_path = "model-outputs/flood_model.pt"
torch.save(flood_model.state_dict(), weight_path)
