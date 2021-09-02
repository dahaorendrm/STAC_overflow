# import
from pathlib import Path
import numpy as np
import pandas as pd
import random
random.seed(9) # set a seed for reproducibility
import utils
import rasterio
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
val_x
# set-up model

# run model

# results
