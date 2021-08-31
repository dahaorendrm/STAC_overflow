from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This is where our downloaded images and metadata live locally
DATA_PATH = Path(".")
train_metadata = pd.read_csv(
    DATA_PATH / "flood-training-metadata.csv", parse_dates=["scene_start"]
)
df = pd.DataFrame(train_metadata)
df.to_csv("flood-training-metadata-original.csv")
#train_metadata.head()
#train_metadata.shape
#train_metadata.chip_id.nunique()
#flood_counts = train_metadata.groupby("flood_id")["chip_id"].nunique()
#flood_counts.describe()


# location_counts = (
#     train_metadata.groupby("location")["chip_id"].nunique().sort_values(ascending=False)
# )
# plt.figure(figsize=(12, 4))
# location_counts.plot(kind="bar", color="lightgray")
# plt.xticks(rotation=45)
# plt.xlabel("Location")
# plt.ylabel("Number of Chips")
# plt.title("Number of Chips by Location")

# year = train_metadata.scene_start.dt.year
# year_counts = train_metadata.groupby(year)["flood_id"].nunique()
# year_counts

# train_metadata.groupby("flood_id")["scene_start"].nunique()

from pandas_path import path

train_metadata["feature_path"] = (
    str(DATA_PATH / "train_features")
    / train_metadata.image_id.path.with_suffix(".tif").path
)
train_metadata["label_path"] = (
    str(DATA_PATH / "train_labels")
    / train_metadata.chip_id.path.with_suffix(".tif").path
)
df = pd.DataFrame(train_metadata)
df.to_csv("flood-training-metadata.csv")
