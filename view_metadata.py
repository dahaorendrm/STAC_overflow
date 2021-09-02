from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from pandas_path import path
# This is where our downloaded images and metadata live locally
DATA_PATH = Path("training_data")
train_metadata = pd.read_csv(
    DATA_PATH / "flood-training-metadata-original.csv", parse_dates=["scene_start"]
)
train_metadata
#print(train_metadata.head())

train_metadata.shape
train_metadata.chip_id.nunique()
flood_counts = train_metadata.groupby("flood_id")["chip_id"].nunique()
flood_counts.describe()

location_counts = (
    train_metadata.groupby("location")["chip_id"].nunique().sort_values(ascending=False)
)
plt.figure(figsize=(12, 4))
location_counts.plot(kind="bar", color="lightgray")
plt.xticks(rotation=45)
plt.xlabel("Location")
plt.ylabel("Number of Chips")
plt.title("Number of Chips by Location")

year = train_metadata.scene_start.dt.year
year_counts = train_metadata.groupby(year)["flood_id"].nunique()
year_counts

train_metadata.groupby("flood_id")["scene_start"].nunique()
