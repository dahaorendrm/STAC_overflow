import torch
import rasterio
import numpy as np
import os


class FloodDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self, x_paths, y_paths=None, transforms=None):
        self.data = x_paths
        self.label = y_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Loads a 2-channel image from a chip-level dataframe
        img = self.data.loc[idx]
        with rasterio.open(img.vv_path) as vv:
            vv_img = vv.read(1)
        with rasterio.open(img.vh_path) as vh:
            vh_img = vh.read(1)
        name_path = img.vh_path[:-6]
        with rasterio.open(name_path+'nasadem.tif') as nasadem:
            nasadem_img = nasadem.read(1)
        with rasterio.open(name_path+'jrc-gsw-extent.tif') as extent:
            extent_img = extent.read(1)
        with rasterio.open(name_path+'jrc-gsw-occurrence.tif') as occurrence:
            occurrence_img = occurrence.read(1)
        with rasterio.open(name_path+'jrc-gsw-recurrence.tif') as recurrence:
            recurrence_img = recurrence.read(1)
        with rasterio.open(name_path+'jrc-gsw-seasonality.tif') as seasonality:
            seasonality_img = seasonality.read(1)
        with rasterio.open(name_path+'jrc-gsw-transitions.tif') as transitions:
            transitions_img = transitions.read(1)
        with rasterio.open(name_path+'jrc-gsw-change.tif') as change:
            change_img = change.read(1)
        x_arr = np.stack([vv_img, vh_img], axis=-1)
        # Min-max normalization
        min_norm = -77 #-79
        max_norm = 26 #28
        x_arr = np.clip(x_arr, min_norm, max_norm)
        x_arr = (x_arr - min_norm) / (max_norm - min_norm)
        min_norm = -64
        max_norm = 2096
        nasadem_img = np.clip(nasadem_img, min_norm, max_norm)
        nasadem_img = (nasadem_img - min_norm) / (max_norm - min_norm)
        min_norm = 0
        max_norm = 255
        extent_img = np.clip(extent_img, min_norm, max_norm)
        extent_img = (extent_img - min_norm) / (max_norm - min_norm)
        occurrence_img = np.clip(occurrence_img, min_norm, max_norm)
        occurrence_img = (occurrence_img - min_norm) / (max_norm - min_norm)
        recurrence_img = np.clip(recurrence_img, min_norm, max_norm)
        recurrence_img = (recurrence_img - min_norm) / (max_norm - min_norm)
        seasonality_img = np.clip(seasonality_img, min_norm, max_norm)
        seasonality_img = (seasonality_img - min_norm) / (max_norm - min_norm)
        transitions_img = np.clip(transitions_img, min_norm, max_norm)
        transitions_img = (transitions_img - min_norm) / (max_norm - min_norm)
        change_img = np.clip(change_img, min_norm, max_norm)
        change_img = (change_img - min_norm) / (max_norm - min_norm)

        #######################adjust supp data range
        # Apply data augmentations, if provided
        if self.transforms:
            x_arr = self.transforms(image=x_arr)["image"]
            nasadem_img = self.transforms(image=nasadem_img)["image"]
            extent_img = self.transforms(image=extent_img)["image"]
            occurrence_img = self.transforms(image=occurrence_img)["image"]
            recurrence_img = self.transforms(image=recurrence_img)["image"]
            seasonality_img = self.transforms(image=seasonality_img)["image"]
            transitions_img = self.transforms(image=transitions_img)["image"]
            change_img = self.transforms(image=change_img)["image"]

            #####################add supplementary
        x_arr = np.transpose(x_arr, [2, 0, 1])

        # Prepare sample dictionary
        sample = {"chip_id": img.chip_id, "chip": x_arr, "nasadem":nasadem_img, "extent":extent_img, "occurrence":occurrence_img,
                        "recurrence":recurrence_img, "seasonality":seasonality_img, "transitions":transitions_img, "change":change_img}

        # Load label if available - training only
        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1)
            # Apply same data augmentations to label
            if self.transforms:
                y_arr = self.transforms(image=y_arr)["image"]
            sample["label"] = y_arr

        return sample

if __name__ == '__main__':
    import pandas as pd
    import utils
    import random
    from pathlib import Path
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import torch
    import os
    DATA_PATH = Path("training_data")
    train_metadata = pd.read_csv(
        DATA_PATH / "flood-training-metadata.csv", parse_dates=["scene_start"]
    )
    # Sample 3 random floods for validation set
    flood_ids = train_metadata.flood_id.unique().tolist()
    val_flood_ids = random.sample(flood_ids, 3)
    val = train_metadata[train_metadata.flood_id.isin(val_flood_ids)]
    train = train_metadata[~train_metadata.flood_id.isin(val_flood_ids)]
    train_x = utils.get_paths_by_chip(train)
    train_y = train[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)

    train_dataset = FloodDataset(
        train_x, train_y
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    sample = next(iter(train_dataloader))
    key_list = ["chip", "nasadem", "extent", "occurrence","recurrence",
    "seasonality", "transitions", "change", 'label']
    if not os.path.isdir('temp'):
        os.mkdir('temp')
    for key,val in sample.items():
        if key in key_list:
            val = torch.squeeze(val)
            print(f'key:{}, max value = {np.amax(val)}')
            if key is 'chip':
                img = torch.moveaxis(val,0,-1)
                img = utils.create_false_color_composite(img.numpy())
                #print(img.shape)
                plt.imsave('temp/'+key+'.png',img)
                #plt.imshow(img)
                #plt.show()
            else:
                plt.imsave('temp/'+key+'.png',val.numpy())
