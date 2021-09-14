import torch
import rasterio
import numpy as np
import os
from pystac_client import Client
from fetch_additional_data import *


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
        try:
            rasterio.open(img.vv_path)
        except:
            rasterio.open(img.vv_path)
        with rasterio.open(img.vv_path) as vv:
            vv_path = vv.read(1)
        try:
            rasterio.open(img.vh_path)
        except:
            rasterio.open(img.vh_path)
        with rasterio.open(img.vh_path) as vh:
            vh_path = vh.read(1)
        x_arr = np.stack([vv_path, vh_path], axis=-1)

        # Min-max normalization
        min_norm = -77
        max_norm = 26
        x_arr = np.clip(x_arr, min_norm, max_norm)
        x_arr = (x_arr - min_norm) / (max_norm - min_norm)

        # Apply data augmentations, if provided
        if self.transforms:
            x_arr = self.transforms(image=x_arr)["image"]
        x_arr = np.transpose(x_arr, [2, 0, 1])

        # Prepare sample dictionary
        sample = {"chip_id": img.chip_id, "chip": x_arr}
        #################################################
        #################################################
        ## Dong please add you code hear to introduce supplimentary data in the sample !!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!Extract training chips
        # Download the flood-train-images.tgz file from competition Data Download page and
        # upload it to the Hub in the same directory as this notebook.

        # Then run the following code to uncompress this. Afterwards you should see an
        # train_features directory containing all of the training chips ending in .tif.
        # !tar -xvf flood-train-images.tgz

        # Use this directory to define the location of the chips, or if you have
        # already uncompressed the chips elsewhere set the location here:
        TRAINING_DATA_DIR = "training_data/train_features"

        # Gather chip paths
        # These chip paths will be used later in the notebook to process the chips.
        # These paths should be to only one GeoTIFF per chip; for example, if both
        # VV.tif and VH.tif are available for a chip, use only one of these paths.
        # The GeoTIFFs at these paths will be read to get the bounds, CRS and resolution
        # that will be used to fetch auxiliary input data. These can be relative paths.
        # The auxiliary input data will be saved in the same directory as the GeoTIFF
        # files at these paths.
        chip_paths = []
        for file_name in os.listdir(TRAINING_DATA_DIR):
            if file_name.endswith("_vv.tif"):
                chip_paths.append(os.path.join(TRAINING_DATA_DIR, file_name))
        print(f"{len(chip_paths)} chips found.")

        # Create the STAC API client¶
        # This will be used in the methods below to query the PC STAC API.
        STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
        catalog = Client.open(STAC_API)

        # Configurate the auxiliary input files that we will generate.
        # Define a set of parameters to pass into create_chip_aux_file
        aux_file_params = [
            ("nasadem", "elevation", "nasadem.tif", Resampling.bilinear),
            ("jrc-gsw", "extent", "jrc-gsw-extent.tif", Resampling.nearest),
            ("jrc-gsw", "occurrence", "jrc-gsw-occurrence.tif", Resampling.nearest),
            ("jrc-gsw", "recurrence", "jrc-gsw-recurrence.tif", Resampling.nearest),
            ("jrc-gsw", "seasonality", "jrc-gsw-seasonality.tif", Resampling.nearest),
            ("jrc-gsw", "transitions", "jrc-gsw-transitions.tif", Resampling.nearest),
            ("jrc-gsw", "change", "jrc-gsw-change.tif", Resampling.nearest),
        ]

        # Generate auxiliary input chips for NASADEM and JRC
        # Iterate over the chips and generate all aux input files.
        count = len(chip_paths)
        for i, chip_path in enumerate(chip_paths):
            print(f"({i+1} of {count}) {chip_path}")
            chip_info = get_chip_info(chip_path)
            for collection_id, asset_key, file_name, resampling_method in aux_file_params:
                print(f"  ... Creating chip data for {collection_id} {asset_key}")
                create_chip_aux_file(
                    chip_info, collection_id, asset_key, file_name, resampling=resampling_method
                )

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
