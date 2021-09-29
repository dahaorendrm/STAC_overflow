import numpy as np
import pytorch_lightning as pl
import rasterio
import segmentation_models_pytorch as smp
import torch


class FloodModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        cnn_denoise = torch.nn.Sequential(
            torch.nn.Conv2d(9, 9, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU())
        #torch.nn.init.normal_(cnn_denoise[0].weight.data, mean=0.0, std=1.0)
        unet_model = smp.Unet(
            encoder_name="nceptionv4",
            encoder_weights=None,
            in_channels=9,
            classes=2,
        )
        self.model = torch.nn.Sequential(cnn_denoise, unet_model)

    def forward(self, image):
        # Forward pass
        return self.model(image)

    def predict(self, data_path, chip_id):
        # Switch on evaluation mode
        self.model.eval()
        torch.set_grad_enabled(False)

        # Create a 2-channel image
        vv_path = data_path / "test_features" / f"{chip_id}_vv.tif"
        vh_path = data_path / "test_features" / f"{chip_id}_vh.tif"
        with rasterio.open(vv_path) as vv:
            vv_img = vv.read(1)
        with rasterio.open(vh_path) as vh:
            vh_img = vh.read(1)
        temp_path = data_path / "nasadem" / f"{chip_id}.tif"
        with rasterio.open(temp_path) as nasadem:
            nasadem_img = nasadem.read(1)
        temp_path = data_path / "jrc_extent" / f"{chip_id}.tif"
        with rasterio.open(temp_path) as extent:
            extent_img = extent.read(1)
        temp_path = data_path / "jrc_occurrence" / f"{chip_id}.tif"
        with rasterio.open(temp_path) as occurrence:
            occurrence_img = occurrence.read(1)
        temp_path = data_path / "jrc_recurrence" / f"{chip_id}.tif"
        with rasterio.open(temp_path) as recurrence:
            recurrence_img = recurrence.read(1)
        temp_path = data_path / "jrc_seasonality" / f"{chip_id}.tif"
        with rasterio.open(temp_path) as seasonality:
            seasonality_img = seasonality.read(1)
        temp_path = data_path / "jrc_transitions" / f"{chip_id}.tif"
        with rasterio.open(temp_path) as transitions:
            transitions_img = transitions.read(1)
        temp_path = data_path / "jrc_change" / f"{chip_id}.tif"
        with rasterio.open(temp_path) as change:
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

        # Transpose
        x_arr = np.transpose(x_arr, [2, 0, 1])
        x_arr = np.expand_dims(x_arr, axis=0)
        nasadem_img = np.expand_dims(np.expand_dims(nasadem_img,0),0)
        extent_img = np.expand_dims(np.expand_dims(extent_img,0),0)
        occurrence_img = np.expand_dims(np.expand_dims(occurrence_img,0),0)
        recurrence_img = np.expand_dims(np.expand_dims(recurrence_img,0),0)
        seasonality_img = np.expand_dims(np.expand_dims(seasonality_img,0),0)
        transitions_img = np.expand_dims(np.expand_dims(transitions_img,0),0)
        change_img = np.expand_dims(np.expand_dims(change_img,0),0)
        # Perform inference
        x = (x_arr, nasadem_img, extent_img, occurrence_img, recurrence_img, seasonality_img, transitions_img, change_img)
        #Error()
        x = np.concatenate(x,1)
        preds = self.forward(torch.from_numpy(x).float())
        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > 0) * 1
        return preds.detach().numpy().squeeze().squeeze()
