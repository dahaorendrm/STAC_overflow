"""
Fetching supplimentary model input from the Planetary Computer for STAC project
This notebook produces additional input layers for the training data used in the
sentinel 1 flood detection competition run by DrivenData.

If fetches JRC Global Surface Water and NASADEM elevation data from the Planetary
Computer (PC) STAC API and creates pixel-aligned chips that match what will be
used in the evaluation process for the competition.

The notebook will iterate through chip paths and query the PC STAC API for the
nasadem and jrc-gsw Collections. It then creates a set of GeoTiffs by
"coregistering" the raster data with the chip GeoTIFF, so that all of the
additional input layers have the same CRS, bounds, and resolution as the chip.
These additional layers are then saved alongside the training chip.
https://github.com/wonderdong11/PlanetaryComputerExamples/blob/main/competitions/s1floods/generate_auxiliary_input.ipynb

"""

from dataclasses import dataclass
import os
from tempfile import TemporaryDirectory
from typing import List, Any, Dict

from shapely.geometry import box, mapping
import rasterio
from rasterio.warp import reproject, Resampling
import pyproj
from osgeo import gdal

from pystac_client import Client
import planetary_computer as pc


# Define functions and classes
# Define a ChipInfo dataclass to encapsulate the required data for the target chip.
# This includes geospatial information that will be used to coregister the
# incoming jrc-gsw and nasadem data.
@dataclass
class ChipInfo:
    """
    Holds information about a training chip, including geospatial info for coregistration
    """

    path: str
    prefix: str
    crs: Any
    shape: List[int]
    transform: List[float]
    bounds: rasterio.coords.BoundingBox
    footprint: Dict[str, Any]


def get_footprint(bounds, crs):
    """Gets a GeoJSON footprint (in epsg:4326) from rasterio bounds and CRS"""
    transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
    minx, miny = transformer.transform(bounds.left, bounds.bottom)
    maxx, maxy = transformer.transform(bounds.right, bounds.top)
    return mapping(box(minx, miny, maxx, maxy))

def get_chip_info(chip_path):
    """Gets chip info from a GeoTIFF file"""
    with rasterio.open(chip_path) as ds:
        chip_crs = ds.crs
        chip_shape = ds.shape
        chip_transform = ds.transform
        chip_bounds = ds.bounds

    # Use the first part of the chip filename as a prefix
    prefix = os.path.basename(chip_path).split("_")[0]

    return ChipInfo(
        path=chip_path,
        prefix=prefix,
        crs=chip_crs,
        shape=chip_shape,
        transform=chip_transform,
        bounds=chip_bounds,
        footprint=get_footprint(chip_bounds, chip_crs),
    )


def reproject_to_chip(
    chip_info, input_path, output_path, resampling=Resampling.nearest
):
    """
    reprojects coregisters raster data to the bounds, CRS and resolution
    described by the ChipInfo.

    Reproject a raster at input_path to chip_info, saving to output_path.

    Use Resampling.nearest for classification rasters. Otherwise use something
    like Resampling.bilinear for continuous data.
    """
    with rasterio.open(input_path) as src:
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": chip_info.crs,
                "transform": chip_info.transform,
                "width": chip_info.shape[1],
                "height": chip_info.shape[0],
                "driver": "GTiff",
            }
        )

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=chip_info.transform,
                    dst_crs=chip_info.crs,
                    resampling=Resampling.nearest,
                )

# This method will take in a set of items and a asset key and write a VRT using
# signed HREFs. This is useful when there's multiple results from the query, so we
# can treat the resulting rasters as a single set of raster data. It uses the
# planetary_computer.sign method to sign the HREFs with a SAS token generated by
# the PC Data Auth API.
def write_vrt(items, asset_key, dest_path):
    """Write a VRT with hrefs extracted from a list of items for a specific asset."""
    hrefs = [pc.sign(item.assets[asset_key].href) for item in items]
    vsi_hrefs = [f"/vsicurl/{href}" for href in hrefs]
    gdal.BuildVRT(dest_path, vsi_hrefs).FlushCache()

# This method ties it all together - for a given ChipInfo, Collection, and Asset,
# write an auxiliary input chip with the given file name.
def create_chip_aux_file(
    chip_info, collection_id, asset_key, file_name, resampling=Resampling.nearest
):
    """
    Write an auxiliary chip file.

    The auxiliary chip file includes chip_info for the Collection and Asset, and is
    saved in the same directory as the original chip with the given file_name.
    """

    # Create the STAC API client¶
    # This will be used in the methods below to query the PC STAC API.
    STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = Client.open(STAC_API)

    output_path = os.path.join(
        os.path.dirname(chip_info.path), f"{chip_info.prefix}_{file_name}"
    )
    search = catalog.search(collections=[collection_id], intersects=chip_info.footprint)
    items = list(search.get_items())
    with TemporaryDirectory() as tmp_dir:
        vrt_path = os.path.join(tmp_dir, "source.vrt")
        write_vrt(items, asset_key, vrt_path)
        reproject_to_chip(chip_info, vrt_path, output_path, resampling=resampling)
    return output_path


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