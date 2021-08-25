# This file is base on the instruction of stac api.
# In this project it's used to access supplimentary data 
#      * elevation data from the NASA Digital Elevation Model (NASADEM)
#      * global surface water data from the European Commission's Joint Research Centre (JRC), including map layers for seasonality, occurrence, change, recurrence, transitions, and extent
# ref: https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/

## S1
from pystac_client import Client

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

## S2
collections = catalog.get_children()
for collection in collections:
    print(f"{collection.id} - {collection.title}")

## S3
landsat = catalog.get_child("landsat-8-c2-l2")
for band in landsat.extra_fields["summaries"]["eo:bands"]:
    name = band["name"]
    description = band["description"]
    common_name = "" if "common_name" not in band else f"({band['common_name']})"
    ground_sample_distance = band["gsd"]
    print(f"{name} {common_name}: {description} ({ground_sample_distance}m resolution)")

## S4
area_of_interest = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.27508544921875, 47.54687159892238],
            [-121.96128845214844, 47.54687159892238],
            [-121.96128845214844, 47.745787772920934],
            [-122.27508544921875, 47.745787772920934],
            [-122.27508544921875, 47.54687159892238],
        ]
    ],
}

time_range = "2020-12-01/2020-12-31"

search = catalog.search(
    collections=["landsat-8-c2-l2"], intersects=area_of_interest, datetime=time_range
)

## S5
items = list(search.get_items())
for item in items:
    print(f"{item.id}: {item.datetime}")

## S6
selected_item = sorted(items, key=lambda item: item.properties["eo:cloud_cover"])[0]

## S7
for asset_key, asset in selected_item.assets.items():
    print(f"{asset_key:<25} - {asset.title}")

## S8
import json

thumbnail_asset = selected_item.assets["thumbnail"]
print(json.dumps(thumbnail_asset.to_dict(), indent=2))

## S9
import requests

requests.get(thumbnail_asset.href)

## S10
import planetary_computer as pc

signed_href = pc.sign(thumbnail_asset.href)

## S11
from PIL import Image
from urllib.request import urlopen

Image.open(urlopen(signed_href))




