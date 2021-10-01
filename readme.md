# STAC overflow

A solution of the competition hosted by [drivendata.org](drivendata.org). My solution rank 23/663. There are three branches in the solution, the master is the one I summited for the competition, the [base_line](https://www.drivendata.co/blog/detect-floodwater-benchmark/) is modified the one offered by Microsoft, and the multi_cnn is the one I think should outperform than the master if I have time to keep tune it. 

## Requirement

  

1. python 3.7
2. pytorch 1.9 with cuda 10.2
3. pytorch-lightning 1.4.5 
4. segmentation-models-pytorch 0.2.0
5. rasterio 1.2.6
6. pandas 1.2.5
7. albumentations 1.0.3



## How to run the code

### Data

1. Download data to the folder <training_data>
2. Run "fetch_additional_data.py" to fetch all supplementary data in the training data folder

### Training

Run stac_train.py to train the model. The trained model weights and optimizer weights are saved in <model-outputs>.

### Test

1. Modify and run "submit.py", the unet architecture will by saved on in your device's cache. Then the test environment will be set up.
2. Copy the test data in <submit-pytorch> folder.
3. Run "main.py" in <submit-pytorch> folder.

