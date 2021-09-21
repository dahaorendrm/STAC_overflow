import os
import numpy as np
from utils import *
import rasterio


def find_max_min(path, name):
    data_list = os.listdir(path)
    li_max = []
    li_min = []
    for data in data_list:
        if name in data:
            with rasterio.open(data) as f:
                img = f.read(1)
            li_max.append(np.amax(img))
            li_min.append(np.amin(img))

    return li_max,li_min


if __name__ == '__main__':
    data_name = 'nasadem'
    li_max,li_min = find_max_min('/training_data/train_features', data_name)
    print(f'{data_name} max:{np.amax(li_max)},  min:{np.amin(li_min)}')
