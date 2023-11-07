import sys
#sys.path.append("/home/orange/Code/BarIlan/Thesis/HyperBenchmarklMerged/HyperBenchmark")
import json
from os.path import exists
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import tqdm
from hyper_data_loader.HyperDataLoader import HyperDataLoader
from sklearn.model_selection import train_test_split


import torch

from algorithms import ISSC, WALUDI, WALUMI, LP, MMCA

if __name__ == '__main__':
    hdl = HyperDataLoader()
    pavia = next(hdl.load_dataset_supervised("KSC", patch_shape=(1, 1)))
    lables = pavia.lables.reshape(pavia.lables.shape[0]*pavia.lables.shape[1])
    data = pavia.image.squeeze()
    data, data_test, lables, lables_test = train_test_split(data,lables,test_size=0.5,shuffle=True)
    algorithms = {
        'ISSC': ISSC,
        'MMCA': MMCA,
        'LP': LP,
        'WALUMI': WALUMI,
        'WALUDI': WALUDI
    }

    k = int(12)
    for algo_name, f in algorithms.items():
        print(f'Using {algo_name} for current iteration')
        model = f(k)

        model.fit(data)

        if algo_name == 'MMCA':
            _, bands = model.predict(data, lables, eps=0.4)
        else:
            _, bands = model.predict(data)

        print(f"{algo_name}=",sorted(bands))

