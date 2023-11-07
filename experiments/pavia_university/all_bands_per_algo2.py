import json
from os.path import exists
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import tqdm

from hyper_data_loader.HyperDataLoader import HyperDataLoader
from sklearn.model_selection import train_test_split

from models.deep_sets.data_loader import create_data_loader
from models.utils.train_test import train_model, simple_test_model

import torch

from algorithms import ISSC, WALUDI, WALUMI, LP, MMCA
from models.mlp.mlp import MlpModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

INPUT_SHAPE_PAVIA = 103
NUM_CLASSES_PAVIA = 10

INPUT_SHAPE_DRIVE = 25
NUM_CLASSES_DRIVE = 10

def filter_hafe_1(X, y):
    # idx = np.argsort(y)
    idx = np.where(y != 1)
    idx_too_much = np.where(y == 1)
    final=np.concatenate((idx[0],idx_too_much[0][int(len(idx_too_much[0])/4):int(len(idx_too_much[0])/2)]))
    y = y[final]
    X = X[final, :]
    return X, y


def data_loaders(bands):
    loader = HyperDataLoader()
    data = loader.generate_vectors("PaviaU", (1, 1), shuffle=True, limit=10)
    labeled_data = next(data)
    X, y = labeled_data.image, labeled_data.lables
    X, y = loader.filter_unlabeled(X, y)
    # X, y = filter_hafe_1(X, y)
    X = X.squeeze()
    X = X.astype(int)
    if bands is not None:
        X = X[:, bands]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = np.eye(NUM_CLASSES_PAVIA)[y_train]
    train_loader = create_data_loader(X_train, y_train, 256)
    test_loader = create_data_loader(X_test, y_test, 256)
    return train_loader,test_loader
    
def te1st_bands_mlp(bands):
    train_loader, test_loader = data_loaders(bands)
    mlp = MlpModel(len(bands), NUM_CLASSES_PAVIA)
    train_model(mlp, train_loader, epochs=100, lr=0.000025, device=device)
    return simple_test_model(mlp, test_loader, device=device)


def load_history(filepath: str) -> Dict[str, List]:
    if not exists(filepath):
        return defaultdict(list)

    with open(filepath, 'r') as f:
        d = json.load(f)

    return d


def save_history(res: Dict[str, Union[List, Dict]], filepath: str):
    with open(filepath, 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    DATASET_NAME = "Salinas"
    MIN_NUM_BANDS = 1  # if len(algs_benchmarks['MMCA']) == 0 else len(algs_benchmarks['MMCA'])
    MAX_NUM_BANDS = 60 #204
    TEST_MODEL = False
    hdl = HyperDataLoader()
    pavia = next(hdl.load_dataset_supervised(DATASET_NAME, patch_shape=(1, 1)))
    lables = pavia.lables
    data = pavia.image.squeeze()

    algorithms = {
        'WALUMI': WALUMI,
        'ISSC': ISSC,
        'MMCA': MMCA,
        'LP': LP,
        'WALUDI': WALUDI
    }

    history_filename = f'acc_results_{DATASET_NAME}.json'
    algo_bands_mapping_filename = f'algo_bands_mapping_results_temp_{DATASET_NAME}.json'
    #algs_benchmarks = load_history(history_filename)
    algo_bands_mapping = {}
    for algo in algorithms.keys():
        algo_bands_mapping[algo] = {}


    for i in tqdm.trange(MIN_NUM_BANDS + 1, MAX_NUM_BANDS + 1, initial=MIN_NUM_BANDS, total=MAX_NUM_BANDS):
        for algo_name, f in algorithms.items():
            print(f'Using {algo_name} for current iteration {i}')
            model = f(i)
            data2 = data/np.amax(data)
            data2 = 256 * data2
            data = data2
            model.fit(data)

            if algo_name == 'MMCA':
                _, bands = model.predict(data, lables, eps=0.4)
            else:
                #9207 amax salinas
                _, bands = model.predict(data)
            algo_bands_mapping[algo_name][i] = sorted(bands.tolist())
            if TEST_MODEL:
                acc = test_bands_mlp(bands)
                algs_benchmarks[algo_name].append(acc)

        #save_history(algs_benchmarks, history_filename)
    if TEST_MODEL:
        for algo_name, accs in algs_benchmarks.items():
            plt.plot(range(MAX_NUM_BANDS), accs, label=algo_name)
        plt.legend()
        plt.savefig('benchmark-algorithms results.png')
        plt.show()
        # acc=test_bands_mlp(range(1,103))
        # print(acc)
    print(algo_bands_mapping)
    save_history(algo_bands_mapping, algo_bands_mapping_filename)
