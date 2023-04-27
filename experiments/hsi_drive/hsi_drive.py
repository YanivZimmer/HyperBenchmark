import itertools

import numpy as np
from torch import Tensor

from experiments.pytorch_assessment import PytorchAssesment
from hyper_data_loader.HyperDataLoader import HyperDataLoader, Labeled_Data
from sklearn.model_selection import train_test_split

import models.pytorch_models.models
from models.deep_sets.data_loader import create_data_loader
from models.deep_sets.deep_sets import train_model, DeepSets, simple_test_model
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

INPUT_SHAPE_PAVIA = 103
NUM_CLASSES_PAVIA = 10

INPUT_SHAPE_DRIVE = 25
NUM_CLASSES_DRIVE = 10

def filter_unlablled(X, y):
    # idx = np.argsort(y)
    idx = np.where(y != 0)
    y = y[idx[0]]
    X = X[idx[0], :]
    # Make lables 0-9
    y -= 1
    return X, y


def combine_hsi_drive(test_size=0.33):
    LIMIT=19
    loader = HyperDataLoader()
    labeled_data = list(loader.generate_vectors("HSI-drive", patch_shape=(3, 3),limit=LIMIT))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X = X.reshape(X.shape[0], X.shape[3], X.shape[1], X.shape[2])
    for item in labeled_data[1:LIMIT+1]:
        X = np.concatenate((X, item.image.reshape(item.image.shape[0],item.image.shape[3],
                                                  item.image.shape[1],item.image.shape[2])))
        y = np.concatenate((y, item.lables))
    X, y = filter_unlablled(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True
    )
    y_train = np.eye(NUM_CLASSES_DRIVE)[y_train]
    return X_train, X_test, y_train, y_test

def train_iter():
    LIMIT=1000
    loader = HyperDataLoader()
    for labeled_data_iter in loader.generate_vectors("HSI-drive", patch_shape=(3, 3),shuffle=True,limit=LIMIT):
        X, y = labeled_data_iter.image, labeled_data_iter.lables
        X = X.reshape(X.shape[0], X.shape[3], X.shape[1], X.shape[2])
        X, y = filter_unlablled(X, y)
        #y_train = np.eye(NUM_CLASSES_DRIVE)[y_train]
        yield Labeled_Data(X,y)

def get_model(n_bands,**kwargs):
    return models.pytorch_models.models.get_model("lee",n_bands=n_bands,n_classes=NUM_CLASSES_PAVIA,patch_size=3,ignored_labels=(),
                                                 **kwargs)# learning_rate=0.00006

def main_sync():
    X_train, X_test, y_train, y_test = combine_hsi_drive(0.1)
    deep_sets = DeepSets(1, 25, 45, NUM_CLASSES_DRIVE)
    train_loader = create_data_loader(X_train, y_train, 256, )
    test_loader = create_data_loader(X_test, y_test, 256)
    train_model(deep_sets, train_loader, epochs=3, lr=0.00005, device=device)
    simple_test_model(deep_sets, test_loader, device=device)

def main_iter_deep_sets(limit):
    deep_sets = DeepSets(1, 25, 45, NUM_CLASSES_DRIVE)
    iter=train_iter()
    for index, item in zip(range(limit), iter):
        print(index, item.image.shape,item.image[10][10][0])
        y = np.eye(NUM_CLASSES_DRIVE)[item.lables]
        train_loader = create_data_loader(item.image, y, 256)
        train_model(deep_sets, train_loader, epochs=1, lr=0.00001, device=device)
    acc_list=[]
    for i in range(20):
        values = next(iter)
        test_loader = create_data_loader(values.image, values.lables, 256)
        acc=simple_test_model(deep_sets, test_loader, device=device)
        acc_list.append(acc)
    print(f"Average acc on {len(acc_list)} images: {sum(acc_list)/len(acc_list)}")

if __name__=='__main__':
    main_iter(210)
