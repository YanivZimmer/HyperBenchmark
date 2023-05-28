import numpy as np

from hyper_data_loader.HyperDataLoader import HyperDataLoader
from sklearn.model_selection import train_test_split

from models.utils.data_loader import create_data_loader
from models.utils.train_test import train_model, simple_test_model

import torch

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
    
def test_bands_mlp(bands):
    train_loader, test_loader = data_loaders(bands)
    mlp = MlpModel(len(bands), NUM_CLASSES_PAVIA)
    train_model(mlp, train_loader, epochs=100, lr=0.000025, device=device)
    return simple_test_model(mlp, test_loader, device=device)

if __name__ == '__main__':
    acc=test_bands_mlp(range(1,103))
    print(acc)

