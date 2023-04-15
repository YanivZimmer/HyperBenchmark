import numpy as np
from torch import Tensor

from experiments.pytorch_assessment import PytorchAssesment
from hyper_data_loader.HyperDataLoader import HyperDataLoader
from sklearn.model_selection import train_test_split

import models.pytorch_models.models
from models.deep_sets.data_loader import create_data_loader
from models.deep_sets.deep_sets import train_model, DeepSets, test_model, simple_test_model
from models.deep_sets.deep_gpt import MyModel
import torch
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


if __name__=='__main__':

    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (3, 3))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = loader.filter_unlabeled(X, y)
    #X, y = filter_hafe_1(X, y)
    X = X.astype(int)
    #X=X[:,:,:,range(1,102,5)]
    X = X.reshape(X.shape[0],X.shape[3],1,X.shape[1],X.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    y_train = np.eye(NUM_CLASSES_PAVIA)[y_train]
    #X_train = X_train.reshape(X_train.shape[0],X_train.shape[3],1,X_train.shape[1],X_train.shape[2])

    deep_sets = DeepSets(1, 10, 30, NUM_CLASSES_PAVIA)
    train_loader = create_data_loader(X_train, y_train, 256)
    test_loader = create_data_loader(X_test,y_test,64)
    train_model(deep_sets,train_loader,epochs=150,lr=0.00001,device=device)
    simple_test_model(deep_sets,test_loader,device=device)
