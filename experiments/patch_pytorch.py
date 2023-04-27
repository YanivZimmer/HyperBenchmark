import logging

import numpy as np
import torch
from pytorch_assessment import PytorchAssesment
from sklearn.model_selection import train_test_split
from torch import Tensor

from hyper_data_loader.HyperDataLoader import HyperDataLoader

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
    handlers=[
        # logging.FileHandler(filename),
        logging.StreamHandler()
    ],
)
import models.pytorch_models.models

N_CLASS=10
def get_model(n_bands,**kwargs):
    return models.pytorch_models.models.get_model("li",n_bands=n_bands,n_classes=N_CLASS,ignored_labels=(),
                                                  patch_size=5,
                                                  learning_rate=0.0000000005,**kwargs)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    NUM_ClASSES=10
    logging.info("Start")
    NUM_OF_CLASSES = 10
    loader = HyperDataLoader()
    labeled_data = loader.load_dataset_supervised("PaviaU", (5, 5))
    X, y = labeled_data[0].image, labeled_data[0].lables
    y = y.reshape(y.shape[0]* y.shape[1])
    X, y = loader.filter_unlabeled(X, y)
    y = np.eye(10)[y]#.astype(np.)
    X=X.astype(np.float32)
    #X=Tensor(X,dtype=torch.float32)
    #X=Tensor(X,device=device)#,dtype=torch.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle=True)
    #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    pa = PytorchAssesment(get_model, X_train, y_train, X_test, y_test)
    pa.assess_bands(list(range(1, 104)),epochs=2,batch_size=32)
