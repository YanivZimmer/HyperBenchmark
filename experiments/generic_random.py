import numpy as np
from sklearn.model_selection import train_test_split

from hyper_data_loader.HyperDataLoader import HyperDataLoader
from pytorch_assessment import PytorchAssesment
import logging
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
def get_model(n_bands):
    return models.pytorch_models.models.get_model("hu",n_bands=n_bands,n_classes=N_CLASS,ignored_labels=())

class GenericRandomSearch:
    def __init__(
        self,
        model_tester
    ):
        pass



if __name__ == "__main__":
    NUM_ClASSES=10
    logging.info("Start")
    NUM_OF_CLASSES = 10
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = loader.filter_unlabeled(X, y)
    y = np.eye(10)[y]#.astype(np.)
    X=X.astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    pa = PytorchAssesment(get_model, X_train, y_train, X_test, y_test)
    pa.assess_bands(list(range(1, 100)))

