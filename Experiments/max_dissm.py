from typing import Callable, Dict

import numpy as np
from keras import Model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from Experiments.Assesment import Assesment
from Experiments.RandomSearch import logger
from hyper_data_loader.HyperDataLoader import HyperDataLoader
from models.cnn1_model import cnn_model
from selection.max_dissim_selector import MaxDissimSelector


class MaxDissm(Assesment):
    def __init__(
        self,
        model_creator: Callable[[int], Model],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        super().__init__(model_creator, X_train, y_train, X_test, y_test)
        candidates = self.generate_candidates()
        vec_size = next(iter(candidates.values())).shape[0]
        self.max_dissm_selector = MaxDissimSelector(vec_size,candidates)

    def generate_candidates(self) -> Dict[int, np.ndarray]:
        candidates = {}
        for band_idx in range(self.X_train.shape[1]):
            candidates[band_idx] = self.X_train[:, band_idx].flatten()
        return candidates


if __name__=='__main__':
    if __name__ == "__main__":
        logger.info("Start max dismm")
        NUM_OF_CLASSES = 10
        loader = HyperDataLoader()
        labeled_data = loader.generate_vectors("PaviaU", (1, 1))
        X, y = labeled_data[0].image, labeled_data[0].lables
        X, y = loader.filter_unlabeled(X, y)
        y = to_categorical(y, num_classes=10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        searcher = MaxDissm(
            lambda x: cnn_model(x, NUM_OF_CLASSES), X_train, y_train, X_test, y_test
        )

