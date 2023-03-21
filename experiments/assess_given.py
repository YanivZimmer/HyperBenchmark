import statistics
from typing import Callable
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import logging
from datetime import datetime
from experiments.assesment import Assesment
from hyper_data_loader.HyperDataLoader import HyperDataLoader
from models.cnn1_model import cnn_model
from tensorflow.keras.utils import to_categorical
import os
import tensorflow as tf

from models.mini_model_fn import mini_model1, mini_model2,mini_model3, mini_model4,mini_model5

logger_tf = tf.get_logger()
logger_tf.setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
    handlers=[
        # logging.FileHandler(filename),
        logging.StreamHandler()
    ],
)


class AssessGiven(Assesment):
    def __init__(
        self,
        model_creator: Callable[[int], Model],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        super().__init__(model_creator, X_train, y_train, X_test, y_test)

    def random_assess(self, min_bands, attempts_per_bands_amounts, *args, **kwargs):
        # expect second last dim of X_train to be number of bands
        best_score = {}
        average_score = {}
        selected = {}
        amount = self.X_train.shape[-2]
        while amount > min_bands:
            scores = []
            candidates = []
            for i in range(attempts_per_bands_amounts):
                bands = np.sort(
                    np.random.choice(
                        range(1, self.X_train.shape[-2] + 1), amount, replace=False
                    )
                )  # random.sample(range(1, self.X_train.shape[-2]+1), amount)
                candidates.append(bands)
                # bands-1 to norm bands index to match array index and start with 0
                model, results = self.assess_bands(bands - 1, *args, **kwargs)
                # results[1] is accuracy
                scores.append(results[1])
            best_score[amount] = max(scores)
            average_score[amount] = statistics.mean(scores)
            selected[amount] = candidates[np.argmax(scores)]
            logging.info(f"Average score for {amount}) is {average_score[amount]}")
            amount -= 1
        return selected, best_score, average_score


if __name__ == "__main__":
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = loader.filter_unlabeled(X, y)
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    try:
        asseser = AssessGiven(
            lambda x: mini_model3(x, 10), X_train, y_train, X_test, y_test
        )
        model, results = asseser.assess_bands([90, 17, 61],epochs=100)
        print(results)
    except Exception as e:
        print(str(e))
