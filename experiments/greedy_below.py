import logging
import multiprocessing
import os

from models.mini_model_fn import mini_model3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
from datetime import datetime
from multiprocessing import Pool
from gpu_utils.gpu_utils import pick_gpu_lowest_memory
import tensorflow.python

# from pathos.multiprocessing import ProcessingPool as Pool
from assesment import Assesment
import tensorflow as tf

logger_tf = tf.get_logger()
logger_tf.setLevel(logging.ERROR)
now = datetime.now()
date_time = now.strftime("%m%d%Y_%H%M")
print(os.getcwd())
filename = f"../logs/Hyperspectral_extensive_{date_time}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler(filename),
        logging.StreamHandler()
    ],
)
logging.info("hi")
# logger = logging.getLogger()
# logger.info("ya")
import statistics
from typing import List, Callable, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from hyper_data_loader import HyperDataLoader
from models.cnn1_model import cnn_model
from tensorflow.keras.utils import to_categorical


class GreedyBelow(Assesment):
    def __init__(
        self,
        model_creator: Callable[[int], Model],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        super().__init__(model_creator, X_train, y_train, X_test, y_test)


    def search_with_one_new(
        self, existing_bands: List[int], optinal_bands: List[int], *args, **kwargs
    ) -> Dict[int, float]:
        """

        :param all_bands:
        :return: Dict of missing band and model score without him
        """
        new_band_to_score = {}
        for band in optinal_bands:
            bands_plus_candidate = list(existing_bands)
            bands_plus_candidate.append(band)
            _, [loss, score] = self.assess_bands(bands_plus_candidate, *args, **kwargs)
            new_band_to_score[band] = score
        return new_band_to_score
    def find_best(self, results: Dict[int, float]) -> int:
        """

        """

        return max(results, key=results.get)

    def search_all(
        self, all_bands: List[int], min_bands, *args, **kwargs
    ):
        remain = list(all_bands)
        best_score = {}
        average_score = {}
        chosen = []
        while len(remain) > min_bands:

            if len(remain) % 1 == 0:
                print(f"Checkpoint on {len(remain)} bands remained")
                logging.info(f"IMPORTANT best_score:={best_score}")
                print(best_score)
                logging.info(f"IMPORTANT average_score:={average_score}")
                print(average_score)

            results = self.search_with_one_new(chosen,remain, *args, **kwargs)
            best_band = self.find_best(results)
            chosen.append(best_band)
            score = results[best_band]
            logging.info(
                f"Best band was {best_band}, testing bands list of size {len(chosen)}."
            )
            best_score[len(chosen)] = score
            average_score[len(chosen)] = statistics.mean(results.values())
            logging.info(
                f"Average score for {len(chosen)} = {average_score[len(chosen)]}"
            )
            remain.remove(best_band)
        return chosen, best_score, average_score


if __name__ == "__main__":
    logging.info("Start")

    NUM_OF_CLASSES = 10
    loader = HyperDataLoader.HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = loader.filter_unlabeled(X, y)
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    searcher = GreedyBelow(
        lambda x: mini_model3(x, NUM_OF_CLASSES), X_train, y_train, X_test, y_test
    )
    temp = list(np.arange(1, 103, 1))
    print(temp)
    removed, best_score, average_score = searcher.search_all(
        temp, min_bands=0, epochs=75
    )
    logging.info(f"IMPORTANT FINAL best_score:={best_score}")
    print(best_score)
    logging.info(f"IMPORTANT FINAL removed:={removed}")
    print(removed)
    logging.info(f"IMPORTANT FINAL average_score:={average_score}")
    print(average_score)
