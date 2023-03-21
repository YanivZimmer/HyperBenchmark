import logging
import time
from typing import List, Callable, Dict
import numpy as np
from tensorflow.keras.models import Model
from gpu_utils.gpu_utils import pick_gpu_lowest_memory
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

logger_tf = tf.get_logger()
logger_tf.setLevel(logging.ERROR)


class Assesment:
    def __init__(
        self,
        model_creator: Callable[[int], Model],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        self.model_creator = model_creator
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def assess_bands(self, bands: List[int], *args, **kwargs) -> (Model, float):
        #masked_x_train = self.X_train[..., bands, :]
        #masked_x_test = self.X_test[..., bands, :]
        masked_x_train = self.X_train[..., bands]
        masked_x_test = self.X_test[..., bands]
        created = False
        max_try = 1
        tries = 0
        model = self.model_creator(len(bands))
        history = model.fit(
            masked_x_train,
            self.y_train,
            verbose=0,
            *args,
            **kwargs,
        )
        results = model.evaluate(masked_x_test, self.y_test, batch_size=256)
        logging.debug("Accuracy over test set is {0}".format(results))
        return model, results