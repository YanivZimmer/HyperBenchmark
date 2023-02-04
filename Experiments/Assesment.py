import logging
from typing import List, Callable, Dict
import numpy as np
from tensorflow.keras.models import Model

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

    def assess_bands(
        self, bands: List[int], *args,  **kwargs
    ) -> (Model, float):
        masked_x_train = self.X_train[...,bands,:]
        masked_x_test = self.X_test[...,bands,:]
        from _datetime import datetime
        logger.debug(datetime.now())
        model = self.model_creator(len(bands))

        history = model.fit(
            masked_x_train,
            self.y_train,
            *args,
            **kwargs
        )
        results = model.evaluate(masked_x_test, self.y_test, batch_size=256)
        logging.debug("Accuracy over test set is {0}".format(results))
        return model, results
