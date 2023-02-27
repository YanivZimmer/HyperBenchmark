import logging
import time
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
        created=False
        while not created:
            try:
                model = self.model_creator(len(bands))
                print("created!!!!!!!!!!!!!")
                created = True
            except Exception as e:
                logging.error(f"failed to create model because: {str(e)} ")
                SLEEP_TIME_MIN = 2
                logging.info(f"Sleep for {SLEEP_TIME_MIN} min")
                time.sleep(60*SLEEP_TIME_MIN)

        history = model.fit(
            masked_x_train,
            self.y_train,
            verbose=0,
            *args,
            **kwargs
        )
        results = model.evaluate(masked_x_test, self.y_test, batch_size=256)
        logging.debug("Accuracy over test set is {0}".format(results))
        return model, results
