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

    def assess_bands(self, bands: List[int], *args, **kwargs) -> (Model, float):
        masked_x_train = self.X_train[..., bands, :]
        masked_x_test = self.X_test[..., bands, :]
        created = False
        max_try = 6
        tries = 0
        while not created:
            try:
                model = self.model_creator(len(bands))
                created = True
            except Exception as e:
                msg=f"failed to create model because: {str(e)} "
                print(msg)
                logging.error(f"failed to create model because: {str(e)} ")
                if max_try < tries:
                    break
                tries += 1
                SLEEP_TIME_MIN = 0.1
                logging.info(f"Sleeping for {SLEEP_TIME_MIN} min")
                time.sleep(60 * SLEEP_TIME_MIN)
        trained = False
        max_try = 6
        tries = 0
        while not trained:
            try:
                history = model.fit(
                    masked_x_train,
                    self.y_train,
                    verbose=0,
                    *args,
                    **kwargs,
                )
                trained = True
            except Exception as e:
                msg=f"failed to train model because: {str(e)}"
                print(msg)
                logging.error(msg)
                if max_try<tries:
                    break
                SLEEP_TIME_MIN = 0.1
                logging.info(f"Sleeping for {SLEEP_TIME_MIN} min")
                time.sleep(60 * SLEEP_TIME_MIN)
                tries += 1
        results = model.evaluate(masked_x_test, self.y_test, batch_size=256)
        logging.debug("Accuracy over test set is {0}".format(results))
        return model, results
