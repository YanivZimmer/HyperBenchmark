import logging
import time
from typing import List, Callable, Dict
import numpy as np
from tensorflow.keras.models import Model
from gpu_utils.gpu_utils import pick_gpu_lowest_memory
import tensorflow
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
        device_num_org=pick_gpu_lowest_memory()
        device_num=device_num_org
        while not created:
            try:
                full_gpu_name=f'/gpu:{device_num}'
                with tensorflow.device(full_gpu_name):
                  model = self.model_creator(len(bands))
                  history = model.fit(masked_x_train,self.y_train,verbose=0,*args,**kwargs,)
                  results = model.evaluate(masked_x_test, self.y_test, batch_size=256)
                  logging.debug("Accuracy over test set is {0}".format(results))
                  return model, results
                  created = True
            except Exception as e:
                msg=f"failed to create\train model because: {str(e)} "
                print(msg)
                logging.error(msg)
                if max_try < tries:
                    break
                tries += 1
                SLEEP_TIME_MIN = 0.1
                logging.info(f"Sleeping for {SLEEP_TIME_MIN} min")
                time.sleep(60 * SLEEP_TIME_MIN)
                device_num += tries
                msg2=f"try again with device {str(device_num)}"
                print(msg2)
                logging.info(msg2)
        results = model.evaluate(masked_x_test, self.y_test, batch_size=256)
        logging.debug("Accuracy over test set is {0}".format(results))
        return model, results
