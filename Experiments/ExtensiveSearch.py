import statistics
from typing import List, Callable, Tuple, Dict
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import logging

class Searcher:
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
        self, bands: List[int], epochs=10, batch_size=256, verbose=1, **kwargs
    ) -> (Model, float):
        model = self.model_creator(len(bands))
        masked_x_train = self.X_train[::bands]
        masked_x_test = self.X_test[::bands]
        history = model.fit(
            masked_x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs
        )
        results = model.evaluate(masked_x_test, self.y_test, batch_size=256)
        print("Accuracy over test set is {0}".format(results))
        return model, results

    def search_without_one(self, all_bands: List[int]) -> Dict[int, float]:
        '''

        :param all_bands:
        :return: Dict of missing band and model score without him
        '''

        missing_band_to_score = {}
        for band in all_bands:
            almost_all_bands = list(all_bands)
            all_bands.remove(band)
            _, score = self.assess_bands(almost_all_bands)
            missing_band_to_score[band] = score
        return missing_band_to_score
    def find_worst(self,results:Dict[int,float])->int:
        '''

        :return: Return band for which without him the score was highest (this implies he is of less importance )
        '''

        return max(results, key=results.get)
    def search_all(self,all_bands: List[int]):
        bands = list(all_bands)
        best_score = {}
        average_score = {}
        removed=[]
        while len(bands) > 10:
            results = self.search_without_one(bands)
            worst_band=self.find_worst(results)
            removed.append(worst_band)
            score=results[worst_band]
            logging.info(f"Words band was {len(bands)}, testing bands list of size {worst_band}. score without him={score}")
            best_score[len(bands)] = score
            average_score[len(bands)] = statistics.mean(results.values())
            logging.info(f"Average score for {len(bands) is average_score[len(bands)]}")
            bands.remove(worst_band)
        return removed,best_score,average_score

