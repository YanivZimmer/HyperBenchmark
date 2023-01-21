import statistics
from typing import List, Callable, Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import logging
from cnn1_model import cnn_model
from HyperDataLoader import HyperDataLoader
from tensorflow.keras.utils import to_categorical

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
        masked_x_train = self.X_train[...,bands,:]
        masked_x_test = self.X_test[...,bands,:]
        from _datetime import datetime
        print(datetime.now())
        model = self.model_creator(len(bands))

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

    def search_without_one(self, all_bands: List[int],*args,**kwargs) -> Dict[int, float]:
        '''

        :param all_bands:
        :return: Dict of missing band and model score without him
        '''
        print("All bands",all_bands)
        print("args",args)
        print("kwargs",kwargs)
        missing_band_to_score = {}
        for band in all_bands:
            almost_all_bands = list(all_bands)
            almost_all_bands.remove(band)
            _, [loss,score] = self.assess_bands(almost_all_bands,*args,**kwargs)
            missing_band_to_score[band] = score
        return missing_band_to_score
    def find_worst(self,results:Dict[int,float])->int:
        '''

        :return: Return band for which without him the score was highest (this implies he is of less importance )
        '''

        return max(results, key=results.get)
    def search_all(self,all_bands: List[int],min_bands,*args,**kwargs):
        bands = list(all_bands)
        best_score = {}
        average_score = {}
        removed=[]
        while len(bands) > min_bands:
            print('len=',len(bands))
            results = self.search_without_one(bands,*args,**kwargs)
            print("here",len(bands))
            worst_band=self.find_worst(results)
            removed.append(worst_band)
            score=results[worst_band]
            logging.info(f"Words band was {len(bands)}, testing bands list of size {worst_band}. score without him={score}")
            best_score[len(bands)] = score
            average_score[len(bands)] = statistics.mean(results.values())
            logging.info(f"Average score for {len(bands) is average_score[len(bands)]}")
            bands.remove(worst_band)
        return removed,best_score,average_score



if __name__== '__main__':
    NUM_OF_CLASSES = 10
    loader=HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    print("bef=", len(X))
    X, y = loader.filter_unlabeled(X, y)
    print("aft=", len(X))
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    searcher = Searcher(lambda x: cnn_model(x,NUM_OF_CLASSES),X_train,y_train,X_test,y_test)
    removed, best_score, average_score = searcher.search_all(list(np.arange(1, 102, 5)),min_bands=14,epochs=11)
    print(removed)
    print()

