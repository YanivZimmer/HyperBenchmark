import logging
import os
from datetime import datetime
from multiprocessing import Pool
from Experiments.Assesment import Assesment
now=datetime.now()
date_time = now.strftime("%m%d%Y_%H%M")
print(os.getcwd())
filename=f'../logs/Hyperspectral_extensive_{date_time}.log'
logging.basicConfig(level=logging.DEBUG,format='%(name)s - %(levelname)s - %(message)s',handlers=[
        logging.FileHandler(filename),
        logging.StreamHandler()
    ])
logging.info("hi")
logger = logging.getLogger()
logger.info("ya")    
import statistics
from typing import List, Callable, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from HyperDataLoader.HyperDataLoader import HyperDataLoader
from models.cnn1_model import cnn_model
from tensorflow.keras.utils import to_categorical

class ESearcher(Assesment):
    def __init__(
        self,
        model_creator: Callable[[int], Model],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        super().__init__(model_creator,
                       X_train,y_train,X_test,y_test)

    def search_without_one(self, all_bands: List[int],*args,**kwargs) -> Dict[int, float]:
        '''

        :param all_bands:
        :return: Dict of missing band and model score without him
        '''
        missing_band_to_score = {}
        for band in all_bands:
            almost_all_bands = list(all_bands)
            almost_all_bands.remove(band)
            _, [loss,score] = self.assess_bands(almost_all_bands,*args,**kwargs)
            missing_band_to_score[band] = score
        return missing_band_to_score
    def search_parallel(self, all_bands: List[int],parallel_runs:int,*args,**kwargs) -> Dict[int, float]:
        missing_band_to_score = {}
        rem_idx=list(filter(lambda x:x % parallel_runs == 0,range(len(all_bands))))
        reminder=len(all_bands) % parallel_runs
        for i in rem_idx[:-2]:
            scores = self.assess_parallel(all_bands[i:i+parallel_runs],*args,**kwargs)
            for k in range(parallel_runs):
                missing_band_to_score[all_bands[i+k]] = scores[k]
        scores = self.assess_parallel(all_bands[rem_idx[-1]:reminder+rem_idx[-1]],*args,**kwargs)
        for k in range(reminder):
            missing_band_to_score[all_bands[rem_idx[-1] + k]] = scores[k]
        return missing_band_to_score

    def assess_once(self,bands,*args,**kwargs) -> float:
        _, [loss, score] = self.assess_bands(bands, *args, **kwargs)
        return score
    def assess_parallel(self, bands_to_test,all_bands,*args,**kwargs) -> List[float]:
        almost_all_list = []
        for band in bands_to_test:
            almost_all_bands = list(all_bands)
            almost_all_bands.remove(band)
            almost_all_list.append(almost_all_bands)

        with Pool() as pool:
            res = pool.map(self.assess_once, almost_all_list)
            print(res)
            return res

    def find_worst(self,results:Dict[int,float]) -> int:
        '''

        :return: Return band for which without him the score was highest (this implies he is of less importance )
        '''

        return max(results, key=results.get)
    def search_all(self,all_bands: List[int],min_bands,parallel_runs=2,*args,**kwargs):
        bands = list(all_bands)
        best_score = {}
        average_score = {}
        removed=[]
        while len(bands) > min_bands:
            results = self.search_parallel(bands,parallel_runs,*args,**kwargs)
            worst_band=self.find_worst(results)
            removed.append(worst_band)
            score=results[worst_band]
            logger.info(f"Words band was {len(bands)}, testing bands list of size {worst_band}. score without him={score}")
            best_score[len(bands)] = score
            average_score[len(bands)] = statistics.mean(results.values())
            logging.info(f"Average score for {len(bands) is average_score[len(bands)]}")
            bands.remove(worst_band)
        return removed,best_score,average_score


if __name__== '__main__':
    logger.info("Start")
    NUM_OF_CLASSES = 10
    loader=HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = loader.filter_unlabeled(X, y)
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    searcher = ESearcher(lambda x: cnn_model(x,NUM_OF_CLASSES),X_train,y_train,X_test,y_test)
    removed, best_score, average_score = searcher.search_all(list(np.arange(1, 103, 10)),min_bands=14,epochs=75)
    logger.critical("best_score:=",best_score)
    logger.critical("removed:=",removed)
    logger.critical("average_score:=",average_score)

