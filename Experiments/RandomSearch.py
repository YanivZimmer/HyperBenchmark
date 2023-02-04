import statistics
from typing import Callable
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import logging
from datetime import datetime
from Experiments.Assesment import Assesment
from HyperDataLoader.HyperDataLoader import HyperDataLoader
from models.cnn1_model import cnn_model
from tensorflow.keras.utils import to_categorical
import random
now=datetime.now()
date_time = now.strftime("%m%d%Y_%H%M")
logging.basicConfig(filename=f'../logs/Hyperspectral_random_{date_time}.log', filemode='w'
                    ,level=logging.DEBUG,format='%(name)s - %(levelname)s - %(message)s')
global logger
logger=logging.getLogger()
logger.info("")

class RandomSearch(Assesment):
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

    def random_assess(self,min_bands,attempts_per_bands_amounts,*args,**kwargs):
        #expect second last dim of X_train to be number of bands
        best_score = {}
        average_score = {}
        selected = {}
        amount = self.X_train.shape[-2]
        while amount > min_bands:
            scores = []
            candidates = []
            for i in range(attempts_per_bands_amounts):
                bands = np.sort(np.random.choice(range(1,self.X_train.shape[-2]+1), amount, replace=False))# random.sample(range(1, self.X_train.shape[-2]+1), amount)
                candidates.append(bands)
                #bands-1 to norm bands index to match array index and start with 0
                model, results = self.assess_bands(bands-1, *args, **kwargs)
                #results[1] is accuracy
                scores.append(results[1])
            best_score[amount] = max(scores)
            average_score[amount] = statistics.mean(scores)
            selected[amount] = candidates[np.argmax(scores)]
            amount -= 1
            logger.info(f"Average score for {amount}) is {average_score[amount]}")
        return selected, best_score, average_score


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
    searcher = RandomSearch(lambda x: cnn_model(x,NUM_OF_CLASSES), X_train,y_train,X_test,y_test)
    selected, best_score, average_score = searcher.random_assess(14, 3,epochs=50) #searcher.search_all(list(np.arange(1, 103, 1)),min_bands=14,epochs=100)
    logger.info("best_score:=",best_score)
    logger.info("average_score:=",average_score)
    logger.info("selected:=", selected)