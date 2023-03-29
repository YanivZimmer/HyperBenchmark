import numpy as np
from sklearn.model_selection import train_test_split

from assesment import Assesment
from hyper_data_loader.HyperDataLoader import HyperDataLoader
from tensorflow.keras.utils import to_categorical

from models.cnn1_model import cnn_model
from models.mini_model_fn import mini_model3

NUM_OF_CLASSES = 10
LOADER = HyperDataLoader()


def run_bands_times(bands_amount:int,times:int,is_cnn,bands_assesment:Assesment):
    scores=[]
    for t in range(times):
        bands = np.sort(
            np.random.choice(
                range(1, 103 + 1), bands_amount, replace=False
            )
        )

        _, results = bands_assesment.assess_bands(bands - 1,is_cnn=is_cnn, epochs=75)
        # results[1] is accuracy
        print(f"iter {t} score= {results[1]}")
        scores.append(results[1])
    return scores


def main_30(times=100):
    labeled_data = LOADER.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = LOADER.filter_unlabeled(X, y)
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    assestor = Assesment(lambda x: cnn_model(x, NUM_OF_CLASSES), X_train, y_train, X_test, y_test)

    scores = run_bands_times(bands_amount=30, times=times,is_cnn=True ,bands_assesment=assestor)
    print("main_30",scores)


def main_same_10(times=20):
    # logger.info("Start")
    scores=[]
    labeled_data = LOADER.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = LOADER.filter_unlabeled(X, y)
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
    assestor = Assesment(lambda x: mini_model3(x, NUM_OF_CLASSES), X_train, y_train, X_test, y_test)
    bands = np.sort(
        np.random.choice(
            range(1, 103 + 1), 10, replace=False
        )
    )
    for t in range(times):
        _,results = assestor.assess_bands(list(bands),is_cnn=False)
        print(f"iter {t} score= {results[1]}")
        scores.append(results[1])
    print("main_same_10",scores)
    return scores

def main_same_30(times=20):
    scores=[]
    labeled_data = LOADER.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = LOADER.filter_unlabeled(X, y)
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    assestor = Assesment(lambda x: cnn_model(x, NUM_OF_CLASSES), X_train, y_train, X_test, y_test)
    bands = np.sort(
        np.random.choice(
            range(1, 103 + 1), 30, replace=False
        )
    )
    for t in range(times):
        _, results = assestor.assess_bands(list(bands),is_cnn=True)
        print(f"iter {t} score= {results[1]}")
        scores.append(results[1])
    print("main_same_30",scores)
    return scores

def main_same_30_suffle_data_per_iteration(times=20):
    scores=[]
    labeled_data = LOADER.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = LOADER.filter_unlabeled(X, y)
    y = to_categorical(y, num_classes=10)
    bands = np.sort(
        np.random.choice(
            range(1, 103 + 1), 30, replace=False
        )
    )
    for t in range(times):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        assestor = Assesment(lambda x: cnn_model(x, NUM_OF_CLASSES), X_train, y_train, X_test, y_test)
        _, results = assestor.assess_bands(list(bands),is_cnn=True)
        print(f"iter {t} score= {results[1]}")
        scores.append(results[1])
    print("main_same_30_suffle_data_per_iteration",scores)
    return scores

def main_10(times=100):
    labeled_data = LOADER.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = LOADER.filter_unlabeled(X, y)
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
    assestor = Assesment(lambda x: mini_model3(x, NUM_OF_CLASSES), X_train, y_train, X_test, y_test)

    scores = run_bands_times(bands_amount=10, times=times,is_cnn=False, bands_assesment=assestor)
    print("main_10",scores)
    return scores
if __name__ == "__main__":
    #logger.info("Start")
    #NUM_OF_CLASSES = 10
    #labeled_data = LOADER.generate_vectors("PaviaU", (1, 1))
    #X, y = labeled_data[0].image, labeled_data[0].lables
    #X, y = LOADER.filter_unlabeled(X, y)
    #y = to_categorical(y, num_classes=10)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle=True)
    #assestor = Assesment(lambda x: mini_model3(x, NUM_OF_CLASSES), X_train, y_train, X_test, y_test)

    #scores = run_bands_times(bands_amount=10,times=100,bands_assesment=assestor)
    #print(scores)
    main_30(1)
    main_10(1)
    main_same_30(1)
    main_same_10(1)
    main_same_30_suffle_data_per_iteration(2)