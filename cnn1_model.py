import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling1D
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization,Conv1D,Conv2D
from HyperDataLoader import HyperDataLoader
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

INPUT_SHAPE = 103
NUM_CLASSES = 10
def cnn1():
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU")
    X, y = labeled_data[0].image, labeled_data[0].lables
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print(X_train[0].shape)
    #model= keras.Sequential()
    #model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',
    #                 input_shape=(INPUT_SHAPE, 1)))
    model = keras.Sequential(
        [
            Conv1D(kernel_size=(11),filters=20*93,input_shape=(103,1),activation='tanh'),
            MaxPooling1D(pool_size=3),
            Flatten(),
            Dense(100,activation='tanh'),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    LEARNING_RATE = 0.000002
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=[
            "accuracy",
        ],
    )
    print(X_train.shape)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(X_train.shape)

    history = model.fit(X_train, y_train, epochs=50, batch_size=256, verbose=1)
    results = model.evaluate(X_test, y_test, batch_size=256)
    print("Accuracy over test set is {0}".format(results))
    model.save("model_all_band")
    return model, X_test, y_test

cnn1()