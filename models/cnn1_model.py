import numpy as np
from tensorflow.keras.layers import MaxPooling1D
from tensorflow import keras
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Conv1D,
)
from hyper_data_loader.HyperDataLoader import HyperDataLoader
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

INPUT_SHAPE_PAVIA = 103
NUM_CLASSES_PAVIA = 10

INPUT_SHAPE_DRIVE = 25
NUM_CLASSES_DRIVE = 10


def pavia_cnn1():
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (1, 1))

    X, y = labeled_data[0].image, labeled_data[0].lables
    print("bef=", len(X))
    X, y = loader.filter_unlabeled(X, y)
    print("aft=", len(X))
    y = to_categorical(y, num_classes=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = cnn_model(INPUT_SHAPE_PAVIA, NUM_CLASSES_PAVIA)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    #X_train = X_train.reshape((1, X_train.shape[0], X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    history = model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=1)
    results = model.evaluate(X_test, y_test, batch_size=256)
    print("Accuracy over test set is {0}".format(results))
    model.save("model_all_band")
    return model, X_test, y_test


def filter_unlablled(X, y):
    # idx = np.argsort(y)
    idx = np.where(y != 0)
    y = y[idx[0]]
    X = X[idx[0], :]
    # Make lables 0-9
    y -= 1
    return X, y


def combine_hsi_drive(test_size=0.33):
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("HSI-drive",patch_shape=(1,1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    for item in labeled_data[1:]:
        X = np.concatenate((X, item.image))
        y = np.concatenate((y, item.lables))
    X, y = filter_unlablled(X, y)
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True
    )
    return X_train, X_test, y_train, y_test


def cnn_model(input_size, num_of_class):
    model = keras.Sequential(
        [
            Conv1D(
                kernel_size=11,
                filters=20 * 93,
                input_shape=(input_size,1),
                activation="tanh",
            ),
            MaxPooling1D(pool_size=3),
            Flatten(),
            Dense(100, activation="tanh"),
            Dense(num_of_class, activation="softmax"),
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
    model.summary()
    return model


def hsi_drive_cnn1():
    X_train, X_test, y_train, y_test = combine_hsi_drive()
    model = cnn_model(INPUT_SHAPE_DRIVE, NUM_CLASSES_DRIVE)
    print("bef", X_train.shape, X_test.shape)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print("aft", X_train.shape, X_test.shape)
    history = model.fit(X_train[:123456], y_train[:123456], epochs=50, batch_size=2048, verbose=1)
    results = model.evaluate(X_test, y_test, batch_size=2048)
    print("Accuracy over test set is {0}".format(results))
    return model, X_test, y_test


#pavia_cnn1()
#hsi_drive_cnn1()
