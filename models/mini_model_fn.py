import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


def mini_model1(input_shape,num_classes):
    LEARNING_RATE = 0.00002
    model = keras.Sequential(
            [
                Dense(2 * input_shape, activation="relu", input_shape=(1,input_shape,)),
                Dropout(0.1),
                BatchNormalization(),
                Dense(int(0.5 * input_shape), activation="relu"),
                Dropout(0.1),
                BatchNormalization(),
                Dense(num_classes, activation="softmax"),
            ]
        )
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=[
            "accuracy",
        ],
    )
    return model
def mini_model2(input_shape,num_classes):
    LEARNING_RATE = 0.00002
    model = keras.Sequential(
            [
                Dense(2 * input_shape, activation="relu", input_shape=(1,1,input_shape)),
                Dropout(0.1),
                BatchNormalization(),
                Dense(int(0.5 * input_shape), activation="relu"),
                Dropout(0.1),
                BatchNormalization(),
                Dense(num_classes, activation="softmax"),
            ]
        )
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=[
            "accuracy",
        ],
    )
    return model
def mini_model3(input_shape,num_classes):
    LEARNING_RATE = 0.00002
    model = keras.Sequential(
            [
                Dense(20 * input_shape, activation="relu", input_shape=(input_shape,)),
                Dropout(0.1),
                BatchNormalization(),
                Dense(100, activation="relu"),
                Dropout(0.1),
                BatchNormalization(),
                Dense(num_classes, activation="softmax"),
            ]
        )
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=[
            "accuracy",
        ],
    )
    return model
def mini_model4(input_shape,num_classes):
    LEARNING_RATE = 0.00002
    model = keras.Sequential(
            [
                Dense(2 * input_shape, activation="relu", input_shape=(1,input_shape)),
                Dropout(0.1),
                BatchNormalization(),
                Dense(int(0.5 * input_shape), activation="relu"),
                Dropout(0.1),
                BatchNormalization(),
                Dense(num_classes, activation="softmax"),
            ]
        )
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=[
            "accuracy",
        ],
    )
    return model

def mini_model5(input_shape,num_classes):
    LEARNING_RATE = 0.00002
    model = keras.Sequential(
            [
                Dense(2 * input_shape, activation="relu", input_shape=(input_shape,1)),
                Dropout(0.1),
                BatchNormalization(),
                Dense(int(0.5 * input_shape), activation="relu"),
                Dropout(0.1),
                BatchNormalization(),
                Dense(num_classes, activation="softmax"),
            ]
        )
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=[
            "accuracy",
        ],
    )
    return model