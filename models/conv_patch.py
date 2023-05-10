import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D


def conv_patch(bands_num, patch_size, num_classes):
    LEARNING_RATE = 0.00002
    model = keras.Sequential(
        [
            # Dense(20 * input_shape, activation="relu", input_shape=(input_shape,)),
            Conv2D(
                20 * bands_num,
                kernel_size=(2, 2),
                input_shape=(patch_size, patch_size, bands_num),
            ),
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
