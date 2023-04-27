from hyper_data_loader.HyperDataLoader import HyperDataLoader
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from hyper_data_loader.HyperDataLoader import HyperDataLoader
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    bands=[10,27,46]
    INPUT_SHAPE=len(bands)
    NUM_CLASSES=10
    model = keras.Sequential(
        [
            Dense(2 * INPUT_SHAPE, activation="relu", input_shape=(INPUT_SHAPE,)),
            Dropout(0.1),
            BatchNormalization(),
            Dense(int(INPUT_SHAPE), activation="relu"),
            Dropout(0.1),
            BatchNormalization(),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    LEARNING_RATE = 0.00002
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=[
            "accuracy",
        ],
    )
    masked_x_train = X_train[..., bands]
    masked_x_test = X_test[..., bands]
    history = model.fit(masked_x_train, y_train, epochs=50, batch_size=256, verbose=1)
    results = model.evaluate(masked_x_test, y_test, batch_size=256)
    print("Accuracy RGB over test set is {0}".format(results))
    #model.save("model_all_band")
    #return model, X_test, y_test