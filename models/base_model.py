import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from hyper_data_loader.HyperDataLoader import HyperDataLoader
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

INPUT_SHAPE = 103
NUM_CLASSES = 10


def train_base_model(load_existing):
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU")
    X, y = labeled_data[0].image, labeled_data[0].lables
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    if load_existing:
        model = keras.models.load_model("../model_all_band")
        return model, X_test, y_test

    model = keras.Sequential(
        [
            Dense(2 * INPUT_SHAPE, activation="relu", input_shape=(INPUT_SHAPE,)),
            Dropout(0.1),
            BatchNormalization(),
            Dense(int(0.5 * INPUT_SHAPE), activation="relu"),
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
    history = model.fit(X_train, y_train, epochs=50, batch_size=256, verbose=1)
    results = model.evaluate(X_test, y_test, batch_size=256)
    print("Accuracy over test set is {0}".format(results))
    model.save("model_all_band")
    return model, X_test, y_test


def train_selected_bands_model(bands_mask):
    input_shape = len(bands_mask)
    model = keras.Sequential(
        [
            Dense(2 * input_shape, activation="relu", input_shape=(input_shape,)),
            Dropout(0.1),
            BatchNormalization(),
            Dense(int(0.5 * input_shape), activation="relu"),
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
    # print(model.summary())
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU")
    X, y = labeled_data[0].image, labeled_data[0].lables
    # print(X.shape)
    X = X[:, bands_mask]
    # print(X.shape)
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # dummy_x = X_train[0].reshape(1, len(X_train[0]))
    # print("Xtrain pred:", model.predict(dummy_x))

    history = model.fit(X_train, y_train, epochs=50, batch_size=256, verbose=1)
    print(history)
    results = model.evaluate(X_test, y_test, batch_size=256)
    print("Accuracy over test set is {0}".format(results))


def build_class_heatmap(model: keras.Model, data, lables, features_num, class_label):
    arr = np.zeros(features_num)
    for x, y in zip(data, lables):
        if y == class_label:
            for i in range(features_num):
                v_i = np.zeros(features_num)
                v_i[i] = x[i]
                pred_i = model.predict(v_i)
                pred_zero = model.predict(np.zeros(features_num))
                diff = pred_i - pred_zero
                arr[i] += diff[i] - diff[np.argmax(diff)]
    return arr


def build_all_heatmaps(model: keras.Model, data, lables, features_num, labels_num):
    arr = np.zeros((labels_num, features_num))
    k = 0
    for x, y_categorical in zip(data, lables):
        y = np.argmax(y_categorical)
        for i in range(features_num):
            k += 1
            v_i = np.zeros((1, features_num))
            pred_zero = model.predict(v_i)
            v_i[0][i] = x[i]
            pred_i = model.predict(v_i)
            diff = pred_i - pred_zero
            arr[y][i] += diff[0][y] - diff[0][np.argmax(diff)]
    return arr


def build_all_heatmaps_fast(model: keras.Model, data, lables, features_num, labels_num):
    arr = np.zeros((labels_num, features_num))
    pred_zero = model.predict(np.zeros((1, features_num)))
    for x, y_categorical in zip(data, lables):
        y = np.argmax(y_categorical)
        mat = np.zeros((features_num, features_num))
        np.fill_diagonal(mat, x)
        pred_all_features = model.predict(mat)
        # b0 = pred_all_features = model.predict(mat[0])
        # b1 = pred_all_features = model.predict(mat[1])
        pred_all_features -= pred_zero
        arr[y] += pred_all_features[:, y] / sum(pred_all_features[:, y])
        # argm = np.argmax(pred_all_features,axis=1)
        # t=tuple(argm)
        # arr[y] -= pred_all_features[:,t]
    return arr


def feature_class_contribution(
    model: keras.Model, feature_idx, features_num, label_idx
):
    v_i = np.zeros((1, features_num))
    pred_zero = model.predict(v_i)
    v_i[0][feature_idx] = 1
    pred_i = model.predict(v_i)
    diff = pred_i - pred_zero
    contribution = diff[0][label_idx] - diff[0][np.argmax(diff)]
    return contribution


def feature_multiple_class_contribution(model: keras.Model, feature_idx, features_num):
    v_i = np.zeros((1, features_num))
    pred_zero = model.predict(v_i)
    v_i[0][feature_idx] = 1
    pred_i = model.predict(v_i)
    diff = pred_i - pred_zero
    return diff


def feature_contributions(model: keras.Model, features_num, labels_num, normalize):
    arr = np.zeros((features_num, labels_num))
    for feature_idx in range(features_num):
        arr[feature_idx] = feature_multiple_class_contribution(
            model, feature_idx, features_num
        )
    if normalize:
        for i in range(len(arr)):
            arr[i] = arr[i] / np.linalg.norm(arr[i])
    return arr


def fast():
    # anomally
    print("Anomally")
    # train_selected_bands_model((5, 6, 9, 38, 39, 48, 50, 53, 66, 84, 85))
    # train_selected_bands_model((5, 6, 48, 50, 66, 84))
    train_selected_bands_model((5, 48, 50, 84))

    # anomally
    print("Anomally Norm")
    # train_selected_bands_model((8, 12, 16, 31, 37, 48, 64, 66, 67, 84, 88))
    # train_selected_bands_model((16, 31, 37, 48, 67, 88))
    train_selected_bands_model((16, 31, 48, 88))
    # random
    print("Random")
    random_selected_bands = tuple(np.random.randint(low=1, high=103, size=4))
    print("Random", random_selected_bands)
    train_selected_bands_model(random_selected_bands)

    print("All")
    train_base_model()

    # fc = feature_contributions(model, len(X_test[0]), NUM_CLASSES,True)
    # print("fc", fc)
    # np.save('data_norm.npy', fc) # save
    print("Done")


def real_heatmaps():
    model, X_test, y_test = train_base_model(True)
    _, X_test, _, y_test = train_test_split(
        X_test, y_test, test_size=0.01, shuffle=True
    )
    arr = build_all_heatmaps_fast(model, X_test, y_test, len(X_test[0]), NUM_CLASSES)
    np.save("datareal.npy", arr)  # save


def compare():
    print("Anomally")
    train_selected_bands_model((2, 4, 7, 10, 19, 38, 41, 52, 74, 87))
    print("All")
    train_base_model(True)
    random_selected_bands = tuple(np.random.randint(low=1, high=103, size=10))
    print("Random", random_selected_bands)
    train_selected_bands_model(random_selected_bands)


if __name__ == "__main__":
    compare()
