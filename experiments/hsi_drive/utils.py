import numpy as np

from hyper_data_loader.HyperDataLoader import HyperDataLoader, Labeled_Data
INPUT_SHAPE_PAVIA = 103
NUM_CLASSES_PAVIA = 10

INPUT_SHAPE_DRIVE = 25
NUM_CLASSES_DRIVE = 10

def train_iter(patch_shape):
    LIMIT=1000
    loader = HyperDataLoader()
    for labeled_data_iter in loader.generate_vectors("HSI-drive", patch_shape=patch_shape,shuffle=True,limit=LIMIT):
        X, y = labeled_data_iter.image, labeled_data_iter.lables
        X = X.reshape(X.shape[0], X.shape[3], X.shape[1], X.shape[2])
        X, y = loader.filter_unlabeled(X, y)
        #y_train = np.eye(NUM_CLASSES_DRIVE)[y_train]
        yield Labeled_Data(X,y)