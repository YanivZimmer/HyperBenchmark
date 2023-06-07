import numpy as np

from experiments.hsi_drive.utils import train_iter
from models.cnn1_model import NUM_CLASSES_DRIVE
from models.regularization.patch.cnn_regularization import CnnMasked
from models.utils.device_utils import get_device
from models.utils.data_loader import create_data_loader
from models.utils.train_test import train_model, simple_test_model

PATCH_SIZE = 3
N_BANDS = 25

def experiment(model):
    limit = 200
    iter = train_iter((PATCH_SIZE,PATCH_SIZE))
    for index, item in zip(range(limit), iter):
        print(index, item.image.shape,item.image[10][10][0])
        y = np.eye(NUM_CLASSES_DRIVE)[item.lables]
        train_loader = create_data_loader(item.image, y, 256)
        get_device()
        train_model(model, train_loader, epochs=1, lr=0.000001, device=get_device(),regularization=True)

    acc_list = []
    for i in range(20):
        values = next(iter)
        test_loader = create_data_loader(values.image, values.lables, 256)
        acc=simple_test_model(model, test_loader, device=get_device())
        acc_list.append(acc)
    print(f"Average acc on {len(acc_list)} images: {sum(acc_list)/len(acc_list)}")


if __name__ == '__main__':
    model = CnnMasked((N_BANDS,PATCH_SIZE,PATCH_SIZE),NUM_CLASSES_DRIVE)
    experiment(model)
