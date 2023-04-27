import numpy as np
import torch

from experiments.hsi_drive.hsi_drive import train_iter, NUM_CLASSES_DRIVE, device
from models.deep_sets.data_loader import create_data_loader
from models.deep_sets.deep_sets import train_model, simple_test_model


def main_iter_unet(limit):
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=25, out_channels=10, init_features=16, pretrained=False)
    iter=train_iter()
    for index, item in zip(range(limit), iter):
        print(index, item.image.shape,item.image[10][10][0])
        y = np.eye(NUM_CLASSES_DRIVE)[item.lables]
        train_loader = create_data_loader(item.image, y, 256)
        train_model(model, train_loader, epochs=1, lr=0.00001, device=device)
    acc_list=[]

    for i in range(20):
        values = next(iter)
        test_loader = create_data_loader(values.image, values.lables, 256)
        acc=simple_test_model(model, test_loader, device=device)
        acc_list.append(acc)
    print(f"Average acc on {len(acc_list)} images: {sum(acc_list)/len(acc_list)}")


if __name__=='__main__':
    main_iter_unet(200)