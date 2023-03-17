from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

class PytorchAssesment:
    def __init__(
        self,
        get_model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        self.test_loader = None
        self.train_loader = None
        self.get_model = get_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.optimizer = None
        self.criterion = None

    def mask_data(self, bands):
        masked_x_train = self.X_train[..., bands, :]
        masked_x_test = self.X_test[..., bands, :]

    def create_data_loaders(self, batch_size=32):
        train_dataset = TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.y_train))
        test_dataset = TensorDataset(torch.from_numpy(self.X_test),torch.from_numpy(self.y_test))
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)


    def test_model(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * (correct / total)
        print(f'Accuracy of the network on the 10000 test images: {acc} %')
        return acc

    def train_model(self,epochs=2):
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0




    def assess_bands(self, bands: List[int], *args, **kwargs) -> (nn.Module, float):
        self.mask_data(bands)
        self.create_data_loaders(batch_size=32)
        self.model, self.optimizer, self.criterion ,_ = self.get_model(len(bands))
        self.train_model()
        acc = self.test_model()
        return 0, acc
