from typing import Callable, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from models.pytorch_models.models import train,val
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

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

    def mask_data(self, data, bands):
       return data[..., bands, :]

    def create_data_loaders(self, batch_size=32):
        train_dataset = TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.y_train))
        test_dataset = TensorDataset(torch.from_numpy(self.X_test), torch.from_numpy(self.y_test))
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)


    def test_model2(self):
        num_correct = 0
        num_samples = 0
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(device=device)
                y = y.to(device=device)
                scores = self.model(x)
                _, predictions = scores.max(1)
                y = y.squeeze()
                y = torch.argmax(y)
                num_correct += (y == predictions).sum().item()
                num_samples += predictions.size(0)

        acc = float(num_correct)/float(num_samples) #float(num_correct) / float(num_samples) * 100:.2f
        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
        return 0,acc
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
                lables=labels.squeeze()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * (correct / total)
        print(f'Accuracy of the network on the 10000 test images: {acc} %')
        return acc

    def train_model(self,epochs):
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                labels=labels.squeeze()
                outputs=outputs.squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0




    def assess_bands(self, bands: List[int],epochs,batch_size, *args, **kwargs) -> (nn.Module, float):
        #self.mask_data(bands)
        self.create_data_loaders(batch_size=batch_size)
        self.model, self.optimizer, self.criterion ,_ = self.get_model(len(bands))
        #train(self.model,self.optimizer,self.criterion,self.train_loader,10)
        self.train_model(epochs=epochs)
        acc = self.test_model2()
        return 0, acc
