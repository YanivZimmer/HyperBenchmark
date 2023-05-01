import torch
import torch.nn as nn


class CustomMLP(nn.Module):
    def __init__(self, x):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(x, x)
        self.fc2 = nn.Linear(x, x, bias=False)
        self.fc2.weight = nn.Parameter(torch.eye(x))

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


import torch.nn as nn


class MlpModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MlpModel, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_shape, 2 * input_shape),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(2 * input_shape)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2 * input_shape, max(int(0.5 * input_shape), 1)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(max(int(0.5 * input_shape), 1))
        )

        self.fc3 = nn.Sequential(
            nn.Linear(max(int(0.5 * input_shape), 1), num_classes),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
