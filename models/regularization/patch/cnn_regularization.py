import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
class MaskLayer(nn.Module):
    def __init__(self, input_shape, threshold=1e-3):
        super(MaskLayer, self).__init__()
        self.mask = nn.Parameter(torch.Tensor(*input_shape))
        self.sigmoid = nn.Sigmoid()

        # Initialize the mask parameters
        nn.init.uniform_(self.mask, a=0.0, b=1.0)

    def forward(self, x):
        mask = self.sigmoid(self.mask)
        return x * mask

    def l1_regularization(self):
        return torch.abs(self.mask).sum()

class CnnMasked(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CnnMasked, self).__init__()

        # Define the masking layer
        self.mask_layer = MaskLayer(input_shape)

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Define additional layers for learning
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * input_shape[1] * input_shape[2], 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.mask_layer(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def regularization(self):
        return self.mask_layer.l1_regularization()
