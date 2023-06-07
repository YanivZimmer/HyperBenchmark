import torch
from torch import nn

class MaskLearnerCNN(nn.Module):
    def __init__(self, input_shape, num_classes, n_out_features):
        super(MaskLearnerCNN, self).__init__()
        self.n_out_features = n_out_features
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

    def prob_to_mask(self, arr):
        return arr.argsort()[-self.n_out_features:][::-1]
