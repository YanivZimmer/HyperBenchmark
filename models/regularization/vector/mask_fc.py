import torch
from torch import nn

from models.utils.device_utils import get_device


class MaskFC(nn.Module):
    def __init__(self, input_shape, threshold):
        super(MaskFC, self).__init__()
        self.threshold = threshold
        self.fc1 = nn.Linear(input_shape, input_shape)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 200)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        #self.mask = torch.eye(input_shape[0], dtype=bool, device=get_device())

    def prune(self):
        pass

    def regularization(self):
        return torch.norm(self.fc1.weight.data,1)

    def mask(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.sig(x)