import torch.nn as nn
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DiagonalLinear(nn.Module):
    def __init__(self, in_features, out_features, threshold=1e-4):
        super(DiagonalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False,device=device)
        self.mask = torch.eye(in_features, dtype=bool,device=device)
        self.threshold = threshold

    def forward(self, x):
        self.linear.weight.data *= self.mask
        over_threshold = torch.abs(self.linear.weight.data) > self.threshold
        self.linear.weight.data *= over_threshold
        return self.linear(x)


class RegMlpModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(RegMlpModel, self).__init__()
        self.one_to_one=DiagonalLinear(input_shape,input_shape)
        self.fc1 = nn.Sequential(
            nn.Linear(input_shape, 2 * input_shape),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(2 * input_shape)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2 * input_shape, int(0.5 * input_shape)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(int(0.5 * input_shape))
        )

        self.fc3 = nn.Sequential(
            nn.Linear(int(0.5 * input_shape), num_classes),
        )

    def regularization(self):
        non_zero = torch.nonzero(self.one_to_one.linear.weight)
        print("non_zero", len(non_zero))
        return torch.norm(self.one_to_one.linear.weight, 1)

    def forward(self, x):
        x = self.one_to_one(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

