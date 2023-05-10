import torch.nn as nn


class DeeperFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels * 2,
                out_channels=out_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels * 2,
                out_channels=out_channels * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels * 4,
                out_channels=out_channels * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels * 4,
                out_channels=out_channels * 8,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels * 8,
                out_channels=out_channels * 8,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
