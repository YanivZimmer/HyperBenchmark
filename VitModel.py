import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchsummary import summary
from torchvision.transforms import Compose, Resize, ToTensor

# from einops import rearrange, reduce, repeat
# from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    def __init__(self, input_patches, patch_size: int = 9, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_patches, emb_size)
            # nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # Rearrange('b e (h) (w) -> b (h w) e'),
        )  # this breaks down the image in s1xs2 patches, and then flat them

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
