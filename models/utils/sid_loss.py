import torch
import torch.nn.functional as F

class SIDLoss(torch.nn.Module):
    def __init__(self):
        super(SIDLoss, self).__init__()

    def forward(self, x, y):
        eps = 1e-6
        x = x + eps
        y = y + eps
        x = F.normalize(x, p=1, dim=0)
        y = F.normalize(y, p=1, dim=0)
        sid = torch.sum(x * torch.log(x/y), dim=0) + torch.sum(y * torch.log(y/x), dim=0)
        sid = torch.mean(sid)
        return sid
