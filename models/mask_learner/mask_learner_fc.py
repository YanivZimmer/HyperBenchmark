import torch
from torch import nn

class MaskLearnerFc(nn.Module):
    def __init__(self, input_size, num_labels):
        super().__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.fc = nn.Linear(input_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Reshape input to have shape (batch_size, input_size)
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

    def arr_to_mask(self, tensor):
        # Get the indices of the top 3 maximum values
        topk_values, topk_indices = torch.topk(tensor, k=3, dim=1)
        result_tensor = torch.zeros(tensor.shape)
        result_tensor[topk_indices] = 1
        return result_tensor



