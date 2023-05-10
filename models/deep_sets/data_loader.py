import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.from_numpy(data).float()
        # Pay attention for squeeze if batch size is 1
        self.labels = (
            torch.squeeze(torch.from_numpy(labels).float())
            if labels is not None
            else None
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.labels == None:
            return self.data[index]
        return self.data[index], self.labels[index]


def create_data_loader(data, labels=None, batch_size=256, shuffle=True):
    dataset = MyDataset(data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
