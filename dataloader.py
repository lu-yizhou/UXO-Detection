import torch
from torch.utils.data import Dataset
import numpy as np


class UXODataset(Dataset):
    def __init__(self, data, label=None, transform=None):
        if isinstance(data, str):
            self.data = np.load(data)
        else:
            self.data = data
        if label is not None:
            if isinstance(label, str):
                self.label = np.load(label)
            else:
                self.label = label
        else:
            self.label = label

        # UXO与背景二分类时，将UXO的标签2转换为标签1
        if self.label is not None:
            self.label = np.array([1 if label == 2 else 0 for label in self.label])
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        x = torch.tensor(x).type(torch.float32)
        if self.label is not None:
            y = self.label[index]
            y = torch.tensor(y).type(torch.long)
            return x, y
        else:
            return x