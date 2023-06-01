import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = None
        if transform != None:
            print("[Info] Applying transform to dataset")
            self.transform = torch.vmap(transform)

    def __getitem__(self, index):
        item = self.data[index]
        target = self.targets[index]
        x = torch.tensor(item)
        y = torch.tensor(target)

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


