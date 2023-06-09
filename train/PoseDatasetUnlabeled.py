import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        item = self.data[index]
        x = torch.tensor(item["kp"])
        filename = item["file_name"]
        #y = torch.tensor(item["target"])

        if self.transform is not None:
            x = self.transform(x)

        return x, filename

    def __len__(self):
        return len(self.data)


