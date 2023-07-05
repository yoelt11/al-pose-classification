import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, data, targets, time_frames, transform=None):
        self.T = time_frames
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
        if isinstance(target, str):
            y = target
        else:
            y = torch.tensor(target)

        if self.transform is not None:
            with torch.no_grad():
                idx = self.gen_random_idx(x.shape[0], self.T)
                x = self.transform(x[idx])
        return x, y

    def __len__(self):
        return len(self.data)

    def gen_random_idx(self, org_time_frames, new_time_frames):
        idx = torch.randperm(org_time_frames)[:new_time_frames]
        ordered_idx = torch.sort(idx).values
        return ordered_idx
