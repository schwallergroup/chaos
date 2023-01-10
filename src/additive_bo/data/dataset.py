from torch import Tensor
from torch.utils.data.dataset import Dataset


class SingleSampleDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y
        self.samples = ((x, y),)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.samples[idx]


from torch.utils.data import Dataset


class DynamicSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def reset(self, dataset):
        self.dataset = dataset
