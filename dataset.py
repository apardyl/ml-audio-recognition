import pathlib
import random

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class AudioSamplePairDataset(Dataset):
    def __init__(self, root_path: str, transform=lambda x: x, shuffle: bool = True, test=False):
        self.root_path = root_path
        dataset = list(str(p) for p in pathlib.Path(root_path).rglob('*.pck'))

        train_dataset, test_dataset = train_test_split(dataset, test_size=0.25)
        dataset = test_dataset if test else train_dataset

        if shuffle:
            random.shuffle(dataset)

        self.dataset = dataset
        self.transform = transform
        self.counter = 0

    def __getitem__(self, index):
        self.counter += 1
        val = torch.load(self.dataset[index])
        contr_idx = abs(hash(self.dataset[index]) + self.counter) % len(self.dataset)
        contr_val = torch.load(self.dataset[contr_idx])
        return self.transform(val), self.transform(val), self.transform(contr_val)

    def __len__(self) -> int:
        return len(self.dataset)
