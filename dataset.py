import pathlib
import random

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram, Vol

from config import OUT_FREQ


def transform_distort_audio(x: Tensor):
    vol_change = random.uniform(-10, 10)
    x = Vol(gain=vol_change, gain_type='db')(x)
    dyn_change = torch.rand(size=x.shape) * 0.4 + 0.8
    x = x * dyn_change
    noise_level = random.uniform(0, 0.05)
    noise = noise_level * torch.rand(size=x.shape) * torch.normal(mean=0, std=1, size=x.shape)
    x = x + noise
    return x


def transform_melspectrogram(x: Tensor, large=False):
    if large:
        x = MelSpectrogram(sample_rate=OUT_FREQ, n_mels=128, n_fft=600, hop_length=OUT_FREQ // 128, power=2.0,
                           normalized=False, f_min=125)(x)[..., :128]
    else:
        x = MelSpectrogram(sample_rate=OUT_FREQ, n_mels=64, n_fft=400, hop_length=OUT_FREQ // 64, power=2.0,
                           normalized=False, f_min=125)(x)[..., :64]
    x = amplitude_to_DB(x, multiplier=10, amin=1e-10, db_multiplier=10., top_db=80.)
    x -= x.amin((1, 2), keepdim=True)
    mx = x.amax((1, 2), keepdim=True)
    mx[mx == 0] = 1
    x /= mx
    return x


def transform_random_sample(x: Tensor):
    rand_offset = random.randint(0, OUT_FREQ // 4)
    x = x[..., rand_offset:OUT_FREQ + rand_offset]
    return x


def transform_sample_normal(x: Tensor, large=False):
    x = transform_random_sample(x)
    x = transform_melspectrogram(x, large)
    return x


def transform_sample_augmented(x: Tensor, large=False):
    x = transform_random_sample(x)
    x = transform_distort_audio(x)
    x = transform_melspectrogram(x, large)
    return x


class AudioSamplePairDataset(Dataset):
    def __init__(self, root_path: str, shuffle: bool = True, test=False, limit=-1, large=False):
        self.root_path = root_path
        dataset = list(str(p) for p in pathlib.Path(root_path).rglob('*.pck'))
        dataset = dataset[0:limit]

        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
        dataset = test_dataset if test else train_dataset

        if shuffle:
            random.shuffle(dataset)
        self.dataset = dataset
        self.large = large

    def __getitem__(self, index):
        val = torch.load(self.dataset[index])
        contr_idx = index
        while contr_idx == index:
            contr_idx = random.randint(0, len(self.dataset) - 1)
        contr_val = torch.load(self.dataset[contr_idx])
        return transform_sample_normal(val, self.large), \
               transform_sample_augmented(val, self.large), \
               transform_sample_augmented(contr_val, self.large)

    def __len__(self) -> int:
        return len(self.dataset)


def show_spec(x):
    plt.imshow(x.squeeze())
    plt.show()


if __name__ == "__main__":
    dataset = AudioSamplePairDataset('data/fma_small_samples', limit=100, large=True)
    samples = [(x, y, z) for _, (x, y, z) in zip(range(9), dataset)]
    fig, axs = plt.subplots(9, 3, figsize=(20, 60))
    fig.tight_layout()
    labels = ['anchor', 'positive', 'negative']
    for i in range(9):
        for j in range(3):
            axs[i, j].set_title('{} - {}'.format(i, labels[j]))
            axs[i, j].imshow(samples[i][j].squeeze())
    plt.show()
