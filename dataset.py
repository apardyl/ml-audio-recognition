import pathlib
import random
import sys

import matplotlib.pyplot as plt
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import Vol

from config import OUT_FREQ
from utils import transform_melspectrogram, transform_distort_audio

torchaudio.set_audio_backend('sox_io')


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


class AudioSamplePairDualDataset(Dataset):
    def __init__(self, root_path: str, shuffle: bool = True, test=False, limit=-1):
        self.root_path = root_path
        dataset = list(str(p) for p in pathlib.Path(root_path).rglob('*.pck'))
        dataset = dataset[0:limit]

        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
        dataset = test_dataset if test else train_dataset

        if shuffle:
            random.shuffle(dataset)
        self.dataset = dataset

    def __getitem__(self, index):
        val = torch.load(self.dataset[index])

        x = transform_random_sample(val)
        x_s = transform_melspectrogram(x, False)
        x_l = transform_melspectrogram(x, True)

        y = transform_random_sample(val)
        y = transform_distort_audio(y)
        y_s = transform_melspectrogram(y, False)
        y_l = transform_melspectrogram(y, True)

        return x_s, y_s, x_l, y_l

    def __len__(self) -> int:
        return len(self.dataset)


class AudioIndexingDataset(Dataset):
    def __init__(self, root_path: str, limit=-1):
        self.root_path = root_path
        dataset = list(str(p) for p in pathlib.Path(root_path).rglob('*.mp3'))
        self.dataset = dataset[0:limit]

    def __getitem__(self, index):
        file_path = self.dataset[index]
        try:
            track, sr = torchaudio.load(file_path, normalize=True, channels_first=True)
            track = track.mean(dim=0, keepdim=True)
            track = torchaudio.transforms.Resample(orig_freq=sr, new_freq=OUT_FREQ).forward(track)
        except Exception as ex:
            print("File {} invalid - {}".format(file_path, ex), file=sys.stderr)
            return None
        if track.shape[1] < OUT_FREQ * 10:
            print("File {} too short".format(file_path), file=sys.stderr)
            return None

        sample_length_points = int(OUT_FREQ * 1)
        idx = 0
        track_len = track.shape[1]
        samples_s = []
        samples_l = []
        while idx + sample_length_points < track_len:
            sample = track[:, idx:idx + sample_length_points]
            samples_s.append(transform_melspectrogram(sample, large=False))
            samples_l.append(transform_melspectrogram(sample, large=True))
            idx += sample_length_points
        return file_path, torch.stack(samples_s, dim=0), torch.stack(samples_l, dim=0)

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
