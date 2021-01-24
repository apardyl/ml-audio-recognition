import random

import torch
from torch import nn, Tensor
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram, Vol

from config import OUT_FREQ


def save_train_state(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, best_score: float,
                     file_path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_score': best_score,
    }, file_path)


def load_train_state(file_path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler):
    data = torch.load(file_path)
    model.load_state_dict(data['model_state'])
    optimizer.load_state_dict(data['optimizer_state'])
    if 'scheduler' in data:
        scheduler.load_state_dict(data['scheduler'])
    return data['epoch'], data.get('best_score', 0)


def load_model_state(file_path, model: nn.Module):
    data = torch.load(file_path)
    model.load_state_dict(data['model_state'])


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


def transform_distort_audio(x: Tensor):
    vol_change = random.uniform(-10, 10)
    x = Vol(gain=vol_change, gain_type='db')(x)
    dyn_change = torch.rand(size=x.shape) * 0.4 + 0.8
    x = x * dyn_change
    noise_level = random.uniform(0, 0.05)
    noise = noise_level * torch.rand(size=x.shape) * torch.normal(mean=0, std=1, size=x.shape)
    x = x + noise
    return x
