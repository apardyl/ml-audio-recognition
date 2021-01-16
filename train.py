import random

import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
from torch import Tensor
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram
from torchsummary import summary

from config import OUT_FREQ
from dataset import AudioSamplePairDataset
from encoder import Encoder


def show_spec(x):
    plt.imshow(x.squeeze())
    plt.show()


def transform_sample(x: Tensor):
    rand_offset = random.randint(0, OUT_FREQ // 5)
    x = x[:, rand_offset:OUT_FREQ + rand_offset]
    x = MelSpectrogram(sample_rate=OUT_FREQ, n_mels=64, n_fft=400, hop_length=OUT_FREQ // 64, power=2.0,
                       normalized=False, f_min=125)(x)
    x = amplitude_to_DB(x, multiplier=10, amin=1e-10, db_multiplier=10., top_db=80.)
    return x


def train_encoder():
    train_data = AudioSamplePairDataset(root_path='data/fma_small_samples', test=False, transform=transform_sample)
    test_data = AudioSamplePairDataset(root_path='data/fma_small_samples', test=False, transform=transform_sample)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=10)

    epochs = 1
    LR = 5e-3  # learning rate

    encoder = Encoder().cuda()

    summary(model=encoder, input_size=(1, 64, 64), device='cuda')

    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    loss_fn = torch.nn.TripletMarginLoss()

    print("Learning started")

    for epoch in range(epochs):
        epoch_losses = []
        for step, (x, y_pos, y_neg) in enumerate(train_loader):
            x, y_pos, y_neg = x.cuda(), y_pos.cuda(), y_neg.cuda()
            x_enc = encoder(x)
            y_pos_enc = encoder(y_pos)
            y_neg_enc = encoder(y_neg)
            loss_val = loss_fn(x_enc, y_pos_enc, y_neg_enc)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            epoch_losses.append(loss_val.item())
            print('    Batch {} of {} loss: {}'.format(step, len(train_data) // 128, loss_val.item()))

        print(f'Epoch: {epoch}  |  train loss: {np.mean(epoch_losses):.4f}')
    encoder.eval()
    return encoder
