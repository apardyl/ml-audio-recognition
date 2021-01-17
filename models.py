import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    latent_dim = 2048

    small_fc_group_input = 32
    small_fc_group_output = 1
    small_fc_groups = (latent_dim // small_fc_group_input)
    small_embedding_dim = small_fc_groups * small_fc_group_output

    def __init__(self):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(4),
            nn.ELU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(8),
            nn.ELU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(1024),
            nn.ELU(inplace=True),
            nn.Conv2d(1024, self.latent_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.latent_dim),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )

        self.encoder_small_div_encode = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.small_fc_group_input, 16),
                nn.BatchNorm1d(16),
                nn.ELU(inplace=True),
                nn.Linear(16, self.small_fc_group_output),
                nn.Sigmoid(),
            ) for _ in range(self.small_fc_groups)])

    def conv(self, x: Tensor):
        return self.encoder_conv(x)

    def small_fingerprint(self, x: Tensor):
        groups = torch.split(x, self.small_fc_group_input, dim=1)
        groups_encoded = []
        for i in range(self.small_fc_groups):
            g = groups[i]
            groups_encoded.append(self.encoder_small_div_encode[i](g))
        return torch.cat(groups_encoded, dim=1)

    def forward(self, x: Tensor):
        conv = self.conv(x)
        return self.small_fingerprint(conv)


class LargeEncoder(nn.Module):
    latent_dim = 4096

    large_fc_group_input = 8
    large_fc_group_output = 2
    large_fc_groups = (latent_dim // large_fc_group_input)
    large_embedding_dim = large_fc_groups * large_fc_group_output

    def __init__(self):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(4),
            nn.ELU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(8),
            nn.ELU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(1024),
            nn.ELU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(2048),
            nn.ELU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(2048),
            nn.ELU(inplace=True),
            nn.Conv2d(2048, self.latent_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.latent_dim),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )

        self.encoder_large_div_encode = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.large_fc_group_input, 4),
                nn.BatchNorm1d(4),
                nn.ELU(inplace=True),
                nn.Linear(4, self.large_fc_group_output),
                nn.Sigmoid(),
            ) for _ in range(self.large_fc_groups)])

    def conv(self, x: Tensor):
        return self.encoder_conv(x)

    def large_fingerprint(self, x: Tensor):
        x = self.large_fc(x)
        groups = torch.split(x, self.large_fc_group_input, dim=1)
        groups_encoded = []
        for i in range(self.large_fc_groups):
            g = groups[i]
            groups_encoded.append(self.encoder_large_div_encode[i](g))
        return torch.cat(groups_encoded, dim=1)

    def forward(self, x: Tensor):
        conv = self.conv(x)
        return self.large_fingerprint(conv)
