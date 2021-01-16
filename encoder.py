import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.latent_dim = 1024
        self.fc_group_input = 16
        self.embedding_dim = self.latent_dim // self.fc_group_input

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(2),
            nn.ELU(inplace=True),
            nn.Conv2d(2, 4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(4),
            nn.ELU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
            nn.Conv2d(512, self.latent_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.latent_dim),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )

        self.encoder_div_encode = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fc_group_input, 10),
                nn.BatchNorm1d(10),
                nn.ELU(inplace=True),
                nn.Linear(10, 1),
            ) for _ in range(self.embedding_dim)])

    def forward(self, x: Tensor):
        conv = self.encoder_conv(x)
        groups = torch.split(conv, self.fc_group_input, dim=1)
        groups_encoded = []
        for i in range(self.embedding_dim):
            g = groups[i]
            groups_encoded.append(self.encoder_div_encode[i](g))
        concat = torch.stack(groups_encoded, dim=1).squeeze(-1)
        return concat
