from torch import nn, Tensor
from torchsummary import summary
from torchvision import models


class AutoEncoder(nn.Module):

    def __init__(self, latent_dim=128):
        super().__init__()

        resnet = models.resnet18(pretrained=False)
        enc_hidden_1 = 1024
        self.latent_dim = latent_dim
        dec_hidden_1 = 1024
        dec_hidden_2 = 2048
        dec_conv_1 = 1024
        dec_conv_2 = 512
        dec_conv_3 = 256
        dec_conv_4 = 64
        dec_conv_5 = 16

        self.encoder = nn.Sequential(
            # resnet without last layer.
            *list(resnet.children())[:-1],
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, enc_hidden_1),
            nn.BatchNorm1d(enc_hidden_1),
            nn.ReLU(inplace=True),
            nn.Linear(enc_hidden_1, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden_1),
            nn.BatchNorm1d(dec_hidden_1),
            nn.ReLU(inplace=True),
            nn.Linear(dec_hidden_1, dec_hidden_2),
            nn.BatchNorm1d(dec_hidden_2),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(dec_hidden_2, 1, 1)),
            nn.Upsample(scale_factor=7),
            nn.Conv2d(dec_hidden_2, dec_conv_1, kernel_size=(1, 1), padding=0, stride=1),
            nn.BatchNorm2d(dec_conv_1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dec_conv_1, dec_conv_2, kernel_size=(5, 5), padding=2, stride=1),
            nn.BatchNorm2d(dec_conv_2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dec_conv_2, dec_conv_3, kernel_size=(5, 5), padding=2, stride=1),
            nn.BatchNorm2d(dec_conv_3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dec_conv_3, dec_conv_4, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(dec_conv_4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dec_conv_4, dec_conv_5, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(dec_conv_5),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dec_conv_5, 1, kernel_size=(1, 1), padding=0, stride=1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor):
        return self.encoder(x.repeat(1, 3, 1, 1))

    def decode(self, x: Tensor):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded
