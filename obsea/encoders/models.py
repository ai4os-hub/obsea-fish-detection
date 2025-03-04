"""Module containing the model classes for the drift analysis."""

# pylint: disable=too-few-public-methods


from obsea import config
from torch import Tensor, nn
from math import prod


class Encoder(nn.Module):
    """Encoder part of the autoencoder using convolutional layers."""

    def __init__(self, in_channels: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * prod(config.image_resize), latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder part of the autoencoder using convolutional layers."""

    def __init__(self, out_channels: int, latent_dim: int):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * prod(config.image_resize)),
            nn.ReLU(),
            nn.Unflatten(1, (32, *config.image_resize)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=3, padding=(1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class Autoencoder(nn.Module):
    """Autoencoder model for drift analysis using convolutional layers."""

    def __init__(self, in_channels: int, latent_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
