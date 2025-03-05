"""Module containing the model classes for the drift analysis."""

# pylint: disable=too-few-public-methods
# pylint: disable=line-too-long

from typing import Tuple

from torch import Tensor, nn

from obsea import config


class Encoder(nn.Module):
    """Encoder part of the autoencoder using convolutional layers."""

    def __init__(self, in_shape: Tuple[int, int, int], latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (in_shape[1] // 4) * (in_shape[2] // 4), latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the encoder."""
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder part of the autoencoder using convolutional layers."""

    def __init__(self, out_shape: Tuple[int, int, int], latent_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * (out_shape[1] // 4) * (out_shape[2] // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (32, out_shape[1] // 4, out_shape[2] // 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_shape[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the decoder."""
        return self.decoder(x)


class Autoencoder(nn.Module):
    """Autoencoder model for drift analysis using convolutional layers."""

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass of the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
