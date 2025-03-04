"""Fixtures for the tests."""

# pylint: disable=redefined-outer-name

import pytest
from torch.utils.data import DataLoader

from obsea import config, datasets, encoders
from obsea.datasets import ImageDataset, transform


@pytest.fixture()
def download_zip(zip_file, version):
    """Return the URL for the test dataset."""
    datasets.download(zip_file, version)


@pytest.fixture()
def extract_zip(zip_file):
    """Extract the test dataset."""
    datasets.extract(zip_file)


@pytest.fixture()
def dataset():
    """Return the test dataset."""
    image_paths = [f"tests/data/images/{i}.png" for i in range(1, 6)]
    return ImageDataset(image_paths, transform=transform)


@pytest.fixture()
def autoencoder(in_channels, latent_dim):
    """Return the autoencoder model."""
    return encoders.Autoencoder(in_channels, latent_dim)


@pytest.fixture()
def sparse_training(autoencoder, dataset):
    """Return the sparse training model."""
    return encoders.train_sparse_autoencoder(
        model=autoencoder,
        dataloader=DataLoader(dataset, 32),
        epochs=1,
        learning_rate=0.001,
        sparsity_weight=0.01,
        device=config.device,
    )
