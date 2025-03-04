"""Tests for the dataset module."""

import torch

import pytest


@pytest.mark.parametrize("in_channels", [3])
@pytest.mark.parametrize("latent_dim", [64])
def test_autoencoder(autoencoder):
    """Test the download_dataset function."""
    assert isinstance(autoencoder, torch.nn.Module)


@pytest.mark.parametrize("in_channels", [3])
@pytest.mark.parametrize("latent_dim", [64])
@pytest.mark.usefixtures("sparse_training")
def test_sparse_training(autoencoder):
    """Test the extract_dataset function."""
    assert autoencoder  # TODO: Add proper assertions
