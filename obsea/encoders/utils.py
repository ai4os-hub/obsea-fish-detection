"""Dataset subpackage for the OBSEA project."""

import logging
import os

import torch
from torch.utils.data import DataLoader

from obsea import config
from obsea.datasets.dataloaders import ImageDataset

logger = logging.getLogger(__name__)


def encode(autoencoder, paths, filename: str, transform=None):
    """Encode the images using the autoencoder and save the encodings."""

    logger.info("Preparing autoencoder model for inference")
    autoencoder.to(config.device)
    autoencoder.eval()

    logger.info("Creating dataset with images")
    dataset = ImageDataset(paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    logger.info("Encoding the dataset with the autoencoder")
    with torch.no_grad():
        encoded = [autoencoder.encoder(x) for x in dataloader]
    encoded = torch.concat(encoded, dim=0)

    logger.info("Saving the encoded data to %s", filename)
    save_encodings(encoded, filename)


def save_encodings(encoded, filename: str):
    """Save the encoded images and labels to the output directory."""
    os.makedirs(config.encoded_path, exist_ok=True)
    output_path = config.encoded_path / f"{filename}.pt"
    torch.save(encoded, output_path)
