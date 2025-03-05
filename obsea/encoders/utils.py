"""Dataset subpackage for the OBSEA project."""

import logging
import os
import torch

from obsea import config

logger = logging.getLogger(__name__)


def save_encodings(encoded, filename: str = None):
    """Save the encoded images and labels to the output directory."""
    os.makedirs(config.encoded_path, exist_ok=True)
    output_path = config.encoded_path / f"{filename}.pt"
    torch.save(encoded, output_path)
