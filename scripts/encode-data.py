# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name

import datetime as dt
import logging
from typing import Literal

import torch
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from obsea import config, datasets, encoders, utils
from obsea.datasets import ImageDataset, transform

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
logger = logging.getLogger(__name__)


class Arguments(utils.BaseArguments):
    """
    This script provides a command-line interface to
    """  # Description for the script help message

    model_config = SettingsConfigDict(
        cli_prog_name=f"python -m {__package__}",
    )

    # General settings
    log_level: LogLevel = Field(
        default="INFO",
        description="Set the logging level.",
    )

    # Dataset settings
    version: str = Field(
        default=config.data_version,
        description="Dataset version to encode.",
    )

    # Model settings
    autoencoder: str = Field(
        default="autoencoder",
        description="Autoencoder to use for encoding.",
    )


def main(args: Arguments):
    """Main function for the dataset script."""
    logging.basicConfig(level=args.log_level, handlers=[RichHandler()])
    logger.info("Dataset script with log level: %s", args.log_level)

    logger.info("Loading autoencoder model from %s", args.autoencoder)
    autoencoder = utils.load_model(f"{args.autoencoder}.pt")
    autoencoder.to(config.device)
    autoencoder.eval()

    logger.info("Loading settings file for version %s", args.version)
    settings = utils.load_config(args.version)
    images_parent = datasets.images_path(args.version)

    logger.info("Loading images from %s", images_parent)
    clean_names = settings["camera_state"]["clean"]
    dirty_names = settings["camera_state"]["dirty"]
    image_names = clean_names + dirty_names
    image_paths = [images_parent / name for name in image_names]

    logger.info("Creating dataset with images")
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    logger.info("Encoding the dataset with the autoencoder")
    with torch.no_grad():
        encoded = [autoencoder.encoder(x) for x in dataloader]
    encoded = torch.concat(encoded, dim=0)

    # Generate the filename for the encoded data
    now = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{args.version}_{args.autoencoder}_{now}"

    logger.info("Saving the encoded data to %s", filename)
    encoders.save_encodings(encoded, filename)


if __name__ == "__main__":
    main(Arguments())
