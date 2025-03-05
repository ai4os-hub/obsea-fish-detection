# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name

import logging
from typing import Literal

import torch
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from obsea import config, datasets, encoders, utils
from obsea.datasets import ImageDataset, get_transform

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

    logger.info("Loading settings file for version %s", args.version)
    settings = utils.load_config(args.version)
    images_parent = datasets.images_path(args.version)
    transform = get_transform(settings["transform"])

    logger.info("Loading clean images from %s", images_parent)
    names = settings["camera_state"]["clean"]
    paths = [images_parent / name for name in names]

    filename = f"{args.version}_{args.autoencoder}_clean"
    logger.info("Encoding clean images to %s", filename)
    encoders.encode(autoencoder, paths, filename, transform)

    logger.info("Loading dirty images from %s", images_parent)
    names = settings["camera_state"]["dirty"]
    paths = [images_parent / name for name in names]

    filename = f"{args.version}_{args.autoencoder}_dirty"
    logger.info("Encoding dirty images to %s", filename)
    encoders.encode(autoencoder, paths, filename, transform)


if __name__ == "__main__":
    main(Arguments())
