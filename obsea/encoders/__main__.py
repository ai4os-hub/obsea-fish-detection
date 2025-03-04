# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name

import logging
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from obsea import config, datasets, encoders, utils
from obsea.datasets import ImageDataset, transform
from obsea.encoders import Autoencoder

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
    output: str = Field(
        default="autoencoder.pt",
        description="Output name for.",
    )

    # Dataset settings
    version: str = Field(
        default=config.data_version,
        description="Dataset version to use for training.",
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for the dataloader.",
    )
    shuffle: bool = Field(
        default=True,
        description="Shuffle the dataset during training.",
    )

    # Encoder settings
    latent_dim: int = Field(
        default=64,
        description="Dimension of the latent space.",
    )

    # Training settings
    epochs: int = Field(
        default=50,
        description="Number of training epochs.",
    )
    learning_rate: float = Field(
        default=1e-3,
        description="Learning rate for the optimizer.",
    )
    sparsity_weight: float = Field(
        default=1e-5,
        description="Weight for the sparsity loss.",
    )


def main(args: Arguments):
    """Main function for the dataset script."""
    logging.basicConfig(level=args.log_level, handlers=[RichHandler()])
    logger.info("Dataset script with log level: %s", args.log_level)

    logger.info("Loading settings file for version %s", args.version)
    settings = utils.load_config(args.version)
    images_parent = datasets.images_path(args.version)
    in_channels = settings["image"]["n_channels"]
    image_names = settings["camera_state"]["clean"]
    image_paths = [images_parent / name for name in image_names]

    logger.info("Creating dataset with images")
    dataset = ImageDataset(image_paths, transform=transform)

    logger.info("Creating the autoencoder model")
    model = Autoencoder(in_channels, args.latent_dim)

    logger.info("Training the autoencoder")
    encoders.train_sparse_autoencoder(
        model=model,
        dataloader=DataLoader(dataset, args.batch_size, args.shuffle),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        sparsity_weight=args.sparsity_weight,
        device=config.device,
    )

    logger.info("Saving the trained model as %s", args.output)
    utils.save_model(model, args.output)


if __name__ == "__main__":
    main(Arguments())
