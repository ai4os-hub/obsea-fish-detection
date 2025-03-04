# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name

import logging
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from rich.logging import RichHandler

from obsea import config, datasets, utils

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
        description="Dataset version to download.",
    )


def main(args: Arguments):
    """Main function for the dataset script."""

    logging.basicConfig(level=args.log_level, handlers=[RichHandler()])
    logger.info("Dataset script with log level: %s", args.log_level)

    logger.info("Downloading dataset version: %s", args.version)
    zip_file = config.datasets_path / f"obsea{args.version}.zip"
    datasets.download(zip_file, args.version)

    logger.info("Extracting dataset version: %s", args.version)
    datasets.extract(zip_file, config.datasets_path)


if __name__ == "__main__":
    main(Arguments())
