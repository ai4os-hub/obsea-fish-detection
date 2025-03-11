# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name

import datetime as dt
import logging
from functools import partial
from typing import Literal

from frouros.callbacks.batch import PermutationTestDistanceBased
from frouros.detectors.data_drift import MMD
from frouros.utils.kernels import rbf_kernel
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from rich.logging import RichHandler

from obsea import config, encoders, utils

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
logger = logging.getLogger(__name__)


class Arguments(utils.BaseArguments):
    """
    This script provides a command-line interface to train a drift detector.
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
        default=f"ddetector-{dt.datetime.now().strftime('%Y%m%d%H%M')}",
        description="Output name for the drift detector.",
    )

    # Dataset settings
    encoded: str = Field(
        default=f"{config.data_version}_autoencoder_clean",
        description="Encoded torch object with training tensors.",
    )
    # Detector settings
    sigma: float = Field(
        default=0.5,
        description="Sigma value for the RBF kernel.",
    )
    permutations: int = Field(
        default=100,
        description="Number of permutations for the permutation test.",
    )


def main(args: Arguments):
    """Main function for the new detector script."""
    logging.basicConfig(level=args.log_level, handlers=[RichHandler()])
    logger.info("New detector script with log level: %s", args.log_level)

    logger.info("Loading encoded data from: %s", args.encoded)
    encoded_data = encoders.load_encodings(args.encoded)

    logger.info("Creating drift detector based on %s", MMD)
    detector = MMD(
        kernel=partial(rbf_kernel, sigma=args.sigma),
        callbacks=[PermutationTestDistanceBased(args.permutations)],
    )

    logger.info("Fitting drift detector with encoded data")
    training_info = detector.fit(X=encoded_data.cpu().numpy())
    logger.debug("Training info: %s", training_info)

    logger.info("Saving drift detector to: %s", args.output)
    utils.save_detector(detector, args.output)


if __name__ == "__main__":
    main(Arguments())
