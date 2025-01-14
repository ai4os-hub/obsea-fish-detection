"""Dataset subpackage for the OBSEA project."""

import logging
import os

import requests
from tqdm import tqdm

from obsea import config

logger = logging.getLogger(__name__)


def download_dataset(url=None, zip_file=None):
    """Download the dataset if it doesn't already exist."""

    # Set the default values for the URL and zip file
    url = url or config.DATA_URL
    zip_file = zip_file or config.DATA_ZIP_FILE

    # Check if the zip file already exists
    if os.path.exists(zip_file):
        logging.info("Zip file already exists. Skipping download.")
        return

    # Download the dataset
    response = requests.get(url, stream=True, timeout=config.DATA_TIMEOUT)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download the dataset from {url}. \n"
            + f"Status code: {response.status_code}"
        )
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte

    # Download the file with a progress bar
    with (
        open(zip_file, "wb") as file,
        tqdm(
            desc="Downloading",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
    logger.info("Download completed successfully.")
