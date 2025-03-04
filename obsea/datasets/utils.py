"""Dataset subpackage for the OBSEA project."""

import logging
import os
import zipfile

import requests
from tqdm import tqdm

from obsea import config

logger = logging.getLogger(__name__)


def download(zip_file, version=None):
    """Download the dataset if it doesn't already exist."""

    # Set the default values for the URL and zip file
    url = _data_url(version or config.data_version)

    # Check if the zip file already exists
    if os.path.exists(zip_file):
        logging.info("Zip file already exists. Skipping download.")
        return

    # Download the dataset
    response = requests.get(url, stream=True, timeout=config.data_timeout)
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
        ) as progress_bar,
    ):
        for data in response.iter_content(block_size):
            file.write(data)
            progress_bar.update(len(data))
    logger.info("Download completed successfully.")


def extract(zip_file, datasets_dir=None):
    """Extract the dataset if it hasn't already been extracted."""

    # Set the default values for the function arguments
    extract_dir = datasets_dir or config.datasets_path

    # Extract the dataset
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info("Decompression completed successfully.")


def images_path(version):
    """Return the path to the images directory."""
    path = config.datasets_path / f"obsea_dataset_{version}"
    return path / "images"


def _data_url(version, base_url=None):
    base_url = base_url or config.database_url.geturl()
    return f"{base_url}/obsea_dataset_{version}.zip?download=1"
