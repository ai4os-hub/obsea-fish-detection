"""Dataset subpackage for the OBSEA project."""

import json
import logging
import os
import zipfile

import requests
import yaml
from tqdm import tqdm

from obsea import config

logger = logging.getLogger(__name__)


def data_url(version, base_url=None):
    """Return the URL for the specified dataset version."""

    # Set the default values for the function arguments
    base_url = base_url or config.DATA_BASE_URL

    # Return the URL for the dataset version
    return f"{base_url}/obsea_dataset_{version}.zip?download=1"


def download(zip_file, version=None):
    """Download the dataset if it doesn't already exist."""

    # Set the default values for the URL and zip file
    url = data_url(version or config.DATA_VERSION)

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
        ) as progress_bar,
    ):
        for data in response.iter_content(block_size):
            file.write(data)
            progress_bar.update(len(data))
    logger.info("Download completed successfully.")


def extract(zip_file, datasets_dir=None):
    """Extract the dataset if it hasn't already been extracted."""

    # Set the default values for the function arguments
    extract_dir = datasets_dir or config.DATASETS_DIR

    # Extract the dataset
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info("Decompression completed successfully.")


def gen_yaml(version, datasets_dir=None):
    """Generate the data.yaml file based on the obsea_dataset.json file."""

    # Set the default values for the function arguments
    datasets_dir = datasets_dir or config.DATASETS_DIR
    class_file = f"{datasets_dir}/obsea_dataset_{version}/obsea_dataset.json"
    data_dir = f"{datasets_dir}/obsea_dataset_{version}/images"
    yaml_file = f"{datasets_dir}/obsea_dataset_{version}/obsea.yml"

    # Load the JSON file and extract the class names
    with open(class_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    class_names = list(data.keys())

    # Generate the YOLO data file for the dataset
    yolo_yaml = {
        "train": os.path.join(data_dir, "images"),
        "val": os.path.join(data_dir, "images"),
        "nc": len(class_names),
        "names": class_names,
    }

    # Write the YOLO data file to disk
    with open(yaml_file, "w", encoding="utf-8") as file:
        yaml.dump(yolo_yaml, file, default_flow_style=False)
    logger.info("obsea.yml file generated successfully.")
