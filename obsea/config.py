"""Configuration module for the OBSEA project."""

import os

# Standard configuration settings
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")


# Data configuration settings
DATA_VERSION = os.getenv("DATA_VERSION", "v3.1")
DATA_ZIP_FILE = f"obsea_dataset_{DATA_VERSION}.zip"
DATA_URL = f"https://zenodo.org/records/13903520/files/{DATA_ZIP_FILE}?download=1"
DATA_YAML = "data.yaml"
DATA_EXTRACT_DIR = "datasets"
DATA_DIR = f"obsea_dataset_{DATA_VERSION}"
DATA_TIMEOUT = 10

DATA_JSON_DIR = os.path.join(DATA_EXTRACT_DIR, DATA_DIR)
DATA_JSON_FILE = os.path.join(DATA_JSON_DIR, "obsea_dataset.json")
