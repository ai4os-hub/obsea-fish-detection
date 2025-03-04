"""Configuration module for the OBSEA project."""

import os

# Constant configuration settings
DATA_BASE_URL = "https://zenodo.org/records/13903520/files"
DATA_VERSION = os.getenv("DATA_VERSION", "v3.1")
DATA_TIMEOUT = 600  # 10 minutes
DATASETS_DIR = os.getenv("DATASETS_DIR", "datasets")


# Dynamic configuration settings
def datasets_path():
    """Return the path to the datasets directory."""
    return os.path.abspath(DATASETS_DIR)
