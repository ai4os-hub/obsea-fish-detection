"""Configuration module for the OBSEA project."""

import os

# Data configuration settings
DATA_BASE_URL = "https://zenodo.org/records/13903520/files"
DATA_VERSION = os.getenv("DATA_VERSION", "v3.1")
DATA_TIMEOUT = 600  # 10 minutes
DATASETS_DIR = "datasets"

# DATA_JSON_DIR = os.path.join(DATA_EXTRACT_DIR, DATA_DIR)
# DATA_JSON_FILE = os.path.join(DATA_JSON_DIR, "obsea_dataset.json")
