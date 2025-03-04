"""Configuration module for the OBSEA project."""

import os
from importlib import resources
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import torch

# Installation configuration settings
install_path: Path = Path(resources.files("obsea"))

# Constant configuration settings
DATABASE_URL = "https://zenodo.org/records/13903520/files"
database_url = urlparse(DATABASE_URL)

DATA_VERSION = os.getenv("DATA_VERSION", "v3.1")
data_version: str = DATA_VERSION

DATA_TIMEOUT = os.getenv("DATA_TIMEOUT", "600")
data_timeout: int = int(DATA_TIMEOUT)

DATASETS_DIR = os.getenv("DATASETS_DIR", "datasets")
datasets_path: Path = Path(DATASETS_DIR)

MODELS_DIR = os.getenv("MODELS_DIR", "models")
models_path: Path = Path(MODELS_DIR)

# Device configuration settings
DEVICE = os.getenv("DEVICE", "cuda")
device: str = DEVICE if torch.cuda.is_available() else "cpu"

# Configuration settings for the application
CONFIG_PATH = os.getenv("CONFIG_PATH", f"{install_path}/config-files")
config_path: Path = Path(CONFIG_PATH)

# Image transformation settings
IMAGE_RESIZE_X = os.getenv("IMAGE_RESIZE", "192")
IMAGE_RESIZE_Y = os.getenv("IMAGE_RESIZE", "108")
image_resize: Tuple[int, int] = (int(IMAGE_RESIZE_Y), int(IMAGE_RESIZE_X))
