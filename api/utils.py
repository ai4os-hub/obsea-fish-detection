"""Utilities module for API endpoints and methods."""

import logging
import sys
from functools import partial

from frouros.detectors.data_drift import MMDStreaming
from frouros.utils.kernels import rbf_kernel
from PIL import Image

import obsea

from . import config

logger = logging.getLogger(__name__)


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""
    def arguments_function():  # fmt: skip
        logger.debug("Web args schema: %s", schema)
        return schema().fields
    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def train_arguments(schema):
    """Decorator to inject schema as arguments to perform training."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_train_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def load_image(image_path):
    """Load an image from a given path."""
    return Image.open(image_path).convert("RGB")


# Load the detector and autoencoder model
detector = MMDStreaming(window_size=10, kernel=partial(rbf_kernel, sigma=0.5))
autoencoder = obsea.utils.load_model("autoencoder")

# Image loader transformer for the autoencoder
transform_settings = obsea.utils.load_config(config.data_version)
transform = obsea.datasets.get_transform(transform_settings)
