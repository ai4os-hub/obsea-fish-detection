"""Expandable Seafloor Observatory (OBSEA) package."""

import dataclasses as dc
import logging
from importlib import resources

from ultralytics import YOLO
from ultralytics.engine.model import Model

from obsea import config

logger = logging.getLogger(__name__)


@dc.dataclass
class TrainSettings:
    """Training settings for the model."""

    epochs: int = 100
    batch: int = 8


def train(version, model_name=None, datasets_dir=None, settings=None):
    """
    Train the model using the specified dataset version.
    Note the dataset version should be extracted and ready to use.
    """

    # Set the default values for the function arguments
    datasets_dir = datasets_dir or config.DATASETS_DIR
    model = Model(model_name) if model_name else YOLO("yolov8n.yaml")
    settings = settings or TrainSettings()
    obsea_yaml = resources.files("obsea.data_files") / f"{version}.yml"

    # Train the model
    model.train(data=obsea_yaml, **settings.__dict__)
    logging.info("Model training completed successfully.")

    # Save the trained model
    model.save(f"yolov8_obsea_{version}.pt")
    logging.info("Model saved as yolov8_obsea_%s.pt.", version)
