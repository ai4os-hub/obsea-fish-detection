"""Expandable Seafloor Observatory (OBSEA) package."""

import dataclasses as dc
import logging

from ultralytics import YOLO
from ultralytics.engine.model import Model

from obsea import config

logger = logging.getLogger(__name__)


@dc.dataclass
class TrainSettings:
    """Training settings for the model."""

    epochs: int = 100
    batch: int = 8
    half: bool = True


def train(version, model_name=None, datasets_dir=None, settings=None):
    """
    Train the model using the specified dataset version.
    Note the dataset version should be extracted and ready to use.
    """

    # Set the default values for the function arguments
    datasets_dir = datasets_dir or config.DATASETS_DIR
    model = Model(model_name) or YOLO("yolov8n.yaml")
    data = f"{datasets_dir}/obsea_dataset_{version}/obsea.yml"
    settings = settings or TrainSettings()

    # Train the model
    model.train(data, **settings.__dict__)
    logging.info("Model training completed successfully.")

    # Save the trained model
    model.save(f"yolov8_obsea_{version}.pt")
    logging.info("Model saved as yolov8_obsea_%s.pt.", version)
