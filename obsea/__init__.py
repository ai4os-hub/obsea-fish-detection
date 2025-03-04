"""Expandable Seafloor Observatory (OBSEA) package."""

import dataclasses as dc
import logging
import os
from importlib import resources

from torch import nn
from torch.optim import Adam, Optimizer
from ultralytics import YOLO
from ultralytics.engine.model import Model

from torch.utils import data

from obsea import config, models

logger = logging.getLogger(__name__)


@dc.dataclass
class TrainSettings:
    """Training settings for the model."""

    callbacks: list = dc.field(default_factory=list)
    epochs: int = 100
    batch: int = 8
    lr: float = 0.001


def train_yolo(version, model_name=None, settings=None):
    """
    Train the model using the specified dataset version.
    Note the dataset version should be extracted and ready to use.
    """

    # Set the default values for the function arguments
    model = Model(model_name) if model_name else YOLO("yolov8n.yaml")
    settings = settings or TrainSettings()
    obsea_yaml = resources.files("obsea.data_files") / f"{version}.yml"

    # Train the model
    model.train(data=obsea_yaml, **settings.__dict__)
    logging.info("Model training completed successfully.")

    # Save the trained model
    model.save(f"yolov8_obsea_{version}.pt")
    logging.info("Model saved as yolov8_obsea_%s.pt.", version)


def train_autoencoder(version, images=None, datasets_dir=None, settings=None):
    """Train the autoencoder model."""

    # Set the default values for the function arguments
    datasets_dir = datasets_dir or config.datasets_path()
    settings = settings or TrainSettings()
    images_path = datasets_dir / f"obsea_dataset_{version}" / "images"
    images = images or os.listdir(images_path)

    # Prepare dataloader with the images
    dataset = models.ImageDataset(images_path, images)
    dataloader = data.DataLoader(dataset, batch_size=settings.batch)

    # Generate model, optimizer, and criterion
    model = models.YOLOAutoencoder("yolov8n.yaml")
    optimizer = Adam(model.parameters(), lr=settings.lr)
    criterion = nn.MSELoss()

    # Train and save the model
    model.train(model, dataloader, optimizer, criterion, settings)
    model.save(f"autoencoder_{version}.pt")
    logger.info("Autoencoder training completed successfully.")


def train(model, dataloader, optimizer, criterion, settings=None):
    """Train the model using the specified dataloader and loss function."""

    # Set the default values and model to training
    settings = settings or TrainSettings()
    model.train()

    # Train the model
    for epoch in range(settings.epochs):
        running_loss = _train_cycle(model, dataloader, optimizer, criterion)
        average_loss = running_loss / len(dataloader)
        logger.info("Epoch %d: loss=%.4f", epoch, average_loss)
        for callback in settings.callbacks:  # Call the callback functions
            callback(model, epoch, running_loss)


def _train_cycle(model, dataloader, optimizer, criterion):
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss
