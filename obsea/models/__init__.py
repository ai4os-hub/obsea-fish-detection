"""Subpackage for models used in the project."""

import torch.nn as nn
from ultralytics import YOLO
from torch.utils import data


class YOLOAutoencoder(nn.Module):
    """YOLO Autoencoder model."""

    def __init__(self, yolo_model: YOLO):
        super().__init__()
        self.encoder = YOLO(yolo_model).model.backbone
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass of the model."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ImageDataset(data.Dataset):
    """Dataset class for images."""

    def __init__(self, images_path, images):
        self.images_path = images_path
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_path = self.images_path / image
        return image_path
