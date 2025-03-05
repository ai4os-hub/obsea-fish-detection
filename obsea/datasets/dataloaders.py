"""Module to create dataloaders for the datasets."""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from obsea import config


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image.to(config.device)


def get_transform(settings):
    """Get the transform for the dataset."""
    return transforms.Compose(
        [
            transforms.Resize(settings["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings["mean"], std=settings["std"]),
        ]
    )
