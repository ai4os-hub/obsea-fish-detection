from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
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
        return image


transform = transforms.Compose(
    [
        transforms.Resize(config.image_resize),
        transforms.ToTensor(),
    ]
)
