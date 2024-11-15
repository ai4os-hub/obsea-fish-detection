"""This script downloads the OBSEA dataset, trains a YOLOv8 encoder model
and saves the trained model as yolov8_obsea.pt.
"""

import os
import requests
import zipfile
import logging
from tqdm import tqdm
from ultralytics import YOLO
import json
import yaml

# Define constants
DATA_VERSION = os.getenv("DATA_VERSION", "v3.1")
ZIP_FILE = f"obsea_dataset_{DATA_VERSION}.zip"
URL = f"https://zenodo.org/records/13903520/files/{ZIP_FILE}?download=1"
YAML_FILE = "data.yaml"
EXTRACT_DIR = "datasets"
DATA_DIR = f"obsea_dataset_{DATA_VERSION}"
JSON_DIR = os.path.join(EXTRACT_DIR, DATA_DIR)
JSON_FILE = os.path.join(JSON_DIR, "obsea_dataset.json")

# Define the logger configuration
logging.basicConfig(level=logging.DEBUG)


def download_dataset(url, zip_file):
    """Download the dataset if it doesn't already exist."""
    if not os.path.exists(zip_file):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kilobyte
            with open(zip_file, "wb") as file, tqdm(
                desc="Downloading",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    file.write(data)
                    bar.update(len(data))
            print("Download completed successfully.")
        else:
            print(
                f"Failed to download the dataset. Status code: {response.status_code}"
            )
    else:
        print("Zip file already exists. Skipping download.")


def extract_dataset(zip_file, extract_dir, data_dir):
    """Extract the dataset if it hasn't already been extracted."""
    if not os.path.exists(f"{extract_dir}/{data_dir}"):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Decompression completed successfully.")
    else:
        print("Data already extracted. Skipping decompression.")


def generate_data_yaml(json_file, yaml_file, data_dir):
    """Generate the data.yaml file based on the obsea_dataset.json file."""
    with open(json_file, "r") as json_file:
        data = json.load(json_file)

    class_names = list(data.keys())

    data_yaml = {
        "train": os.path.join(data_dir, "images"),
        "val": os.path.join(data_dir, "images"),
        "nc": len(class_names),
        "names": class_names,
    }

    with open(yaml_file, "w") as yaml_file:
        yaml.dump(data_yaml, yaml_file, default_flow_style=False)

    print("data.yaml file generated successfully.")


def train_model(yaml_file):
    """Train the YOLOv8 model and save the trained model."""
    model = YOLO("yolov8n.yaml")  # Load a YOLOv8 model configuration
    model.train(data=yaml_file, epochs=100)  # Train the model
    model.save("yolov8_obsea.pt")  # Save the trained model
    print("Model training completed and saved as yolov8_obsea.pt.")


def main():
    download_dataset(URL, ZIP_FILE)
    extract_dataset(ZIP_FILE, EXTRACT_DIR, DATA_DIR)
    generate_data_yaml(JSON_FILE, YAML_FILE, DATA_DIR)
    train_model(YAML_FILE)


if __name__ == "__main__":
    main()
