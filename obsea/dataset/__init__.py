"""Dataset subpackage for the OBSEA project."""

import json
import logging
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import requests
import yaml
from tqdm import tqdm

from obsea import config

logger = logging.getLogger(__name__)


def data_url(version, base_url=None):
    """Return the URL for the specified dataset version."""

    # Set the default values for the function arguments
    base_url = base_url or config.DATA_BASE_URL

    # Return the URL for the dataset version
    return f"{base_url}/obsea_dataset_{version}.zip?download=1"


def download(zip_file, version=None):
    """Download the dataset if it doesn't already exist."""

    # Set the default values for the URL and zip file
    url = data_url(version or config.DATA_VERSION)

    # Check if the zip file already exists
    if os.path.exists(zip_file):
        logging.info("Zip file already exists. Skipping download.")
        return

    # Download the dataset
    response = requests.get(url, stream=True, timeout=config.DATA_TIMEOUT)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download the dataset from {url}. \n"
            + f"Status code: {response.status_code}"
        )
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte

    # Download the file with a progress bar
    with (
        open(zip_file, "wb") as file,
        tqdm(
            desc="Downloading",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar,
    ):
        for data in response.iter_content(block_size):
            file.write(data)
            progress_bar.update(len(data))
    logger.info("Download completed successfully.")


def extract(zip_file, datasets_dir=None):
    """Extract the dataset if it hasn't already been extracted."""

    # Set the default values for the function arguments
    extract_dir = datasets_dir or config.DATASETS_DIR

    # Extract the dataset
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info("Decompression completed successfully.")


def gen_yaml(version, datasets_dir=None):
    """Generate the data.yaml file based on the obsea_dataset.json file."""

    # Set the default values for the function arguments
    datasets_dir = datasets_dir or config.DATASETS_DIR
    class_file = f"{datasets_dir}/obsea_dataset_{version}/obsea_dataset.json"

    # Load the JSON file and extract the class names
    with open(class_file, "r", encoding="utf-8") as file:
        class_info = json.load(file)

    # Generate the YOLO data file for the dataset
    yolo_yaml = {
        "path": f"../{datasets_dir}",
        "train": f"obsea_dataset_{version}",
        "val": f"obsea_dataset_{version}",
        "nc": len(class_info),
        "names": dict(enumerate(class_info.keys())),
    }

    # Write the YOLO data file to disk
    with open(f"obsea_{version}.yml", "w", encoding="utf-8") as file:
        yaml.dump(yolo_yaml, file, default_flow_style=False)
    logger.info("obsea.yml file generated successfully.")


def voc_to_yolo(version, datasets_dir=None):
    """Convert the VOC annotations to YOLO format."""

    # Set the default values for the function arguments
    datasets_dir = datasets_dir or config.DATASETS_DIR
    class_file = f"{datasets_dir}/obsea_dataset_{version}/obsea_dataset.json"
    labels_dir = f"{datasets_dir}/obsea_dataset_{version}/labels"

    # Load the JSON file and extract the class names
    with open(class_file, "r", encoding="utf-8") as file:
        classes = list(json.load(file))

    # Create temporary directories for the YOLO annotations
    temp_dir = tempfile.mkdtemp()

    # Convert the VOC annotations to YOLO format
    for xml_file in os.listdir(labels_dir):
        _voc_to_yolo(xml_file, labels_dir, temp_dir, classes)

    # Replace labels in the VOC annotations with YOLO labels
    shutil.rmtree(labels_dir)
    shutil.move(temp_dir, labels_dir)

    # Log the completion of the conversion
    logger.info("VOC to YOLO conversion completed successfully.")


def _voc_to_yolo(xml_file, labels_dir, out_dir, classes):
    """Convert one VOC file annotation to YOLO format."""

    # Check if the file is a valid XML file
    if not xml_file.endswith(".xml"):
        raise ValueError(f"Invalid file format: {xml_file}")

    # Parse the XML file and extract the annotations
    tree = ET.parse(os.path.join(labels_dir, xml_file))
    root = tree.getroot()

    # Extract the image dimensions
    image_dims = {
        "image_width": int(root.find("size/width").text),
        "image_height": int(root.find("size/height").text),
    }

    # Prepare the YOLO annotations for the image
    yolo_annotations = []

    # Loop object and append annotations in YOLO format
    for obj in root.findall("object"):
        _annotations = _yolo_annotation(obj, classes, **image_dims)
        yolo_annotations.append(" ".join(map(str, _annotations)))

    yolo_file = os.path.join(out_dir, os.path.splitext(xml_file)[0] + ".txt")
    with open(yolo_file, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_annotations))


def _yolo_annotation(obj, classes, image_width, image_height):
    """Extract the VOC annotations for the specified classes."""

    # Extract the class name and ID
    class_name = obj.find("name").text
    if class_name not in classes:
        raise ValueError(f"Invalid class name: {class_name}")
    class_id = classes.index(class_name)

    # Extract the bounding box coordinates
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    # Normalize the bounding box coordinates
    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    # Return the annotations in YOLO format
    return class_id, x_center, y_center, width, height
