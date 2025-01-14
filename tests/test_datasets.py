"""Tests for the dataset module."""

import os

import pytest

from obsea import dataset


@pytest.fixture()
def download_zip(zip_file, version):
    """Return the URL for the test dataset."""
    dataset.download(zip_file, version)


@pytest.fixture()
def extract_zip(zip_file):
    """Extract the test dataset."""
    dataset.extract(zip_file)


@pytest.mark.parametrize("zip_file", ["obsea.zip"])
@pytest.mark.parametrize("version", ["v3.1"])
@pytest.mark.usefixtures("download_zip")
def test_download(zip_file):
    """Test the download_dataset function."""
    assert os.path.exists(zip_file)


@pytest.mark.parametrize("zip_file", ["obsea.zip"])
@pytest.mark.parametrize("version", ["v3.1"])
@pytest.mark.usefixtures("extract_zip")
def test_extract(version):
    """Test the extract_dataset function."""
    assert os.path.exists(f"datasets/obsea_dataset_{version}")
