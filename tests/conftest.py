"""Fixtures for the tests."""

import pytest

import obsea
from obsea import dataset


@pytest.fixture()
def download_zip(zip_file, version):
    """Return the URL for the test dataset."""
    dataset.download(zip_file, version)


@pytest.fixture()
def extract_zip(zip_file):
    """Extract the test dataset."""
    dataset.extract(zip_file)


@pytest.fixture()
def gen_yaml(version):
    """Generates the ultralytics data file."""
    dataset.gen_yaml(version)


@pytest.fixture()
def model_train(version):
    """Train the model for ultralytic."""
    obsea.train(version)
