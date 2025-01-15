"""Tests for the dataset module."""

import os

import pytest


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


@pytest.mark.parametrize("version", ["v3.1"])
@pytest.mark.usefixtures("gen_yaml")
def test_yaml(version):
    """Test the extract_dataset function."""
    assert os.path.exists(f"obsea_{version}.yml")
