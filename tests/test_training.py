"""Tests for training functions."""

import os

import pytest


@pytest.mark.parametrize("version", ["v3.1"])
@pytest.mark.usefixtures("model_train")
def test_training(version):
    """Test the download_dataset function."""
    assert os.path.exists(f"yolov8_obsea_{version}.pt")
