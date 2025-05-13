"""Module to define CONSTANTS used across the DEEPaaS Interface.

This module is used to define CONSTANTS used across the API interface.
Do not misuse this module to define variables that are not CONSTANTS or
that are not used across the `api` package. You can use the `config.py`
file on your model package to define CONSTANTS related to your model.

By convention, the CONSTANTS defined in this module are in UPPER_CASE.
"""

import os
from importlib import metadata

from obsea.config import data_version, device

# Ensure that your model package has a config.py file with the following
# pylint: disable=unused-import

# Get AI model metadata
API_NAME = "obsea-drift-detection"
api_metadata = metadata.metadata(API_NAME)

# Fix metadata for emails from pyproject parsing
_emails = api_metadata["Author-email"].split(", ")
_emails = map(lambda s: s[:-1].split(" <"), _emails)
api_metadata["Author-emails"] = dict(_emails)

# Fix metadata for authors from pyproject parsing
_authors = api_metadata.get("Author", "").split(", ")
_authors = [] if _authors == [""] else _authors
_authors += api_metadata["Author-emails"].keys()
api_metadata["Authors"] = sorted(_authors)

# MyToken configuration
my_token = os.getenv("DRIFT_MONITOR_MYTOKEN", None)
if my_token is None:
    err = "Please set the environment variable DRIFT_MONITOR_MYTOKEN"
    raise RuntimeError(err)

# DriftWatch configuration
tags = ["data-drift", "DDM", "example_service", "streaming"]
driftwatch = os.getenv("DRIFT_MONITOR_URL", None)
if driftwatch is None:
    err = "Please set the environment variable DRIFT_MONITOR_URL"
    raise RuntimeError(err)

# Store for uploaded images
store_dir = os.getenv("DRIFT_MONITOR_STORE_DIR", None)
store_url = os.getenv("DRIFT_MONITOR_STORE_URL", None)
