"""Module for defining custom API response parsers and content types.
This module is used by the API server to convert the output of the requested
method into the desired format.

The module shows simple but efficient example functions. However, you may
need to modify them for your needs.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder:
    """Custom JSON Encoder for NumPy data types."""

    @staticmethod
    def decode(o):
        """Convert NumPy data types to standard Python types."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.bool):
            return o.item()
        if isinstance(o, dict):
            return {k: NumpyEncoder.decode(v) for k, v in o.items()}
        if isinstance(o, list):
            return [NumpyEncoder.decode(i) for i in o]
        if isinstance(o, tuple):
            return tuple(NumpyEncoder.decode(i) for i in o)
        return o


def json_response(data):
    """Convert the data to JSON format."""
    logger.debug("Response result: %s", data)
    return NumpyEncoder.decode(data)


content_types = {
    "application/json": json_response,
}
