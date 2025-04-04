"""Module for defining custom API response parsers and content types.
This module is used by the API server to convert the output of the requested
method into the desired format.

The module shows simple but efficient example functions. However, you may
need to modify them for your needs.
"""

import logging

logger = logging.getLogger(__name__)
content_types = {"application/json": None}
