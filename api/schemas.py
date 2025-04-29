"""Module for defining custom web fields to use on the API interface."""

import marshmallow
from webargs import fields, validate

from . import responses


class PredArgsSchema(marshmallow.Schema):
    """Prediction arguments schema for api.predict function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    input_file = fields.Field(
        metadata={
            "description": "Image used to evaluate the data drift.",
            "type": "file",
            "location": "form",
        },
        required=True,
    )
    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(list(responses.content_types)),
    )
    warning_distance = fields.Float(
        metadata={
            "description": "Steps before prediction round is finished.",
        },
        load_default=0.100,
        validate=validate.Range(min=0.0),
    )
    drift_distance = fields.Float(
        metadata={
            "description": "Return format for method response.",
        },
        load_default=0.125,
        validate=validate.Range(min=0.0),
    )
