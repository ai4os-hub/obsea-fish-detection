"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""

import datetime as dt
import logging
import os
import shutil

import drift_monitor as dw

from obsea.encoders.utils import load_encodings

from . import config, responses, schemas, utils

logger = logging.getLogger(__name__)
dw.register(accept_terms=True)


def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = {
            "author": config.api_metadata.get("authors"),
            "author-email": config.api_metadata.get("author-emails"),
            "description": config.api_metadata.get("summary"),
            "license": config.api_metadata.get("license"),
            "version": config.api_metadata.get("version"),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


def warm():
    """Function to run preparation phase before anything else can start.

    Raises:
        RuntimeError: Unexpected errors aim to stop model loading.
    """
    try:  # Warm up the detector with clean data
        logger.info("Warming up the detector with local data")
        clean = load_encodings(f"{config.data_version}_autoencoder_clean")
        utils.detector.fit(clean.cpu().numpy())  # Warm up with clean data
        for sample in clean[: utils.detector.window_size]:
            utils.detector.update(sample.cpu().numpy())
    except Exception as err:
        logger.error("Error when warming up: %s", err, exc_info=True)
        raise  # Reraise the exception after log
    try:  # Register the experiment in the drift monitor service
        logger.info("Registering in drift monitor service")
        description = config.api_metadata.get("summary")
        dw.new_experiment("obsea-camera", description, public=True)
    except ValueError:
        logger.info("Experiment already exists. Skipping creation.")
    except Exception as err:
        logger.error("Error registering: %s", err, exc_info=True)
        raise  # Reraise the exception after log


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(input_file, accept="application/json", **options):
    """Performs {model} prediction from given input data and parameters.

    Arguments:
        model_name -- Model name from registry to use for prediction values.
        input_file -- File with data to perform predictions from model.
        accept -- Response parser type, default is json.
        **options -- Arbitrary keyword arguments from PredArgsSchema.

    Options:
        batch_size -- Number of samples per batch.
        steps -- Steps before prediction round is finished.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values or files.
    """
    time = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    link = f"https://{config.store_url}?path={time}"
    if config.store_dir:  # If store configured, move to permanent storage
        image_path = f"{config.store_dir}/{time}/image.jpg"
        logger.debug("Saving image to store: %s", config.store_url)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        shutil.move(input_file.filename, image_path)
    else:  # If not, copy to a temporary location
        image_path = input_file.filename
    try:  # Load the image and encode it
        logger.debug("Loading image from input_file: %s", image_path)
        image = utils.load_image(image_path)
        normalized = utils.transform(image).to(config.device)
        encoded = utils.autoencoder.encoder(normalized.unsqueeze(0))[0]
    except Exception as err:
        logger.error("Error loading image: %s", err, exc_info=True)
        raise  # Reraise the exception after log
    try:  # Check if the image is clean
        logger.debug("Detecting drift with options: %s", options)
        v, tags = config.data_version, config.tags
        with dw.DriftMonitor("obsea-camera", v, tags) as monitor:
            result, _ = utils.detector.update(encoded.detach().cpu().numpy())
            warning = result.distance > options["warning_distance"]
            detected = result.distance > options["drift_distance"]
            parameters = {
                "distance": result.distance, "warning": warning,
                "warning_distance": options["warning_distance"],
                "drift_distance": options["drift_distance"],
                "link": link if config.store_url else None,
            } # fmt: skip
            monitor(detected, parameters)
    except Exception as err:
        logger.error("Error detecting drift: %s", err, exc_info=True)
        raise  # Reraise the exception after log
    logger.debug("Return results as format: %s", accept)
    return responses.content_types[accept](
        {
            "drift": detected, "parameters": parameters,
            "tags": tags, "version": v,
        } # fmt: skip
    )
