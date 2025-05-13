# Dockerfile may have following Arguments:
# tag - tag for the Base image, (e.g. 2.9.1 for tensorflow)
# branch - user repository branch to clone, i.e. test (default: master)
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .
#
# [!] Note: For the Jenkins CI/CD pipeline, input args are defined inside the
# Jenkinsfile, not here!

ARG tag=2.7.0-cuda11.8-cudnn9-runtime

# Base image, e.g. pytorch/pytorch:2.x.x-cuda...
FROM pytorch/pytorch:${tag}

LABEL maintainer='Borja Esteban'
LABEL version='1.0.0'

# What user branch to clone [!]
ARG branch=drift-camera

# Drift Watch arguments
ARG MONITOR_URL=https://drift-watch.dev.ai4eosc.eu
ARG MYTOKEN

# Install Ubuntu packages
# - gcc is needed in Pytorch images because deepaas installation might break otherwise (see docs)
#   (it is already installed in tensorflow images)
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Update python packages
# [!] Remember: DEEP API V2 only works with python>=3.6
RUN python3 --version && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Set LANG environment
ENV LANG=C.UTF-8

# Set the working directory and create the storage folder
WORKDIR /srv
RUN mkdir /storage/camera-images

# EXPERIMENTAL: install deep-start script
# N.B.: This repository also contains run_jupyter.sh
RUN git clone https://github.com/deephdc/deep-start /srv/.deep-start && \
    ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start && \
    ln -s /srv/.deep-start/run_jupyter.sh /usr/local/bin/run_jupyter

# Install JupyterLab
ENV JUPYTER_CONFIG_DIR=/srv/.deep-start/
# Necessary for the Jupyter Lab terminal
ENV SHELL=/bin/bash
RUN pip3 install --no-cache-dir jupyterlab

# Install user app
RUN git clone --depth 1 -b $branch https://github.com/ai4os-hub/obsea-fish-detection.git && \
    pip3 install --no-cache-dir -e ./obsea-fish-detection

# Open ports: DEEPaaS (5000), Monitoring (6006), Jupyter (8888)
EXPOSE 5000 6006 8888

# Define environment variables to track models and encoded images
ENV DATASETS_DIR=/srv/obsea-fish-detection/datasets
ENV MODELS_DIR=/srv/obsea-fish-detection/models
ENV ENCODED_DIR=/srv/obsea-fish-detection/encoded

# Define location for API interface
ENV DRIFT_MONITOR_STORE=/storage/camera-images
ENV DRIFT_MONITOR_URL=${MONITOR_URL}
ENV DRIFT_MONITOR_MYTOKEN=${MYTOKEN}

# Launch deepaas
ENTRYPOINT [ "deep-start" ]
CMD ["--deepaas"]
