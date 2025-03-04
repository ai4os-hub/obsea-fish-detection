# Dockerfile may have following Arguments:
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .
#
# Be Aware! For the Jenkins CI/CD pipeline, 
# input args are defined inside the JenkinsConstants.groovy, not here!

# Base image, e.g. tensorflow/tensorflow:2.9.1
FROM ultralytics/ultralytics:8.3.61-python

LABEL maintainer='Borja Esteban Sanchis'
LABEL version='0.0.1'

# Set the working directory
WORKDIR /srv

# Create a user and group
RUN groupadd -r ai4os && useradd -r -g ai4os ai4os

# Copy the application to the working directory
COPY --chown=ai4os:ai4os . /srv

# Install the application as editable with the requirements
RUN pip install --no-cache -e .

# Copy updated pyproject.toml to include OBSEA authors and rename the module
# Re-install application with the updated pyproject.toml
RUN cd /srv/ai4os-yolov8-torch && \
    module=$(cat pyproject.toml |grep '\[project\]' -A1 |grep 'name' | cut -d'=' -f2 |tr -d ' ' |tr -d '"') && \
    pip uninstall -y $module
ENV MODEL_NAME="obsea_fish_detection"
COPY ./pyproject-child.toml /srv/ai4os-yolov8-torch/pyproject.toml
RUN cd /srv/ai4os-yolov8-torch && pip install --no-cache -e .

RUN mkdir -p /srv/ai4os-yolov8-torch/models/yolov8_obsea_xlarge/weights && \
    curl -L https://github.com/EnocMartinez/obsea-fish-detection/releases/download/model/12sp_1537img_xlarge_lr_0_000375_1920_best.pt \
    --output /srv/ai4os-yolov8-torch/models/yolov8_obsea_xlarge/weights/best.pt && \
    mkdir -p /srv/ai4os-yolov8-torch/models/yolov8_obsea_nano/weights && \
    curl -L https://github.com/EnocMartinez/obsea-fish-detection/releases/download/model/12sp_1537img_nano_lr_0_000375_1920_best.pt \
    --output /srv/ai4os-yolov8-torch/models/yolov8_obsea_nano/weights/best.pt
