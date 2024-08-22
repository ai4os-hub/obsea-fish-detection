# Dockerfile may have following Arguments:
# tag - tag for the Base image, (e.g. 2.9.1 for tensorflow)
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .
#
# Be Aware! For the Jenkins CI/CD pipeline, 
# input args are defined inside the JenkinsConstants.groovy, not here!

ARG tag=latest

# Base image, e.g. tensorflow/tensorflow:2.9.1
FROM ai4oshub/ai4os-yolov8-torch:${tag}

LABEL maintainer='Enoc Martinez, Oriol Prat Bayarri, Pol Banos Castello'
LABEL version='0.0.1'
# AI-based fish detection and classification algorithm based on YOLOv8. The model has been finetuned to detect and classify fish at the OBSEA underwater observatory.

# Download new model weights and remove old ones
# Download model from provisional server

# You can use the following as "reference" - https://github.com/ai4os-hub/ai4os-image-classification-tf/blob/master/Dockerfile
###############
### FILL ME ###
###############

# Define default YoloV8 models
ENV YOLOV8_DEFAULT_WEIGHTS="yolov8_obsea_nano,yolov8_obsea_xlarge"
ENV YOLOV8_DEFAULT_TASK_TYPE="det"

# Uninstall existing module ("yolov8_api")
# Update MODEL_NAME to obsea_fish_detection
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
