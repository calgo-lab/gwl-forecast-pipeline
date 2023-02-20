FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update -y && apt-get install -y \
    git \
    unrar \
    python3-pip \
    gdal-bin \
    libgdal-dev
RUN python3 -m pip install git+https://github.com/calgo-lab/gwl-forecast-pipeline
