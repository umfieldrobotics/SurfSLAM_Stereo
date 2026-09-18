ARG BASE_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}


# Prevent anything requiring user input
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Basic packages
RUN apt-get -y update \
    && apt-get -y install \
    python3-pip \
    sudo \
    vim \
    wget \
    curl \
    software-properties-common \
    doxygen \
    git \
    tmux \
    g++ \
    gcc \
    build-essential \
    checkinstall \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get -y update \
    && apt-get -y install \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update \
    && apt-get -y install \ 
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
                torch==2.7.0 \
                torchvision==0.22.0 \
                torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN pip3 install \
        scikit-image \
        omegaconf \
        opencv-contrib-python \
        imgaug \
        Ninja \
        timm \
        albumentations \
        nodejs \
        jupyterlab \
        scipy \
        joblib \
        scikit-learn \
        ruamel.yaml \
        trimesh \
        pyyaml \
        imageio \
        open3d \
        transformations \
        einops \
        gdown \
        tensorboard &&\
    pip3 install flash-attn --no-build-isolation 

# Extra misc installs
RUN apt-get -y update \
    && sudo apt-get -y install \ 
    libomp-dev \
    mesa-utils \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*  
RUN apt-get -y update \
    && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*  

RUN pip3 install \
    huggingface_hub \
    opt_einsum

RUN pip3 install --upgrade timm
