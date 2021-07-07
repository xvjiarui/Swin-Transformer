#FROM nvidia/cuda:11.1-cudnn8-devel
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /workspace

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    g++ \
    wget \
    bzip2 \
    git \
    vim \
    tmux \
    htop \
    git \
    zip \
    unzip \
    ca-certificates \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    libglu1-mesa \
    libxext6 \
    libxtst6 \
    libxrender1 \
    libxi6 \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswresample-dev \
    less \
    groff

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl https://rclone.org/install.sh | bash

# Set timezone
RUN ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime

# Get Conda-ified Python.
#RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#    sh ~/miniconda.sh -b -p /opt/conda && \
#    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# Install pytorch
#RUN conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge -y

# Install deit related
#RUN conda install -c conda-forge yacs
#RUN conda install -c iopath iopath
#RUN conda install -c conda-forge -c iopath -c fvcore fvcore

# Install TIMM
RUN pip install timm==0.3.2

# Install einops
RUN pip install einops

# Install zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -t robbyrussell -p git

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# Install Apex
RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install WebDataset
RUN pip install -e git://github.com/tmbdev/webdataset.git#egg=webdataset

# Install Swin Dependences
RUN pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8

ENV WANDB_API_KEY=18a953cf069a567c46b1e613f940e6eb8f878c3d

ENV BITBUCKET_PWD=FymbWdVYdSnV4augTHbX

RUN pip install git+https://gitlab-master.nvidia.com/dl/gwe/gpu_affinity
