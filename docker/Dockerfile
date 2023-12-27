ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu22.04

RUN apt-get update && apt-get install -y locales gcc g++ python3-dev
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    libprotobuf-dev \
    libprotobuf-c-dev

RUN pip3 install --upgrade pip
RUN pip3 install packaging

# Install torch
RUN CU=$(echo "${CUDA_VERSION%.*}" | sed 's/\.//g'); pip3 install torch --index-url "https://download.pytorch.org/whl/cu$CU"

# Install apex
RUN mkdir /setup
WORKDIR /setup
RUN git clone https://github.com/nvidia/apex
WORKDIR /setup/apex
RUN pip3 install ninja
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.6"
RUN pip3 install -v --no-build-isolation \
    --config-settings --global-option="--cpp_ext" \
    --config-settings --global-option="--cuda_ext" \
    --config-settings --global-option="--deprecated_fused_adam" \
    --global-option="--xentropy" \
    --global-option="--fast_multihead_attn" \
    ./

# Install flash-attention
RUN pip install flash-attn --no-build-isolation

# Install AutoAWQ
RUN pip install autoawq

COPY . /opennmt-py
WORKDIR /opennmt-py
RUN pip install -r requirements.opt.txt
RUN pip install -e .

WORKDIR /

ENTRYPOINT /bin/bash
