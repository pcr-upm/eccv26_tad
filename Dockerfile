# This is our first build stage, it will not persist in the final image
FROM ubuntu as intermediate
RUN apt-get -y update && apt-get install -y git
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN chmod 400 /root/.ssh/id_rsa
# Make sure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
# Download the computer vision framework
RUN git clone git@github.com:pcr-upm/eccv26_tad.git eccv26_tad
ADD data /eccv26_tad/data

# Copy the repository from the previous image
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
USER 0
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt-get update && apt-get install ffmpeg libsm6 libxext6 build-essential git wget software-properties-common libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev libavdevice-dev -y
RUN mkdir /home/username
WORKDIR /home/username
COPY --from=intermediate /eccv26_tad /home/username/eccv26_tad
LABEL maintainer="roberto.valle@upm.es"
# Setup conda environment
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/username/miniconda.sh
RUN chmod +x /home/username/miniconda.sh
RUN /home/username/miniconda.sh -b -p /home/username/conda
RUN /home/username/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /home/username/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /home/username/conda/bin/conda update -n base -c conda-forge conda 
RUN /home/username/conda/bin/conda install -n base conda-libmamba-solver 
RUN /home/username/conda/bin/conda config --set solver libmamba 
RUN /home/username/conda/bin/conda create -n eccv26 python=3.10.12
# Activate conda environment
ENV PATH /home/username/conda/envs/eccv26/bin:/home/username/conda/bin:$PATH
# Make RUN commands use the new environment (source activate eccv26)
SHELL ["conda", "run", "-n", "eccv26", "/bin/bash", "-c"]
# Install dependencies
RUN conda run -n eccv26 pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url=https://download.pytorch.org/whl/cu118 -U && conda run -n eccv26 pip install --no-cache-dir deepspeed
RUN conda run -n eccv26 pip install --no-cache-dir openmim && conda run -n eccv26 mim install mmcv==2.0.1 && conda run -n eccv26 mim install mmaction2==1.1.0 
# Install flash-attn
RUN conda run -n eccv26 pip install --no-cache-dir flash-attn==2.5.4 --no-build-isolation
# Build and install flash-attention layer_norm extension
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.5.4 && \
    cd csrc/layer_norm && \
    MAX_JOBS=16 conda run -n eccv26 pip install . --no-build-isolation
RUN conda run -n eccv26 pip install --no-cache-dir images-framework torchinfo timm==1.0.24 --no-build-isolation
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/local/cuda/libnvcuvid.so