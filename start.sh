#!/usr/bin/env bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate eccv26

pip install -r requirements.txt --no-build-isolation
mim install "mmpose>=1.1.0" --no-build-isolation
rm -rf decord
git clone --recursive https://github.com/dmlc/decord ./decord && \
    cd ./decord && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    pip install . --no-build-isolation
cd ../..
rm -rf decord

# Install OpenTAD custom ops
pip install -e . --no-build-isolation