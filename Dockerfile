# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install basic tools
RUN apt-get update && apt-get install -y git tree ffmpeg wget
RUN rm /bin/sh && ln -s /bin/bash /bin/sh && ln -s /lib64/libcuda.so.1 /lib64/libcuda.so
RUN apt-get -y update && apt-get -y install build-essential cmake ninja-build libgl1-mesa-dev ffmpeg

COPY ./requirements_docker.txt /requirements_docker.txt
COPY ./scripts/test_environment.py /scripts/test_environment.py

RUN pip install --upgrade pip && pip install cmake ninja

# Install cosmos-predict1 dependencies. This will take a while.
RUN echo "Installing dependencies. This will take a while..." && \
    echo "Original PyTorch version: $(python -c 'import torch; print(torch.__version__)')" && \
    echo "Preserving original PyTorch version from base container..." && \
    pip install --no-cache-dir --upgrade-strategy only-if-needed -r /requirements_docker.txt && \
    echo "Done installing base dependencies"

# Install transformer-engine 1.12.0 with proper library path setup
RUN echo "Installing transformer-engine 1.12.0 with library path fix..." && \
    pip uninstall -y transformer-engine && \
    pip install --no-cache-dir --no-build-isolation transformer-engine[pytorch]==1.12.0 && \
    ldconfig && \
    echo "Transformer-engine 1.12.0 installed"

# Set library paths as environment variables
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib/python3.10/dist-packages/transformer_engine:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/usr/local/lib/python3.10/dist-packages:${PYTHONPATH}"

# Verify installation works
RUN nvcc --version && \
    echo "=== Version Verification ===" && \
    python /scripts/test_environment.py && \
    echo "✅ All libraries verified successfully with preserved PyTorch version"

# Default command
CMD ["/bin/bash"]
