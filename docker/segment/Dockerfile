# Text-prompted instance segmentation for the plausibility critic (POC 28): Grounding DINO (boxes from text prompts)
# + SAM 2.1 (masks from boxes), both Apache-2.0, run through Hugging Face transformers. pipeline/segment_check.py.
#   docker build -t segment docker/segment
#   weights: ~/models/segment/{grounding-dino-base,sam2.1-hiera-large} (hfdl), mounted at /models
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/bin/python
RUN pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install "transformers==4.57.1" "numpy<2" pillow scipy opencv-python-headless accelerate
WORKDIR /work
