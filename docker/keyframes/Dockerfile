# Keyframe image (MOSS-Image via diffusers). Built and tagged "zoommorph" (its original name):
#   docker build -t zoommorph docker/keyframes
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip3 install "diffusers>=0.41.0" transformers accelerate safetensors sentencepiece \
        numpy scipy pillow librosa soundfile matplotlib \
        numexpr opencv-python-headless scikit-image gguf peft bitsandbytes ftfy

WORKDIR /work
