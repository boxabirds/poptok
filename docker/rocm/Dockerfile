# NOT YET WORKING (2026-10-23): on ROCm 7.14 (torch 2.11-2.13) attention fails at MOSS sizes ("invalid argument"); tritus
# stays on chronoedit-rocm (torch 2.10 + ROCm 7.13 alpha, efficient attention OK). Retry when AMD fixes it.
#
# Keyframe image for AMD Strix Halo (gfx1151): AMD official ROCm 7.14 PyTorch from the multi-arch index (torch +
# the gfx1151 device package; torch 2.11 lacks the AOTriton attention kernels for gfx1151 (only slow math SDPA), 2.12 has
# the memory-efficient one (flash still fails: "invalid argument", so poptok disables flash on ROCm); bundles its own ROCm runtime, the host needs only the amdgpu driver, kernel >= 6.18.4).
#   podman build -t poptok-rocm:7.14 docker/rocm        (on the Strix box)
FROM docker.io/library/python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/ \
      "torch==2.12.0+rocm7.14.1" "torchvision==0.27.0+rocm7.14.1" "amd-torch-device-gfx1151==2.12.0+rocm7.14.1"
RUN pip install "diffusers>=0.41" transformers accelerate safetensors sentencepiece ftfy numpy pillow \
      opencv-python-headless imageio imageio-ffmpeg peft scikit-image soundfile librosa
WORKDIR /work
