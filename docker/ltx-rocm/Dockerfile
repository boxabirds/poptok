# LTX-2 / LTX-2.5 reference pipelines (github.com/Lightricks/LTX-2 @ e98f6d9, packages ltx-core + ltx-pipelines) for AMD
# Strix Halo (gfx1151, tritus), podman. Base = chronoedit-rocm (torch 2.10 + ROCm 7.13 nightly for gfx1151, with
# Triton; its SDPA flash/efficient kernels work at video sizes once TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1).
# Build context is the LTX-2 checkout, like docker/ltx (the Sparks' image):
#   podman build -t ltx2-rocm -f docker/ltx-rocm/Dockerfile ref/LTX-2        (on tritus; LTX-2 rsynced there)
# Models at /models (LTX_MODELS=/models): ltx-2.5/... and ltx-2.3/ltx2.3-transition.safetensors.
FROM localhost/chronoedit-rocm:latest
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 LTX_MODELS=/models
# gcc: Triton builds its launcher stubs, inductor its C++ wrappers.
RUN apt-get update && apt-get install -y --no-install-recommends git gcc g++ libc6-dev && rm -rf /var/lib/apt/lists/*
# Keep the base image's ROCm torch: every later pip install is constrained to it.
RUN pip freeze | grep -E '^(torch|torchvision|torchaudio|pytorch-triton-rocm|triton)(==| @)' > /etc/pip-constraints.txt; \
    cat /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
COPY packages/ltx-core /ltx/packages/ltx-core
COPY packages/ltx-pipelines /ltx/packages/ltx-pipelines
# torchaudio matching the base torch, from the same gfx1151 nightly index; then the packages and their other deps.
RUN pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "torchaudio==$(python -c 'import torch; print(torch.__version__)')" \
    && pip install --no-deps /ltx/packages/ltx-core /ltx/packages/ltx-pipelines \
    && pip install einops numpy colour-science av "transformers>=5.8.0,<5.15" safetensors accelerate "scipy>=1.14" \
       tqdm pillow openimageio "cloudpickle>=3.1" opencv-python-headless numexpr scikit-image \
    && python -c "import torch; assert 'rocm' in torch.__version__, torch.__version__; print(torch.__version__)"
WORKDIR /work
