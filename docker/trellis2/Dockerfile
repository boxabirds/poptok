# TRELLIS.2 (microsoft/TRELLIS.2, MIT; weights microsoft/TRELLIS.2-4B, MIT) on a DGX Spark (aarch64, GB10 sm_121): her
# photo -> a textured 3D mesh for the world layouts (pipeline/world/, docs/research/photo-to-3d-avatar.md).
# Built on the InfiniteTalk image, which already has nvcc, torch (cu130) and xformers for aarch64; the sparse attention
# runs on xformers (ATTN_BACKEND=xformers; flash-attn is not built here).
#   docker build -t trellis2 docker/trellis2            (on a Spark)
# Patched: background removal (briaai/RMBG-2.0, CC BY-NC, gated) is not loaded when TRELLIS2_NO_REMBG=1 — we pass images
# whose background is already removed (RGBA), which the pipeline uses as is.
FROM infinitetalk:latest
ENV TORCH_CUDA_ARCH_LIST="12.1" ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers MAX_JOBS=8
# headers for utils3d's glcontext (moderngl) build on aarch64
RUN apt-get update && apt-get install -y --no-install-recommends libx11-dev libgl1-mesa-dev libegl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
      tensorboard pandas lpips zstandard kornia timm \
    && pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
RUN git clone -b main --recursive https://github.com/microsoft/TRELLIS.2.git /opt/trellis2
RUN mkdir -p /tmp/ext && cd /tmp/ext \
    && git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git && pip install --no-build-isolation ./nvdiffrast \
    && git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git && pip install --no-build-isolation ./nvdiffrec \
    && git clone --recursive https://github.com/JeffreyXiang/CuMesh.git && pip install --no-build-isolation ./CuMesh \
    && git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git && pip install --no-build-isolation ./FlexGEMM \
    && pip install --no-build-isolation /opt/trellis2/o-voxel && rm -rf /tmp/ext
RUN sed -i "s|        pipeline.rembg_model = getattr(rembg, args\['rembg_model'\]\['name'\])(\*\*args\['rembg_model'\]\['args'\])|        pipeline.rembg_model = None if os.environ.get('TRELLIS2_NO_REMBG') else getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])|" /opt/trellis2/trellis2/pipelines/trellis2_image_to_3d.py \
    && grep -q "TRELLIS2_NO_REMBG" /opt/trellis2/trellis2/pipelines/trellis2_image_to_3d.py \
    && (grep -q "^import os" /opt/trellis2/trellis2/pipelines/trellis2_image_to_3d.py || sed -i '1i import os' /opt/trellis2/trellis2/pipelines/trellis2_image_to_3d.py)
ENV PYTHONPATH=/opt/trellis2 TRELLIS2_NO_REMBG=1 OPENCV_IO_ENABLE_OPENEXR=1
WORKDIR /work
