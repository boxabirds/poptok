# InfiniteTalk (MeiGen-AI, audio-driven singing video on Wan2.1-I2V-14B) for AMD Strix Halo (gfx1151, tritus), podman.
# Same InfiniteTalk commit and fixes as docker/infinitetalk (the DGX Spark build), on the chronoedit-rocm base (torch
# 2.10 + ROCm 7.13 nightly for gfx1151, whose SDPA flash/efficient kernels work at Wan's 33k-token self-attention; the
# ROCm 7.14 poptok-rocm image fails there). xformers has no ROCm build: xformers_shim/ stands in for the one call
# InfiniteTalk makes (memory_efficient_attention, + BlockDiagonalMask) on torch SDPA.
#   podman build -t infinitetalk-rocm docker/infinitetalk-rocm        (on tritus; ~5 min)
# Weights in ~/models/infinitetalk (same layout as the Sparks). Run: see run_insert.sh.
# Measured 2026-10-24 (slippery-road insert_0, 10.7 s, fast 4-step mode, no offload): 61.6 min wall (2.1 min load,
# 197 s per denoising step vs 77 s on a Spark), i.e. ~5.8 min of compute per second of video (Spark ~2.4).
# The two ROCm-specific speed fixes: contiguous q/k/v for SDPA (sdpa_fallback.py) and the VAE's conv3d as conv2d.
FROM localhost/chronoedit-rocm:latest
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
# gcc: inductor compiles the @torch.compile in wan/utils/multitalk_utils.py (calculate_x_ref_attn_map).
RUN apt-get update && apt-get install -y --no-install-recommends git gcc g++ libc6-dev && rm -rf /var/lib/apt/lists/*
# Keep the base image's ROCm torch: every later pip install is constrained to it.
RUN pip freeze | grep -E '^(torch|torchvision|torchaudio|pytorch-triton-rocm|triton)(==| @)' > /etc/pip-constraints.txt; \
    cat /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
RUN pip install "transformers>=4.49,<5" "diffusers>=0.31,<0.36" "accelerate>=1.1.1" tokenizers tqdm imageio imageio-ffmpeg \
      easydict ftfy scikit-image loguru "numpy>=1.23.5,<2" pyloudnorm optimum-quanto==0.2.6 scenedetect moviepy==1.0.3 \
      opencv-python-headless librosa soundfile safetensors sentencepiece einops regex pyyaml scipy requests xfuser==0.4.1 \
    && python -c "import torch; assert 'rocm' in torch.__version__, torch.__version__; print(torch.__version__)"
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /opt/InfiniteTalk && cd /opt/InfiniteTalk && git checkout 50aa0a94184315407a991ae804d9b58d6d311ba8
WORKDIR /opt/InfiniteTalk
# decord has no Python 3.12 wheel; InfiniteTalk only uses it to read a conditioning video (a photo never calls it).
RUN d=$(python -c "import site; print(site.getsitepackages()[0])")/decord && mkdir -p $d \
    && printf 'def cpu(*a, **k):\n    return None\n\nclass VideoReader:\n    def __init__(self, *a, **k):\n        raise RuntimeError("decord is not available: use a photo, not a video, as cond_video")\n' > $d/__init__.py
# xformers stand-in on torch SDPA (see xformers_shim/xformers/ops/fmha).
COPY xformers_shim/xformers /usr/local/lib/python3.12/site-packages/xformers
# xfuser parses torch.version.cuda at import, which is None on ROCm; give it a CUDA-12-like value (only version checks).
RUN f=$(python -c "import site; print(site.getsitepackages()[0])")/xfuser/envs.py \
    && sed -i 's/version.parse(torch.version.cuda)/version.parse(torch.version.cuda or "12.4")/' $f && grep -q 'cuda or "12.4"' $f
# wan/multitalk.py imports inspect.ArgSpec, which Python 3.11 removed; it is never used.
RUN sed -i '/^from inspect import ArgSpec$/d' wan/multitalk.py && python -c "import ast; ast.parse(open('wan/multitalk.py').read())"
# generate_infinitetalk.py imports its bundled Kokoro TTS at load time (used only for text input), which needs misaki.
RUN pip install "misaki[en]" && python -c "import misaki, torch; assert 'rocm' in torch.__version__"
# transformers 4.57 loads wav2vec2 with SDPA attention, which refuses the output_attentions InfiniteTalk asks for.
RUN sed -i 's/Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True)/Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True, attn_implementation="eager")/' generate_infinitetalk.py \
    && grep -q 'attn_implementation="eager"' generate_infinitetalk.py
# Wan's flash_attention() asserts flash-attn is installed; route it to torch SDPA keeping the padding mask.
COPY sdpa_fallback.py /tmp/sdpa_fallback.py
RUN python /tmp/sdpa_fallback.py wan/modules/attention.py
# MIOpen's 3D convolutions are very slow on gfx1151; run the VAE's CausalConv3d as exact sums of 2D convolutions.
COPY conv3d_as_conv2d.py /tmp/conv3d_as_conv2d.py
RUN python /tmp/conv3d_as_conv2d.py wan/modules/vae.py
COPY test_shim.py bench_vae.py /opt/
# (xfuser probes the GPU at import, so the import check runs in the container, not at build: python /opt/test_shim.py)
