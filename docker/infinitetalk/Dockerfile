# InfiniteTalk (MeiGen-AI, audio-driven talking/singing video on Wan2.1-I2V-14B) for the DGX Spark (ARM64, GB10 = sm_121).
# No prebuilt aarch64 xformers exists for torch cu130, so it is compiled here for sm_121. Pinned to v0.0.35: later
# releases moved the attention kernels into the separate "mslk" package (memory_efficient_attention missing) (its own flash-attn copy is not
# built: Wan's attention falls back to torch SDPA, and xformers' memory-efficient kernels cover the audio cross-attention).
#   docker build -t infinitetalk docker/infinitetalk          (on a Spark; ~1 h, mostly the xformers compile)
# Weights in ~/models/infinitetalk (Wan2.1-I2V-14B-480P, chinese-wav2vec2-base, InfiniteTalk/single).
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ffmpeg libgl1 \
      libglib2.0-0 build-essential ninja-build && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/bin/python
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
ENV TORCH_CUDA_ARCH_LIST=12.1 XFORMERS_DISABLE_FLASH_ATTN=1 MAX_JOBS=16
RUN pip3 install -v --no-build-isolation "git+https://github.com/facebookresearch/xformers.git@v0.0.35#egg=xformers" \
    && python3 -c "import xformers, xformers.ops; print('xformers', xformers.__version__)"
RUN pip3 install "transformers>=4.49,<5" "diffusers>=0.31,<0.36" "accelerate>=1.1.1" tokenizers tqdm imageio imageio-ffmpeg \
      easydict ftfy scikit-image loguru "numpy>=1.23.5,<2" pyloudnorm optimum-quanto==0.2.6 scenedetect moviepy==1.0.3 \
      opencv-python-headless librosa soundfile safetensors sentencepiece
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /opt/InfiniteTalk && cd /opt/InfiniteTalk && git checkout 50aa0a94184315407a991ae804d9b58d6d311ba8
WORKDIR /opt/InfiniteTalk
# Modules InfiniteTalk imports at load time but leaves out of requirements.txt (einops, regex, yaml). decord has no
# aarch64 wheel for Python 3.12; InfiniteTalk only uses it to read a video as the conditioning input, so a stub that
# fails loudly stands in (a photo as input never calls it). xfuser is imported at load time even for one GPU.
RUN pip3 install einops regex pyyaml scipy requests xfuser==0.4.1 \
    && d=$(python3 -c "import site; print(site.getsitepackages()[0])")/decord && mkdir -p $d \
    && printf 'def cpu(*a, **k):\n    return None\n\nclass VideoReader:\n    def __init__(self, *a, **k):\n        raise RuntimeError("decord is not available on aarch64: use a photo, not a video, as cond_video")\n' > $d/__init__.py \
    && python3 -c "import einops, decord"
# wan/multitalk.py imports inspect.ArgSpec, which Python 3.11 removed; it is never used.
RUN sed -i '/^from inspect import ArgSpec$/d' wan/multitalk.py && python3 -c "import ast; ast.parse(open('wan/multitalk.py').read())"
# generate_infinitetalk.py imports its bundled Kokoro TTS at load time (used only for text input), which needs misaki.
RUN pip3 install "misaki[en]" && python3 generate_infinitetalk.py --help > /dev/null 2>&1 || python3 -c "import misaki"
# transformers 4.57 loads wav2vec2 with SDPA attention, which refuses the output_attentions InfiniteTalk asks for.
# The audio encoder is small, so plain attention costs nothing.
RUN sed -i 's/Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True)/Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True, attn_implementation="eager")/' generate_infinitetalk.py \
    && grep -q 'attn_implementation="eager"' generate_infinitetalk.py
# Wan calls flash_attention() directly and asserts flash-attn is installed; it isn't built for GB10. Route it to
# torch SDPA, keeping the padding mask (see sdpa_fallback.py).
COPY sdpa_fallback.py /tmp/sdpa_fallback.py
RUN python3 /tmp/sdpa_fallback.py wan/modules/attention.py
