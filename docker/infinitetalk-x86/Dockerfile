# InfiniteTalk (MeiGen-AI, audio-driven singing video on Wan2.1-I2V-14B) for gruntus (x86_64, RTX 4090 24 GB, sm_89),
# running the official quantized DiT (quant_models/*_int8_lora or *_fp8) with every weight on the GPU, no CPU offload.
# Same pinned commit and fixes as docker/infinitetalk (the Spark build); here torch/xformers come as prebuilt x86 cu124
# wheels, so nothing is compiled. flash-attn is not installed: sdpa_fallback.py routes Wan's attention to torch SDPA.
#   docker build -t infinitetalk-x86 docker/infinitetalk-x86
# Weights in ~/models/infinitetalk-quant (see vram_fit.py for what makes it fit in 24 GB).
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 TORCH_CUDA_ARCH_LIST=8.9
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ffmpeg libgl1 \
      libglib2.0-0 build-essential ninja-build && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/bin/python
RUN pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 xformers==0.0.29.post3 \
      --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -c "import xformers.ops, torch; print('torch', torch.__version__, 'xformers ok')"
RUN pip3 install "transformers>=4.49,<5" "diffusers>=0.31,<0.36" "accelerate>=1.1.1" tokenizers tqdm imageio imageio-ffmpeg \
      easydict ftfy scikit-image loguru "numpy>=1.23.5,<2" pyloudnorm optimum-quanto==0.2.6 scenedetect moviepy==1.0.3 \
      opencv-python-headless librosa soundfile safetensors sentencepiece decord einops regex pyyaml scipy requests xfuser==0.4.1
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /opt/InfiniteTalk && cd /opt/InfiniteTalk && git checkout 50aa0a94184315407a991ae804d9b58d6d311ba8
WORKDIR /opt/InfiniteTalk
# generate_infinitetalk.py imports its bundled Kokoro TTS at load time (used only for text input), which needs misaki.
RUN pip3 install "misaki[en]"
# transformers 4.57 loads wav2vec2 with SDPA attention, which refuses the output_attentions InfiniteTalk asks for.
RUN sed -i 's/Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True)/Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True, attn_implementation="eager")/' generate_infinitetalk.py \
    && grep -q 'attn_implementation="eager"' generate_infinitetalk.py
COPY sdpa_fallback.py vram_fit.py peak.py /opt/
# (no --help smoke test: xfuser probes the GPU at import, and the build has none)
RUN python3 /opt/sdpa_fallback.py wan/modules/attention.py && python3 /opt/vram_fit.py wan/multitalk.py wan/utils/multitalk_utils.py wan/modules/multitalk_model.py wan/modules/vae.py
# The HF repo ships the T5 encoder only as fp8 (t5_fp8.safetensors); the int8 DiTs (e.g. single_int8_lora, the only
# single-person quant with the step-distill LoRA merged) would otherwise look for a t5_int8 that does not exist.
RUN sed -i 's/f"t5_{quant}.safetensors"/f"t5_{t5q}.safetensors"/g; s/f"t5_map_{quant}.json"/f"t5_map_{t5q}.json"/' wan/modules/t5.py \
    && sed -i 's/^\(\s*\)if quant is not None:\s*$/&\n\1    t5q = quant if os.path.exists(os.path.join(quant_dir, f"t5_{quant}.safetensors")) else "fp8"/' wan/modules/t5.py \
    && grep -n "t5q" wan/modules/t5.py && python3 -c "import ast; ast.parse(open('wan/modules/t5.py').read())"
# optimum-quanto JIT-compiles its CUDA kernels on first use, into its own package dir, for the GPU it detects (so it
# cannot be done at build time, which has no GPU). Make that dir writable for --user runs; mount a host dir over it
# (-v ~/models/infinitetalk-quant/quanto-build:$QB) so the ~minutes-long compile happens once, not per run.
ENV QB=/usr/local/lib/python3.10/dist-packages/optimum/quanto/library/extensions/cuda/build
RUN mkdir -p $QB && chmod 777 $QB
