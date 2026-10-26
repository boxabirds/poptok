#!/usr/bin/env bash
# Quantized InfiniteTalk on gruntus (RTX 4090 24 GB): official int8 weights with the speed LoRA merged + fp8 T5, every
# weight on the GPU, no CPU offload. Fits only with docker/infinitetalk-x86/vram_fit.py (peak ~24.1 of 24.6 GB), so it
# needs the whole card: it runs under the poptok GPU lock. ~62 s of compute per second of video (2.6x a Spark at bf16);
# user verdict 2026-10-24: better than the Spark render.
#   bin/infinitetalk_quant.sh WORKDIR JSON OUTSTEM [STEPS] [QUANT_FILE]      (WORKDIR holds the json, png, wav)
# Image: docker build -t infinitetalk-x86 docker/infinitetalk-x86. Weights: ~/models/infinitetalk-quant.
set -euo pipefail
W=$(realpath "$1"); J=$2; O=$3; STEPS=${4:-4}; QF=${5:-infinitetalk_single_int8_lora}
Q=int8; [[ $QF == *fp8* ]] && Q=fp8
flock /tmp/poptok-gpu.lock docker run --gpus all --rm --user $(id -u):$(id -g) --ipc=host -e HOME=/tmp -e USER=$(id -un) \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $HOME/models/infinitetalk-quant:/weights:ro \
  -v $HOME/models/infinitetalk-quant/quanto-build:/usr/local/lib/python3.10/dist-packages/optimum/quanto/library/extensions/cuda/build \
  -v "$W":/work -w /work infinitetalk-x86 \
  python /opt/peak.py --ckpt_dir /weights/Wan2.1-I2V-14B-480P --wav2vec_dir /weights/chinese-wav2vec2-base \
  --infinitetalk_dir /weights/unused.safetensors \
  --quant $Q --quant_dir /weights/InfiniteTalk/quant_models/$QF.safetensors \
  --input_json /work/$J --size infinitetalk-480 --sample_steps $STEPS --sample_text_guide_scale 1.0 \
  --sample_audio_guide_scale 2.0 --sample_shift 2 --mode streaming --motion_frame 9 --offload_model False \
  --save_file /work/$O ${MAX_FRAMES:+--max_frame_num $MAX_FRAMES}
