#!/usr/bin/env bash
# Render one InfiniteTalk insert on tritus (AMD Strix Halo, podman), with the same fast settings as bin/sing_inserts.sh on
# the Sparks (lightx2v LoRA, 4 steps, text guide 1, audio guide 2, shift 2, streaming, motion_frame 9, no offload).
# Run ON tritus from the directory holding the insert's files (its JSON's paths are /work/...):
#   cd ~/poptok/<run>/sing && bash ~/poptok-build/infinitetalk-rocm/run_insert.sh it_insert_0.json it_out_0
# Writes it_out_0.mp4 and it_out_0.log there. Takes the poptok GPU lock (one GPU job per box).
set -euo pipefail
JSON=$1; OUTNAME=$2
LOCK=${TMPDIR:-/tmp}/poptok-gpu.lock
CACHE=$HOME/.cache/infinitetalk-rocm      # MIOpen kernel picks + inductor/Triton builds survive between runs (first run
mkdir -p "$CACHE/miopen"                  # tunes: ~+2 min)
flock "$LOCK" podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
  --ipc=host -e HOME=/tmp -e MIOPEN_USER_DB_PATH=/cache/miopen -e MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton -v "$CACHE:/cache" -v "$HOME/models/infinitetalk:/weights:ro" -v "$PWD:/work" -w /work localhost/infinitetalk-rocm \
  python /opt/InfiniteTalk/generate_infinitetalk.py --ckpt_dir /weights/Wan2.1-I2V-14B-480P \
  --wav2vec_dir /weights/chinese-wav2vec2-base --infinitetalk_dir /weights/InfiniteTalk/single/infinitetalk.safetensors \
  --lora_dir /weights/lora/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors --lora_scale 1.0 \
  --input_json "/work/$JSON" --size infinitetalk-480 --sample_steps 4 --sample_text_guide_scale 1.0 \
  --sample_audio_guide_scale 2.0 --sample_shift 2 --mode streaming --motion_frame 9 --offload_model False \
  --save_file "/work/$OUTNAME" > "$OUTNAME.log" 2>&1
