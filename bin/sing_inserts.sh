#!/usr/bin/env bash
# Render a run's sung inserts (pipeline/sing_inserts.py plan) on gruntus and/or DGX Sparks, one GPU job per box: each
# host takes its share of the inserts in turn. "gruntus" (or "local") renders here with the int8 engine
# (bin/infinitetalk_quant.sh, ~2.6x a Spark); other hosts are Sparks (docker) or ROCm boxes (podman, ROCM_HOSTS) over ssh. ENGINE=infinitetalk (default; fast 4-step lightx2v mode, no offload) or longcat.
#   bin/sing_inserts.sh OUT "gruntus:0,3 thing1:1 thing2:2"
# Needs the image (docker/infinitetalk or docker/longcat) and its weights in ~/models on each host. Clips come back to
# OUT/sing/ (it_out_K.mp4, or out_K/ for LongCat).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$1; PLAN=$2
source bin/hosts.sh
# frames InfiniteTalk may make for insert K (25 fps): its default cap of 1000 (40 s) cut Chronophobia's 74 s insert short
maxf() { python3 -c "import json,math;print(math.ceil([r for r in json.load(open('$OUT/sing/inserts.json')) if r['k']==$1][0]['duration']*25)+25)"; }
render() {   # host, comma-separated insert numbers
  local h=$1 ks=$2 D=$OUT/sing
  if [ "$h" = gruntus ] || [ "$h" = local ]; then
    for k in ${ks//,/ }; do
      [ -s "$D/it_out_$k.mp4" ] && { echo "insert $k: already rendered"; continue; }
      echo "[$(date +%H:%M:%S)] insert $k on gruntus (int8)"
      MAX_FRAMES=$(maxf "$k") bin/infinitetalk_quant.sh "$D" it_insert_$k.json it_out_$k > "$D/it_insert_$k.log" 2>&1 \
        || echo "insert $k failed on gruntus (see $D/it_insert_$k.log)"
      echo "[$(date +%H:%M:%S)] insert $k done"
    done
    return
  fi
  ssh "$h" "mkdir -p ~/poptok/$D"; rsync -a --delete "$D"/ "$h":poptok/"$D"/   # its copy mirrors this one: no stale clips
  for k in ${ks//,/ }; do
    if [ "${ENGINE:-infinitetalk}" = infinitetalk ] && [ -s "$D/it_out_$k.mp4" ]; then echo "insert $k: already rendered"; continue; fi
    n=$(python3 -c "import json;print([r for r in json.load(open('$D/inserts.json')) if r['k']==$k][0]['segments'])")
    echo "[$(date +%H:%M:%S)] insert $k ($n segments) on $h"
    if [ "${ENGINE:-infinitetalk}" = infinitetalk ]; then
      ssh "$h" "cd ~/poptok/$D && $PREP while $CTRS | grep -qE '$BUSY_RE|longcat'; do sleep 20; done; \
        test -s it_out_$k.mp4 || flock $LOCK $(talk_ctr "$h") python /opt/InfiniteTalk/generate_infinitetalk.py --ckpt_dir /weights/Wan2.1-I2V-14B-480P \
        --wav2vec_dir /weights/chinese-wav2vec2-base --infinitetalk_dir /weights/InfiniteTalk/single/infinitetalk.safetensors \
        --lora_dir /weights/lora/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors --lora_scale 1.0 \
        --input_json /work/it_insert_$k.json --size infinitetalk-480 --sample_steps 4 --sample_text_guide_scale 1.0 \
        --sample_audio_guide_scale 2.0 --sample_shift 2 --mode streaming --motion_frame 9 --offload_model False \
        --max_frame_num $(maxf "$k") --save_file /work/it_out_$k > it_insert_$k.log 2>&1" || echo "insert $k failed on $h (see $D/it_insert_$k.log there)"
      rsync -a "$h":poptok/"$D"/it_out_$k.mp4 "$h":poptok/"$D"/it_insert_$k.log "$D"/ || true
      echo "[$(date +%H:%M:%S)] insert $k done"; continue
    fi
    ssh "$h" "cd ~/poptok/$D && while docker ps --format '{{.Image}}' | grep -qE '$BUSY_RE|longcat|infinitetalk'; do sleep 20; done; \
      flock $LOCK docker run --gpus all --rm --user \$(id -u):\$(id -g) --ipc=host -e HOME=/tmp -v \$HOME/models/longcat:/weights \
      -v \$PWD:/work -w /work longcat torchrun --nproc_per_node=1 /opt/LongCat-Video/run_demo_avatar_single_audio_to_video.py \
      --context_parallel_size=1 --checkpoint_dir=/weights/LongCat-Video-Avatar-1.5 --stage_1=ai2v --input_json=/work/insert_$k.json \
      --num_segments=$n --ref_img_index=10 --mask_frame_range=3 --use_distill --model_type avatar-v1.5 --resolution 480p \
      --output_dir /work/out_$k > insert_$k.log 2>&1" || echo "insert $k failed on $h (see $D/insert_$k.log there)"
    rsync -a "$h":poptok/"$D"/out_$k "$h":poptok/"$D"/insert_$k.log "$D"/ || true
    echo "[$(date +%H:%M:%S)] insert $k done"
  done
}
for job in $PLAN; do render "${job%%:*}" "${job#*:}" & done
wait
