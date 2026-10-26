#!/usr/bin/env bash
# World-building test on one Spark: look frames (pipeline/world/looks.py), then LTX-2.5 Layout-To-Render on the two
# clay passes of pipeline/world/room_blockout.py: A with look_A; B with look_B_indep (styled on its own) and with
# look_B_carry (styled with look_A as reference). Each GPU job under the box's poptok lock.
#   pipeline/world/world_test.sh pieces/_bench/world/room thing1
set -euo pipefail
cd "$(dirname "$0")/../.."
O=$1; H=${2:-thing1}
source bin/hosts.sh
for p in A B; do ffmpeg -v error -y -framerate 24 -i $O/clay_$p/%04d.png -c:v libx264 -pix_fmt yuv420p -crf 12 $O/clay_$p.mp4; done
ssh $H "mkdir -p ~/poptok/$O ~/poptok/pieces/artists"
rsync -a pipeline $H:poptok/; rsync -a pieces/artists/r.amber.jpeg $H:poptok/pieces/artists/
rsync -a $O/ $H:poptok/$O/ --exclude 'clay_A/' --exclude 'clay_B/' --exclude '*.blend'
ssh $H "mkdir -p ~/poptok/$O/clay_A ~/poptok/$O/clay_B"; rsync -a $O/clay_A/0001.png $H:poptok/$O/clay_A/; rsync -a $O/clay_B/0001.png $H:poptok/$O/clay_B/
wait_free="while $CTRS | grep -qE '$BUSY_RE'; do sleep 20; done"
echo "[$(date +%H:%M)] looks (round 1; round 2: looks.py --round2)"
ssh $H "cd ~/poptok && $wait_free; flock $LOCK $(keyframe_ctr $H) python3 pipeline/world/looks.py $O pieces/artists/r.amber.jpeg"
M=/models/ltx-2.5
PROMPT_A="A ruined apartment room made of thick felted wool at night, the camera moving slowly in through the open door toward the window, where a young woman soldier in a dark felt uniform sits on the floor holding a wool cloth. Muted greens, greys and browns, soft fuzzy surfaces, a distant glow through the window."
PROMPT_B="Close beside a young woman soldier in a dark felt uniform sitting under the window of a ruined apartment room made of thick felted wool, the camera turning slowly away from her across the room to the open door. Muted greens, greys and browns, soft fuzzy surfaces."
for job in "A look2_A" "B look2_B_indep"; do
  set -- $job; P=$1; L=$2; [ "$P" = A ] && PR=$PROMPT_A || PR=$PROMPT_B
  [ -s $O/l2r_$L.mp4 ] && continue
  echo "[$(date +%H:%M)] layout-to-render $P with $L"
  ssh $H "cd ~/poptok && $wait_free; flock $LOCK $(ltx_ctr $H) python3 -m ltx_pipelines.ic_lora \
    --transformer-path $M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path $M/vae/ltx-2.5-video-vae-bf16.safetensors --audio-vae-path $M/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --spatial-upsampler-path $M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --lora $M/loras/ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors 1.0 --stage-2-ic-lora \
    --video-conditioning $O/clay_$P.mp4 1.0 --image $O/$L.png 0 1.0 \
    --prompt \"$PR\" --height 1088 --width 1920 --num-frames 81 --frame-rate 24 --seed 42 \
    --output-path $O/l2r_$L.mp4" > $O/l2r_$L.log 2>&1 || { echo "failed: see $O/l2r_$L.log"; tail -5 $O/l2r_$L.log; }
  rsync -a $H:poptok/$O/ $O/ --include '*.png' --include '*.mp4' --exclude '*' || true
done
echo "[$(date +%H:%M)] world test done"
