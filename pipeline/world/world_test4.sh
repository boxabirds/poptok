#!/usr/bin/env bash
# World-building test, round 4, on one Spark: the blockout textured by projecting styled views onto it
# (project_texture.py), so every view shows the same room; look frames = her added to the textured frames (looks4.py);
# Layout-To-Render driven by the TEXTURED layout video, with and without the Ingredients sheet of round 3.
#   pipeline/world/world_test4.sh pieces/_bench/world/room3/p2 thing1
set -euo pipefail
cd "$(dirname "$0")/../.."
O=$1; H=${2:-thing1}; R=$(dirname $O)          # round 3's folder: figure.json, sheet_1920.mp4
source bin/hosts.sh
for p in A B; do ffmpeg -v error -y -framerate 24 -i $O/tex_$p/%04d.png -c:v libx264 -pix_fmt yuv420p -crf 12 $O/tex_$p.mp4; done
ssh $H "mkdir -p ~/poptok/$O"; rsync -a pipeline $H:poptok/
rsync -a $O/ $H:poptok/$O/ --exclude 'tex_A/' --exclude 'tex_B/' --exclude 'texviews/' --exclude '*.blend*'
rsync -a $R/figure.json $R/sheet_1920.mp4 $H:poptok/$R/
wait_free="while $CTRS | grep -qE '$BUSY_RE'; do sleep 20; done"
echo "[$(date +%H:%M)] looks"
ssh $H "cd ~/poptok && $wait_free; flock $LOCK $(keyframe_ctr $H) python3 pipeline/world/looks4.py $O $R/figure.json pieces/artists/r.amber.jpeg" \
  > $O/looks4.log 2>&1 || { echo "looks failed: $O/looks4.log"; exit 1; }
rsync -a $H:poptok/$O/ $O/ --include 'look4*.png' --exclude '*'
M=/models/ltx-2.5
MODELS="--transformer-path $M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path $M/vae/ltx-2.5-video-vae-bf16.safetensors --audio-vae-path $M/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upsampler-path $M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
L2R="--lora $M/loras/ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors 1.0"
ING="--lora $M/loras/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors 1.0"
SHEET="Reference sheet: a young blond woman soldier, a face close-up and a full-body turnaround front, side and back, in a dark green-brown felted wool field uniform, her face streaked with green and black camouflage paint; a cream wool cloth with a stitched peace symbol; a ruined apartment room made of thick felted wool."
SHOT_A="A ruined apartment room made of thick felted wool at night, the camera moving slowly in through the open door toward the window, where the young woman soldier sits on the floor against the wall, knees up, holding the wool cloth. Muted greens and browns, soft fuzzy surfaces, a distant glow through the window."
SHOT_B="Close beside the young woman soldier sitting on the floor under the window of a ruined apartment room made of thick felted wool, the camera turning slowly away from her across the room toward the tilted table. Muted greens and browns, soft fuzzy surfaces."
run() {
  [ -s $O/$1.mp4 ] && return
  echo "[$(date +%H:%M)] $1"
  ssh $H "cd ~/poptok && $wait_free; flock $LOCK $(ltx_ctr $H) python3 -m ltx_pipelines.ic_lora $MODELS $2 \
    --prompt \"$3\" --seed 42 --frame-rate 24 --output-path $O/$1.mp4" > $O/$1.log 2>&1 \
    || echo "  failed: $(grep -vE 'leaked|nanobind|^ - ' $O/$1.log | grep -E 'Error' | tail -1)"
  rsync -a $H:poptok/$O/$1.mp4 $O/ 2>/dev/null || true
}
for p in A B; do
  [ $p = A ] && SHOT=$SHOT_A || SHOT=$SHOT_B
  LOOKS="--image $O/look4_${p}_0001.png 0 1.0 --image $O/look4_${p}_0041.png 40 1.0 --image $O/look4_${p}_0081.png 80 1.0"
  GEOM="--video-conditioning $O/tex_$p.mp4 1.0 --stage-2-ic-lora --height 1088 --width 1920 --num-frames 81"
  run l2r_$p "$L2R $GEOM $LOOKS" "$SHOT"
  run l2r_ing_$p "$L2R $ING $GEOM --video-conditioning $R/sheet_1920.mp4 1.0 $LOOKS" "$SHEET Generated video: $SHOT"
done
echo "[$(date +%H:%M)] round 4 done"
