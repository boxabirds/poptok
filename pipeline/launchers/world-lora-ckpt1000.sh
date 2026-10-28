#!/usr/bin/env bash
# World LoRA test (decision D3, approved 2026-10-28): checkpoint 1000 of the felt-room LoRA (trained on top of
# Layout-To-Render, so it replaces it) on the DISTILLED model, with the model card's first-frame recipe, shots A and B,
# on thing2 under its GPU lock. Also each shot without the first frame (the LoRA's look from text + layout alone).
# Dry run first (shot A, 25 frames, same command), checked, then the full 81-frame runs.
# Out: pieces/_bench/review/2026-10-28/world-lora-1000-{A,B}.jpg (+ the mp4s next to them).
#   setsid nohup bash pipeline/launchers/world-lora-ckpt1000.sh >> pieces/_logs/world-lora-ckpt1000.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
H=thing2; T=pieces/_bench/train/v2v-ic-lora-noble-felt-room-20261026; F=pieces/_bench/world/first
O=pieces/_bench/world/lora1000; R=pieces/_bench/review/2026-10-28; mkdir -p $O $R
for p in A B; do                        # the layouts at the render size
  [ -s $O/layout_$p.mp4 ] || ffmpeg -v error -y -i $T/val/layout_$p.mp4 -vf scale=1920:1088:flags=neighbor -c:v libx264 -pix_fmt yuv420p -crf 12 $O/layout_$p.mp4
done
cp $F/first_A.png $F/first_B.png $O/
ssh $H "mkdir -p ~/poptok/$O" && rsync -a $O/ $H:poptok/$O/
M=/models/ltx-2.5
MODELS="--transformer-path $M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path $M/vae/ltx-2.5-video-vae-bf16.safetensors --audio-vae-path $M/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upsampler-path $M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
LORA="--lora /work/$T/outputs/checkpoints/lora_weights_step_01000.safetensors 1.0"
SHOT_A=$(python3 -c "import yaml;print(yaml.safe_load(open('$T/config.yaml'))['validation']['samples'][0]['prompt'])")
SHOT_B=$(python3 -c "import yaml;print(yaml.safe_load(open('$T/config.yaml'))['validation']['samples'][1]['prompt'])")
wait_free="while $CTRS | grep -qE '$BUSY_RE'; do sleep 20; done"
run() {                                 # name, frames, extra args, prompt
  [ -s $O/$1.mp4 ] && return 0
  echo "[$(date +%H:%M)] $1"
  ssh $H "cd ~/poptok && $wait_free; flock $LOCK $(ltx_ctr $H) python3 -m ltx_pipelines.ic_lora $MODELS $LORA $3 \
    --stage-2-ic-lora --height 1088 --width 1920 --num-frames $2 --prompt \"$4\" --seed 42 --frame-rate 24 \
    --output-path $O/$1.mp4" > $O/$1.log 2>&1 || { echo "  $1 FAILED: $(grep -E 'Error' $O/$1.log | tail -1)"; return 1; }
  rsync -a $H:poptok/$O/$1.mp4 $O/
  n=$(ffprobe -v error -count_frames -select_streams v -show_entries stream=nb_read_frames -of csv=p=0 $O/$1.mp4)
  [ "$n" -eq "$2" ] || { echo "  $1 has $n frames, expected $2"; return 1; }
  echo "  ok ($n frames)"
}
run dry_A 25 "--video-conditioning $O/layout_A.mp4 1.0 --image $O/first_A.png 0 1.0" "$SHOT_A" || { echo "dry run FAILED: stop"; exit 1; }
for p in A B; do
  [ $p = A ] && SHOT=$SHOT_A || SHOT=$SHOT_B
  run first_$p 81 "--video-conditioning $O/layout_$p.mp4 1.0 --image $O/first_$p.png 0 1.0" "$SHOT"
  run nofirst_$p 81 "--video-conditioning $O/layout_$p.mp4 1.0" "$SHOT"
done
python3 - <<EOF
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
O, B = "$O", "pieces/_bench/world/flat_dressed/ctl"
for p in "AB":
    rows = [("layout", f"{O}/layout_{p}.mp4"), ("base L2R + first frame (earlier)", f"{B}/l2r_first_{p}.mp4"),
            ("LoRA 1000 + first frame", f"{O}/first_{p}.mp4"), ("LoRA 1000, no first frame", f"{O}/nofirst_{p}.mp4")]
    W, H = 480, 272; idx = [0, 27, 54, 80]
    sheet = Image.new("RGB", (W * 4, H * len(rows)))
    for r, (name, path) in enumerate(rows):
        c = cv2.VideoCapture(path)
        for j, i in enumerate(idx):
            c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = c.read()
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((W, H)) if ok else Image.new("RGB", (W, H))
            d = ImageDraw.Draw(im); lab = f"{r+1}{'abcd'[j]} f{i}" + (f"  {name}" if j == 0 else "")
            d.rectangle((0, 0, d.textlength(lab, font=F) + 10, 30), fill=0); d.text((5, 3), lab, font=F, fill=(255, 255, 0))
            sheet.paste(im, (j * W, r * H))
    sheet.save(f"$R/world-lora-1000-{p}.jpg", quality=88)
EOF
cp $O/first_A.mp4 $R/world-lora-1000-first-A.mp4 2>/dev/null; cp $O/first_B.mp4 $R/world-lora-1000-first-B.mp4 2>/dev/null
cp $O/nofirst_A.mp4 $R/world-lora-1000-nofirst-A.mp4 2>/dev/null; cp $O/nofirst_B.mp4 $R/world-lora-1000-nofirst-B.mp4 2>/dev/null
echo "[$(date +%H:%M)] world lora 1000 done"
