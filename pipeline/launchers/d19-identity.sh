#!/usr/bin/env bash
# D19 (approved 2026-10-28): D12's shot B clip was her only at frame 0 (3a f0 "the best yet") and degraded once she
# turned away, because the clip prompt still described the old person ("long dark-blond hair") while only frame 0 was
# pinned to her photo. Step 1: the same clip with the prompt's person replaced by D12's photo-based description; same
# first frame, layout, LoRA 1000, seed. Scored every 8 frames; row 4 of the shot B sheet. Shot A dropped (D18).
# Step 2 (mid/last-frame conditioning) only if row 4 still degrades.
# Waits until thing2 has no Blender running (song3's last chunks ignore the lock), then thing2's GPU lock.
# Out: pieces/_bench/review/2026-10-28/world-lora-1000-her-B.jpg (row 4), world-lora-1000-d19-B.mp4
#   setsid nohup bash pipeline/launchers/d19-identity.sh >> pieces/_logs/d19-identity.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
H=thing2; O=pieces/_bench/world/lora1000; B=pieces/_bench/world/flat_dressed/ctl
T=pieces/_bench/train/v2v-ic-lora-noble-felt-room-20261026; R=pieces/_bench/review/2026-10-28
PHOTO=pieces/artists/r.amber.jpeg
OLD="a young woman with long dark-blond hair in a dark green-brown felted wool field uniform"
PERSON="a young woman with long straight light-blonde hair, blue eyes and fair skin, a bare clean face, in a worn dark green-brown felted wool field uniform with long sleeves and boots"
SHOT=$(python3 -c "
import yaml, sys
s = yaml.safe_load(open('$T/config.yaml'))['validation']['samples'][1]['prompt']
assert '$OLD' in s, 'old person not in shot B prompt'
print(s.replace('$OLD', '$PERSON'))") || { echo "FAILED: prompt"; exit 1; }
echo "[$(date +%H:%M)] prompt: $SHOT"
while ssh $H 'pgrep -x blender' > /dev/null; do sleep 60; done
echo "[$(date +%H:%M)] d19_B on $H"
M=/models/ltx-2.5
MODELS="--transformer-path $M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path $M/vae/ltx-2.5-video-vae-bf16.safetensors --audio-vae-path $M/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upsampler-path $M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
LORA="--lora /work/$T/outputs/checkpoints/lora_weights_step_01000.safetensors 1.0"
rm -f $O/d19_B.mp4
ssh $H "cd ~/poptok && $SIDE flock $LOCK $(ltx_ctr $H) python3 -m ltx_pipelines.ic_lora $MODELS $LORA \
  --video-conditioning $O/layout_B.mp4 1.0 --image $O/first_her_B.png 0 1.0 \
  --stage-2-ic-lora --height 1088 --width 1920 --num-frames 81 --prompt \"$SHOT\" --seed 42 --frame-rate 24 \
  --output-path $O/d19_B.mp4" > $O/d19_B.log 2>&1 || { echo "  FAILED: $(grep Error $O/d19_B.log | tail -1)"; exit 1; }
rsync -a $H:poptok/$O/d19_B.mp4 $O/
python3 pipeline/eval_identity.py $PHOTO $O/d19_B.mp4 --every 8 --json $O/d19_B.id.json | tail -1
python3 - <<PY
import cv2, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
O, B = "$O", "$B"
def ident(path):
    r = [x["similarity"] for x in json.load(open(path[:-4] + ".id.json")) if x["similarity"] is not None]
    return f"her? median {np.median(r):.2f} min {min(r):.2f} ({len(r)} faces)" if r else "her? no face found"
rows = [("base L2R + old first frame", f"{B}/l2r_first_B.mp4"), ("LoRA 1000 + old first frame", f"{O}/first_B.mp4"),
        ("LoRA 1000 + D12 first frame", f"{O}/first_her_B.mp4"), ("+ D19 prompt describes her", f"{O}/d19_B.mp4")]
W, H = 480, 272; idx = [0, 27, 54, 80]
sheet = Image.new("RGB", (W * 4, H * len(rows)))
for r, (name, path) in enumerate(rows):
    c = cv2.VideoCapture(path)
    for j, i in enumerate(idx):
        c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = c.read()
        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((W, H)) if ok else Image.new("RGB", (W, H))
        d = ImageDraw.Draw(im)
        for y, lab in ((3, f"{r+1}{'abcd'[j]} f{i}" + (f"  {name}" if j == 0 else "")), (240, ident(path) if j == 0 else "")):
            if lab:
                d.rectangle((0, y - 3, d.textlength(lab, font=F) + 10, y + 27), fill=0); d.text((5, y), lab, font=F, fill=(255, 255, 0))
        sheet.paste(im, (j * W, r * H))
sheet.save("$R/world-lora-1000-her-B.jpg", quality=88)
# consecutive frames 16-47 (she turns away around f27) of D12 then D19, to judge where she degrades
c12, c19 = cv2.VideoCapture(f"{O}/first_her_B.mp4"), cv2.VideoCapture(f"{O}/d19_B.mp4")
w, h = 240, 136; s = Image.new("RGB", (w * 8, h * 8))
for r, (c, n) in enumerate([(c12, "D12")] * 4 + [(c19, "D19")] * 4):
    for j in range(8):
        i = 16 + (r % 4) * 8 + j; c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = c.read()
        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((w, h)) if ok else Image.new("RGB", (w, h))
        d = ImageDraw.Draw(im); lab = f"{n} f{i}"; f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        d.rectangle((0, 0, d.textlength(lab, font=f) + 8, 22), fill=0); d.text((4, 2), lab, font=f, fill=(255, 255, 0))
        s.paste(im, (j * w, r * h))
s.save("$R/d19-B-frames-16-47.jpg", quality=88)
PY
cp $O/d19_B.mp4 $R/world-lora-1000-d19-B.mp4
echo "[$(date +%H:%M)] d19 done"
