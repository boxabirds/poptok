#!/usr/bin/env bash
# D19 step 2 (approved 2026-10-28 with D19: "only if row 4 still degrades"; it did): the corrected prompt kept her
# blonde and facing the camera to about f32, but her identity drifted off after f0 (every 8 frames: 0.61, -0.05, 0.31,
# 0.30, 0.31, 0.10, 0.14, 0.24; same-person threshold 0.363). Now she is pinned at later frames too: the D19 clip's own
# frames 16 and 32 (where she faces the camera, so the motion stays the clip's), face-passed against her photo (12
# tries, gate 0.5; a face under 80 px can't be judged and is dropped), as LTX guide keyframes beside frame 0. Same
# prompt, layout, LoRA, seed as D19. Row 5 of the shot B sheet.
# Out: pieces/_bench/review/2026-10-28/world-lora-1000-her-B.jpg (row 5), world-lora-1000-d19s2-B.mp4,
#   d19s2-B-keys.jpg, d19s2-B-frames-16-47.jpg
#   setsid nohup bash pipeline/launchers/d19-step2.sh >> pieces/_logs/d19-step2.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
H=thing2; O=pieces/_bench/world/lora1000; B=pieces/_bench/world/flat_dressed/ctl; K=$O/d19s2_keys
T=pieces/_bench/train/v2v-ic-lora-noble-felt-room-20261026; R=pieces/_bench/review/2026-10-28
PHOTO=pieces/artists/r.amber.jpeg; MIN=0.5; KF="16 32"
SHOT=$(sed -n 's/^\[[0-9:]*\] prompt: //p' pieces/_logs/d19-identity.log | tail -1)
[ -n "$SHOT" ] || { echo "FAILED: no D19 prompt in pieces/_logs/d19-identity.log"; exit 1; }
rm -rf "${K:?}"; mkdir -p $K/in $K/out
for f in $KF; do ffmpeg -v error -i $O/d19_B.mp4 -vf "select=eq(n\,$f)" -frames:v 1 $K/in/k$f.png; done
echo "[$(date +%H:%M)] face pass on frames $KF (12 tries, gate $MIN on faces >= 80 px)"
ssh $H "mkdir -p ~/poptok/$K" && rsync -a pipeline $H:poptok/ && rsync -a $K/ $H:poptok/$K/
ssh $H "cd ~/poptok && $SIDE flock $LOCK $(keyframe_ctr $H) python3 pipeline/face_pass.py $PHOTO $(for f in $KF; do echo -n "$K/in/k$f.png "; done) \
  --out $K/out --tries 12 --angles pieces/artists/r.amber-angles --min-face 80" > $K/prep.log 2>&1
rsync -a $H:poptok/$K/ $K/
grep -E "similarity|too small|no face|Error" $K/prep.log
IMGS="--image $O/first_her_B.png 0 1.0"; used=""
for f in $KF; do
  s=$(grep "^k$f.png: similarity" $K/prep.log | grep -oE "[-0-9.]+$")
  if grep -q "k$f.png: face .* too small" $K/prep.log || [ -z "$s" ] || python3 -c "import sys; sys.exit(float('$s') >= $MIN)"; then
    echo "  frame $f dropped (similarity ${s:-none}, or face too small to judge)"
  else IMGS="$IMGS --image $K/out/k$f.png $f 1.0"; used="$used $f"; fi
done
[ -n "$used" ] || { echo "[$(date +%H:%M)] not her: no keyframe passed the gate; no clip made"; exit 1; }
echo "[$(date +%H:%M)] d19s2_B on $H, her pinned at frames 0$used"
M=/models/ltx-2.5
MODELS="--transformer-path $M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path $M/vae/ltx-2.5-video-vae-bf16.safetensors --audio-vae-path $M/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upsampler-path $M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
LORA="--lora /work/$T/outputs/checkpoints/lora_weights_step_01000.safetensors 1.0"
rm -f $O/d19s2_B.mp4
ssh $H "cd ~/poptok && $SIDE flock $LOCK $(ltx_ctr $H) python3 -m ltx_pipelines.ic_lora $MODELS $LORA \
  --video-conditioning $O/layout_B.mp4 1.0 $IMGS \
  --stage-2-ic-lora --height 1088 --width 1920 --num-frames 81 --prompt \"$SHOT\" --seed 42 --frame-rate 24 \
  --output-path $O/d19s2_B.mp4" > $O/d19s2_B.log 2>&1 || { echo "  FAILED: $(grep Error $O/d19s2_B.log | tail -1)"; exit 1; }
rsync -a $H:poptok/$O/d19s2_B.mp4 $O/
python3 pipeline/eval_identity.py $PHOTO $O/d19s2_B.mp4 --every 8 --json $O/d19s2_B.id.json | tail -1
python3 - <<PY
import cv2, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
f16 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
O, B, K = "$O", "$B", "$K"
def lab(im, y, t, f=F):
    d = ImageDraw.Draw(im); d.rectangle((0, y - 3, d.textlength(t, font=f) + 10, y + f.size + 5), fill=0); d.text((5, y), t, font=f, fill=(255, 255, 0))
def ident(path):
    r = [x["similarity"] for x in json.load(open(path[:-4] + ".id.json")) if x["similarity"] is not None]
    return f"her? median {np.median(r):.2f} min {min(r):.2f} ({len(r)} faces)" if r else "her? no face found"
rows = [("base L2R + old first frame", f"{B}/l2r_first_B.mp4"), ("LoRA 1000 + old first frame", f"{O}/first_B.mp4"),
        ("LoRA 1000 + D12 first frame", f"{O}/first_her_B.mp4"), ("+ D19 prompt describes her", f"{O}/d19_B.mp4"),
        ("+ step 2: her at f" + ",".join(["0"] + "$used".split()), f"{O}/d19s2_B.mp4")]
W, H = 480, 272; idx = [0, 27, 54, 80]
sheet = Image.new("RGB", (W * 4, H * len(rows)))
for r, (name, path) in enumerate(rows):
    c = cv2.VideoCapture(path)
    for j, i in enumerate(idx):
        c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = c.read()
        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((W, H)) if ok else Image.new("RGB", (W, H))
        lab(im, 3, f"{r+1}{'abcd'[j]} f{i}" + (f"  {name}" if j == 0 else ""))
        if j == 0: lab(im, 240, ident(path))
        sheet.paste(im, (j * W, r * H))
sheet.save("$R/world-lora-1000-her-B.jpg", quality=88)
# the keyframes before and after the face pass (row 1 before, row 2 after)
ks = "$KF".split(); ks_im = Image.new("RGB", (W * len(ks), H * 2))
for j, f in enumerate(ks):
    for r, sub in enumerate(("in", "out")):
        try: im = Image.open(f"{K}/{sub}/k{f}.png").convert("RGB").resize((W, H))
        except FileNotFoundError: im = Image.new("RGB", (W, H))
        lab(im, 3, f"{r+1}{'ab'[j]} f{f} " + ("D19 frame" if r == 0 else "+ face pass")); ks_im.paste(im, (j * W, r * H))
ks_im.save("$R/d19s2-B-keys.jpg", quality=88)
# consecutive frames 16-47 of D19 then step 2
w, h = 240, 136; s = Image.new("RGB", (w * 8, h * 8))
cs = [(cv2.VideoCapture(f"{O}/d19_B.mp4"), "D19")] * 4 + [(cv2.VideoCapture(f"{O}/d19s2_B.mp4"), "step 2")] * 4
for r, (c, n) in enumerate(cs):
    for j in range(8):
        i = 16 + (r % 4) * 8 + j; c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = c.read()
        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((w, h)) if ok else Image.new("RGB", (w, h))
        lab(im, 2, f"{n} f{i}", f16); s.paste(im, (j * w, r * h))
s.save("$R/d19s2-B-frames-16-47.jpg", quality=88)
PY
cp $O/d19s2_B.mp4 $R/world-lora-1000-d19s2-B.mp4
echo "[$(date +%H:%M)] d19 step 2 done"
