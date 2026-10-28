#!/usr/bin/env bash
# D12 (approved 2026-10-28): D11's face pass could not make the first frames her (best of 12 tries: 0.36 and 0.42, gate
# 0.5; her own photos score 0.55-0.92) because the person described for the restyle contradicted her photo (dark hair,
# camouflage face paint) and the face pass keeps hair colour and face marks. The person is now described from her photo
# (pieces/artists/r.amber.jpeg), the director's felt uniform kept; both clay first frames restyled with it, then the face
# pass gated at 0.5 on faces >= 80 px (shot A's face is ~38 px: too small to judge, reported); only then the LoRA 1000
# first-frame clips, each scored every 8 frames, scores on the sheets. thing2, under its GPU lock (between song3 chunks).
# Out: pieces/_bench/review/2026-10-28/world-lora-1000-her-{A,B}.jpg, -first-her-{A,B}.mp4, d12-first-frames.jpg
#   setsid nohup bash pipeline/launchers/d12-identity.sh >> pieces/_logs/d12-identity.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
H=thing2; F=pieces/_bench/world/first; V=pieces/_bench/world/first_d12; G=$V/her; O=pieces/_bench/world/lora1000
T=pieces/_bench/train/v2v-ic-lora-noble-felt-room-20261026; R=pieces/_bench/review/2026-10-28
PHOTO=pieces/artists/r.amber.jpeg; MIN=0.5
PERSON="a young woman with long straight light-blonde hair, blue eyes and fair skin, a bare clean face, in a worn dark green-brown felted wool field uniform with long sleeves and boots"
mkdir -p $V/views $G $R
cp $F/views/000_clay.png $F/views/001_clay.png $F/style.json $V/views/
echo "[$(date +%H:%M)] restyle with her described from her photo, then face pass (12 tries, gate $MIN on faces >= 80 px)"
ssh $H "mkdir -p ~/poptok/$V ~/poptok/pieces/artists" && rsync -a pipeline $H:poptok/ && rsync -a $V/ $H:poptok/$V/ \
  && rsync -a pieces/artists/ $H:poptok/pieces/artists/
ssh $H "cd ~/poptok && flock $LOCK $(keyframe_ctr $H) bash -c 'python3 pipeline/world/style_views.py $V/views $V/views/style.json \
  --person \"$PERSON\" && python3 pipeline/face_pass.py $PHOTO $V/views/000_styled.png $V/views/001_styled.png --out $G \
  --tries 12 --angles pieces/artists/r.amber-angles --min-sim $MIN --min-face 80'" > $V/prep.log 2>&1
rc=$?; grep -E "similarity|below|too small|no face|Error" $V/prep.log
rsync -a $H:poptok/$V/ $V/
python3 - <<PY
from PIL import Image, ImageDraw, ImageFont
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
rows = [("old first frame", "$F/first_{}.png"), ("restyled, her described", "$V/views/{}_styled.png"),
        ("+ face pass", "$G/{}_styled.png")]
W, H = 960, 544; s = Image.new("RGB", (W * 2, H * 3))
for r, (n, pat) in enumerate(rows):
    for j, (shot, num) in enumerate((("A", "000"), ("B", "001"))):
        try: im = Image.open(pat.format(shot if r == 0 else num)).convert("RGB").resize((W, H))
        except FileNotFoundError: im = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(im); lab = f"{r+1}{'ab'[j]} shot {shot}  {n}"
        d.rectangle((0, 0, d.textlength(lab, font=F) + 10, 30), fill=0); d.text((5, 3), lab, font=F, fill=(255, 255, 0))
        s.paste(im, (j * W, r * H))
s.save("$R/d12-first-frames.jpg", quality=88)
PY
[ $rc -eq 0 ] || { echo "[$(date +%H:%M)] not her (gate $MIN): no video made from these frames (sheet: $R/d12-first-frames.jpg)"; exit 1; }
cp $G/000_styled.png $O/first_her_A.png; cp $G/001_styled.png $O/first_her_B.png
rm -f $O/first_her_A.mp4 $O/first_her_B.mp4
rsync -a $O/first_her_A.png $O/first_her_B.png $H:poptok/$O/
M=/models/ltx-2.5
MODELS="--transformer-path $M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path $M/vae/ltx-2.5-video-vae-bf16.safetensors --audio-vae-path $M/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upsampler-path $M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
LORA="--lora /work/$T/outputs/checkpoints/lora_weights_step_01000.safetensors 1.0"
for p in A B; do
  k=$([ $p = A ] && echo 0 || echo 1)
  SHOT=$(python3 -c "import yaml;print(yaml.safe_load(open('$T/config.yaml'))['validation']['samples'][$k]['prompt'])")
  echo "[$(date +%H:%M)] first_her_$p"
  ssh $H "cd ~/poptok && flock $LOCK $(ltx_ctr $H) python3 -m ltx_pipelines.ic_lora $MODELS $LORA \
    --video-conditioning $O/layout_$p.mp4 1.0 --image $O/first_her_$p.png 0 1.0 \
    --stage-2-ic-lora --height 1088 --width 1920 --num-frames 81 --prompt \"$SHOT\" --seed 42 --frame-rate 24 \
    --output-path $O/first_her_$p.mp4" > $O/first_her_$p.log 2>&1 || { echo "  FAILED: $(grep Error $O/first_her_$p.log | tail -1)"; exit 1; }
  rsync -a $H:poptok/$O/first_her_$p.mp4 $O/
done
B=pieces/_bench/world/flat_dressed/ctl
for p in A B; do
  for v in $B/l2r_first_$p.mp4 $O/first_$p.mp4 $O/first_her_$p.mp4; do
    python3 pipeline/eval_identity.py $PHOTO $v --every 8 --json ${v%.mp4}.id.json | tail -1
  done
done
python3 - <<PY
import cv2, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
O, B = "$O", "$B"
def ident(path):
    r = [x["similarity"] for x in json.load(open(path[:-4] + ".id.json")) if x["similarity"] is not None]
    return f"her? median {np.median(r):.2f} min {min(r):.2f} ({len(r)} faces)" if r else "her? no face found"
for p in "AB":
    rows = [("base L2R + old first frame", f"{B}/l2r_first_{p}.mp4"), ("LoRA 1000 + old first frame", f"{O}/first_{p}.mp4"),
            ("LoRA 1000 + D12 first frame", f"{O}/first_her_{p}.mp4")]
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
    sheet.save(f"$R/world-lora-1000-her-{p}.jpg", quality=88)
PY
cp $O/first_her_A.mp4 $R/world-lora-1000-first-her-A.mp4; cp $O/first_her_B.mp4 $R/world-lora-1000-first-her-B.mp4
echo "[$(date +%H:%M)] d12 done"
