#!/usr/bin/env bash
# D38 test 1 on a free Spark (MOSS needs ~33 GB), then the labelled sheet.
#   setsid nohup bash lean/poc_medium.sh >> pieces/_logs/lean-poc.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
R=pieces/_bench/review/2026-10-28/lean-poc; O=pieces/_bench/lean/medium
mkdir -p $R $O
H=$(pick_host thing1 thing2)
echo "[$(date +%H:%M)] test 1 (medium) on $H"
ssh $H "mkdir -p ~/poptok/$O" && rsync -a pipeline lean $H:poptok/
ssh $H "cd ~/poptok && flock $LOCK $(keyframe_ctr $H) python3 lean/poc_medium.py $O" 2>&1 | grep -E "seed|rror|Traceback"
rsync -a $H:poptok/$O/ $O/
python3 - <<PY
import json
from PIL import Image, ImageDraw, ImageFont
O, R = "$O", "$R"
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
W = json.load(open(f"{O}/wordings.json")); w, h = 768, 432
S = Image.new("RGB", (w * 2, h * len(W)))
for r, x in enumerate(W):
    for c in (0, 1):
        im = Image.open(f"{O}/{x['name'][0]}{c + 1}.png").convert("RGB").resize((w, h)); d = ImageDraw.Draw(im)
        t = f"{x['name'][0]}{c + 1}  {x['name'][2:]}" + ("" if c else f"  (seed {c + 1})")
        d.rectangle((0, 0, d.textlength(t, font=F) + 12, 30), fill=0); d.text((6, 4), t, font=F, fill=(255, 255, 0))
        S.paste(im, (c * w, r * h))
S.save(f"{R}/1-medium.jpg", quality=88)
open(f"{R}/1-medium-prompts.md", "w").write("\n\n".join(f"**{x['name']}**: {x['prompt']}" + (f"  \n*negative:* {x['negative']}" if x['negative'] else "") for x in W) + "\n")
PY
echo "[$(date +%H:%M)] test 1 done: $R/1-medium.jpg"
