"""Fix a detail in a keyframe by instruction (Qwen-Image-Edit-2511), instead of redrawing the whole still with a new
seed: "make the symbol on the cloth a peace symbol", "remove the rifle", "the torso is the young woman's from picture 2".
The rest of the still is kept. With "photo": true her reference photo goes in as picture 2, for fixes about her.

    python3 pipeline/fix_keys.py OUT FIXES.json [--photo PHOTO] [--device cuda|mps] [--models DIR] [--in-place]
FIXES.json: [{"key": "shot_042.png", "fix": "instruction", "photo": false}, ...]
Writes OUT/keys_fixed/<key> (and a before|after sheet OUT/keys_fixed/<key>.cmp.jpg); --in-place also replaces
OUT/keys/<key>, keeping the original as OUT/keys/<key>.orig.png.
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deps, qwen_edit

KEEP = ("Keep everything else in picture 1 exactly as it is: the composition, camera angle, lighting, colours, style "
        "and every other person and object. Change only this: ")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("fixes")
ap.add_argument("--photo")
ap.add_argument("--device", default="cuda")
ap.add_argument("--models", default="/models", help="the folder holding Qwen-Image-Edit-2511 (+ -Lightning)")
ap.add_argument("--in-place", action="store_true")
a = ap.parse_args()

qwen_edit.MODEL = f"{a.models}/Qwen-Image-Edit-2511"
edit = qwen_edit.load(lightning=f"{a.models}/Qwen-Image-Edit-2511-Lightning/"
                      "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", device=a.device)
dst = os.path.join(a.out, "keys_fixed")
os.makedirs(dst, exist_ok=True)
for f in json.load(open(a.fixes)):
    src = os.path.join(a.out, "keys", f["key"])
    im = Image.open(src).convert("RGB")
    refs = [im] + ([Image.open(a.photo).convert("RGB")] if f.get("photo") and a.photo else [])
    t = time.time()
    out = edit(refs, KEEP + f["fix"], seed=f.get("seed", 0), size=im.size)
    out.save(os.path.join(dst, f["key"]))
    cmp = Image.new("RGB", (im.width * 2, im.height))
    cmp.paste(im, (0, 0)); cmp.paste(out.resize(im.size), (im.width, 0))
    cmp.resize((im.width, im.height // 2)).save(os.path.join(dst, f["key"] + ".cmp.jpg"), quality=88)
    if a.in_place:
        if not os.path.exists(src + ".orig.png"):
            shutil.copy(src, src + ".orig.png")
        out.save(src)
    print(f"{f['key']}: {time.time() - t:.0f} s  {f['fix']}", flush=True)
if a.in_place:
    # a fix keeps the shot (composition, cast): the merged plan and scene list stay valid. The video and lip-sync
    # depend on the stills, so they are made again from the fixed ones (pipeline/deps.py)
    deps.accept(a.out, "merge", "scenes")
