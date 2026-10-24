"""Restyle pass (POC 22 Q1, approach C): Qwen-Image-Edit places the performer well but renders the look photographically;
a light MOSS image-to-image pass over each of her stills, with the shot's prompt and the piece's style, brings them into
the same look as MOSS's own shots while keeping who she is (checked with eval_identity.py).

    python3 pipeline/restyle.py OUT --s0 0.5 [--keys keys --dst keys_restyled] [--only-cast]
s0 = how much is redrawn (MOSS noise level: 0 keeps the still, 1 ignores it). Reads OUT/shots.json for prompts and style.
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--s0", type=float, default=0.5)
ap.add_argument("--keys", default="keys")
ap.add_argument("--dst", default=None, help="default: keys_s<s0>")
ap.add_argument("--only-cast", action="store_true", help="only the performer's shots (scene / sing / part)")
ap.add_argument("--style", default=None, help="replace the piece's style (e.g. a POC 24 style brief)")
ap.add_argument("--lead", default="", help="words put before each prompt (e.g. 'stained glass:')")
a = ap.parse_args()
from PIL import Image
from keyframe_model import load

plan = json.load(open(os.path.join(a.out, "shots.json")))
style = a.style if a.style is not None else plan.get("style", "")
dst = os.path.join(a.out, a.dst or f"keys_s{a.s0:g}")
os.makedirs(dst, exist_ok=True)
r = load("moss", offload=False)
for s in plan["shots"]:
    src = os.path.join(a.out, a.keys, f"shot_{s['i']:03d}.png")
    out = os.path.join(dst, f"shot_{s['i']:03d}.png")
    if os.path.exists(out) or not os.path.exists(src):
        continue
    if a.only_cast and s.get("cast") not in ("scene", "sing", "part"):
        Image.open(src).save(out)
        continue
    t = time.time()
    p = f"{a.lead} {s['prompt']}".strip()
    im, _ = r.render(Image.open(src).convert("RGB"), [(f"{p}, {style}" if style else p, 1.0)], a.s0,
                     seed=1000 + s["i"])
    im.save(out)
    print(f"shot {s['i']:3d} {s.get('cast', '-'):5s} s0 {a.s0:g} {time.time() - t:.0f}s", flush=True)
