"""Q1 (POC 22): can Amber be placed in a strong style, not just a photographic one? For a few shots of an existing run,
Qwen-Image-Edit (Lightning) re-renders her from her photo with a POC 24 style brief leading the prompt.

    python3 pipeline/amber_style_test.py OUT PLAN.json --photo PHOTO --shots 8,11,12,13 \
        --styles "stained glass,claymation,boxes and string"
OUT/shots.json gives the shot prompts; PLAN.json is a style_bench.py plan (its style briefs). Writes
OUT/amber_styles/<style>_shot_XXX.png. Runs in morph-spark on a Spark.
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("plan")
ap.add_argument("--photo", required=True)
ap.add_argument("--shots", default="8,11,12,13")
ap.add_argument("--styles", default="stained glass,claymation,boxes and string")
ap.add_argument("--dst", default="amber_styles")
a = ap.parse_args()
shots = {s["i"]: s for s in json.load(open(os.path.join(a.out, "shots.json")))["shots"]}
briefs = json.load(open(a.plan))["styles"]
dst = os.path.join(a.out, a.dst)
os.makedirs(dst, exist_ok=True)
from qwen_edit import load
edit = load()
for style in a.styles.split(","):
    for n in (int(x) for x in a.shots.split(",")):
        s = shots[n]
        prompt = (f"The whole image is in the {style} style: {briefs[style]['brief']} The woman from the reference image, "
                  f"the same person (keep her face, hair and age), rendered in this {style} style too. {s['prompt']}")
        t = time.time()
        edit([a.photo], prompt, seed=1000 + n).save(os.path.join(dst, f"{style.replace(' ', '_')}_shot_{n:03d}.png"))
        print(f"{style} shot {n}: {time.time() - t:.0f}s", flush=True)
