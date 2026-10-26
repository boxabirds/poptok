"""The performer's head from several angles, made once from her photo (2026-10-26), so the face pass has a reference
that matches the still: given only a frontal photo, Qwen-Image-Edit made her side and three-quarter faces into other
people (universe sheets, every "a" shot), while frontal shots held.

    python3 pipeline/her_angles.py PHOTO OUT_DIR [--seeds 3]
Writes OUT_DIR/<angle>.png (head and shoulders, plain grey background, soft even light) for each angle, the best of
--seeds tries by SFace similarity to the photo, and OUT_DIR/angles.json {angle: {"turn": t, "similarity": s}}, where
turn is the measured head turn (face_pass.turn: nose offset from between the eyes, about -1..1). face_pass.py picks the
angle whose turn is nearest the still's. Runs in morph-spark on a Spark.
"""
from __future__ import annotations

import argparse, json, os, sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_identity import rec
from face_pass import face, similarity, turn
from qwen_edit import load

ANGLES = {
    "front": "facing the camera straight on",
    "three-quarter-left": "her head turned about 45 degrees to her right, so we see the left side of her face more",
    "three-quarter-right": "her head turned about 45 degrees to her left, so we see the right side of her face more",
    "profile-left": "her head turned 90 degrees to her right: an exact side profile, we see only the left side of her face",
    "profile-right": "her head turned 90 degrees to her left: an exact side profile, we see only the right side of her face",
}
PROMPT = ("A head-and-shoulders portrait of the woman in the picture, the same person: keep her face, eyes, hair and "
          "age exactly. She is {pose}. Neutral expression, plain mid-grey background, soft even studio light. No text.")

ap = argparse.ArgumentParser()
ap.add_argument("photo")
ap.add_argument("out")
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
p = cv2.imread(a.photo)
ref = rec.feature(rec.alignCrop(p, face(p)))
edit = load()
meta = {}
for name, pose in ANGLES.items():
    best = None
    for s in range(a.seeds):
        im = edit([a.photo], PROMPT.format(pose=pose), seed=s, size=(768, 1024))
        bgr = cv2.cvtColor(__import__("numpy").array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        sim, f = similarity(bgr, ref), face(bgr)
        if f is not None and (best is None or (sim or -1) > best[1]):
            best = (im, sim or -1, turn(f))
    if best:
        best[0].save(f"{a.out}/{name}.png")
        meta[name] = {"turn": best[2], "similarity": best[1]}
        print(f"{name}: turn {best[2]:+.2f}, similarity {best[1]:.2f}", flush=True)
json.dump(meta, open(f"{a.out}/angles.json", "w"), indent=1)
