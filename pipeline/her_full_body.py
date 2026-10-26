"""A full-length, A-pose image of the performer in a piece's wardrobe, from her photo (2026-10-26), as the input for a
photo-to-3D model (TRELLIS.2, docs/research/photo-to-3d-avatar.md). Her photos are a head-and-shoulders portrait and a
three-quarter-length shot with her arms crossed; image-to-3D fuses crossed arms into the body, so it needs her standing
straight, arms slightly out, head to feet, on a plain background.

    python3 pipeline/her_full_body.py PHOTO OUT_DIR --wardrobe "..." [--seeds 4] [--angles DIR]
Writes OUT_DIR/front_<seed>.png (the image, whole-head face pass applied) and OUT_DIR/front_<seed>_rgba.png (background
cut out by colour distance from the plain light-grey background: no background-removal model needed). Runs in
morph-spark on a Spark.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_pass import load as load_face
from qwen_edit import load

ap = argparse.ArgumentParser()
ap.add_argument("photo")
ap.add_argument("out")
ap.add_argument("--wardrobe", required=True)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--angles")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
PROMPT = ("A full-length photograph of the woman in the picture, the same person: keep her face, eyes, hair and age "
          "exactly. She stands straight facing the camera in an A-pose: arms held slightly away from her sides, hands open, "
          "feet a little apart, the whole body from the top of her head to her feet in frame with a margin, centred. She "
          "wears " + a.wardrobe + ". Her long wavy hair hangs down behind her shoulders. Plain, even, light grey studio "
          "background (RGB about 200,200,200), soft even front light, no shadow on the background. No text.")
edit = load()
fix = load_face(edit)
for s in range(a.seeds):
    im = edit([a.photo], PROMPT, seed=s, size=(768, 1344))
    im, sim = fix(im, a.photo, seed=s, angles=a.angles)
    im.save(f"{a.out}/front_{s}.png")
    rgb = np.array(im.convert("RGB")).astype(np.float32)
    bg = np.median(np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3),
                                   rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)]), axis=0)
    d = np.linalg.norm(rgb - bg, axis=2)                     # distance from the background colour
    alpha = np.clip((d - 12) / 18, 0, 1)
    alpha = np.array(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5)))
    Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha])).save(f"{a.out}/front_{s}_rgba.png")
    print(f"front_{s}: face similarity {sim}", flush=True)
