"""Face-pass benchmark (D45), step 2, on gruntus (segment image): each case's hair mask (Grounding DINO "hair" boxes,
SAM 2.1 masks, segment_check.Segmenter), OUT/<id>-hair.png (white = hair), for the face pass's "nohair" region.
    python3 lean/facebench/hair.py OUT"""
import glob, os, sys

sys.path.insert(0, "pipeline")
import numpy as np
from PIL import Image

from segment_check import Segmenter

S = Segmenter()
for p in sorted(glob.glob(f"{sys.argv[1]}/*.png")):
    if "-" in os.path.basename(p):
        continue
    im = Image.open(p).convert("RGB")
    dets = S.detect(im, groups=[(["hair"], 0.25)])
    ms = S.masks(im, dets)
    m = np.any(ms, axis=0) if ms else np.zeros(im.size[::-1], bool)
    Image.fromarray((m * 255).astype(np.uint8)).save(p[:-4] + "-hair.png")
    print(f"{os.path.basename(p)}: hair {m.mean():.1%} of the frame ({len(dets)} boxes)", flush=True)
