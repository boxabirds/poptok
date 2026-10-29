"""D39 keyframes (on a Spark, morph-spark container, from ~/poptok), after D38 test 2 found the style a cliff: none at
75%, full charcoal at 100%, and at 100% she is no longer Amber. Two things:
  1. the cliff: the 0% reference (l000.png) repainted at 80, 85, 90 and 95%;
  2. her in full charcoal: the 100% keyframe face-passed with her photo, as is (f100) and asked to keep the drawing's
     medium (s100); and her photo itself, cropped to 16:9 round her face, repainted at 90 and 95% (p090, p095).
Writes k<id>.png and OUT/levels2.json ({id, label, face}, same face score as levels.json).

    python3 lean/poc_sweep2.py OUT STYLE_FILE PHOTO     (OUT holds D38's sweep: l000.png, l100.png)
"""
import json, os, sys

sys.path.insert(0, "pipeline")
import cv2
import torch
from PIL import Image

SINGER = ("a young woman with long straight light-blonde hair and a dark heavy coat, medium close-up, her face and "
          "shoulders filling the frame, in a dark forest, looking up, singing with her mouth open, forlorn")  # = poc_sweep.py's

CLIFF = [80, 85, 90, 95]
out, style, photo = sys.argv[1], open(sys.argv[2]).read().strip(), sys.argv[3]
import face_pass
me = cv2.imread(photo); f = face_pass.face(me)
H, W = me.shape[:2]; h = int(W * 9 / 16); y = int(min(max(f[1] + f[3] / 2 - h / 2, 0), H - h))
cv2.imwrite(f"{out}/photo169.png", me[y:y + h])          # the photo at the video's shape, her face in it

from keyframe_model import load
r = load("moss", offload=False)
todo = [(f"k{lv:03d}", f"{out}/l000.png", lv) for lv in CLIFF] + \
       [(f"kp{lv:03d}", f"{out}/photo169.png", lv) for lv in (90, 95)]
for name, src, lv in todo:
    if not os.path.exists(f"{out}/{name}.png"):
        r.render(Image.open(src).convert("RGB"), [(f"{style}: {SINGER}", 1.0)], lv / 100, seed=1,
                 size=(1024, 576))[0].save(f"{out}/{name}.png")
        print(f"{name}: done", flush=True)
del r; torch.cuda.empty_cache()
fix = face_pass.load()
for name, st in (("kf100", None), ("ks100", style)):
    if not os.path.exists(f"{out}/{name}.png"):
        im, sim = fix(f"{out}/l100.png", photo, seed=1, tries=6, angles=os.path.splitext(photo)[0] + "-angles", style=st)
        im.save(f"{out}/{name}.png"); print(f"{name}: face {sim}", flush=True)
feat = face_pass.rec.feature(face_pass.rec.alignCrop(me, f))
rows = [(f"{lv:03d}", f"{lv}% style") for lv in CLIFF] + \
       [("f100", "100% + face pass"), ("s100", "100% + face pass kept in charcoal"),
        ("p090", "her photo, 90% style"), ("p095", "her photo, 95% style")]
res = [{"id": i, "label": lab, "face": face_pass.similarity(cv2.imread(f"{out}/k{i}.png"), feat)} for i, lab in rows]
json.dump(res, open(f"{out}/levels2.json", "w"), indent=1)
print("levels2:", res, flush=True)
