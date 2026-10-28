"""D38 test 2, the keyframes (on a Spark, morph-spark container, from ~/poptok): Amber singing, as a photograph made
her by the face pass (the 0% reference), then that frame repainted by MOSS toward the style at each intensity (img2img,
noise level = intensity; 100%: drawn afresh from the prompt). No face pass on the styled frames: whether she survives
the style is the question. Her face score (SFace cosine, same person >= 0.363) per level in OUT/levels.json.

    python3 lean/poc_sweep.py OUT STYLE_FILE PHOTO
"""
import json, os, sys, time

sys.path.insert(0, "pipeline")
import torch

LEVELS = [0, 10, 25, 50, 75, 100]
SINGER = ("a young woman with long straight light-blonde hair and a dark heavy coat, medium close-up, her face and "
          "shoulders filling the frame, in a dark forest, looking up, singing with her mouth open, forlorn")
out, style, photo = sys.argv[1], open(sys.argv[2]).read().strip(), sys.argv[3]
os.makedirs(out, exist_ok=True)
from keyframe_model import load
r = load("moss", offload=False)
if not os.path.exists(f"{out}/ref_world.png"):
    r.render(None, [(f"A photograph: {SINGER}, soft overcast light, natural colour", 1.0)], 1.0, seed=1,
             size=(1024, 576))[0].save(f"{out}/ref_world.png")
del r; torch.cuda.empty_cache()
import face_pass
if not os.path.exists(f"{out}/l000.png"):
    fix = face_pass.load()
    im, sim = fix(f"{out}/ref_world.png", photo, seed=1, tries=6, angles=os.path.splitext(photo)[0] + "-angles")
    im.save(f"{out}/l000.png"); print(f"reference: face {sim}", flush=True)
    del fix; torch.cuda.empty_cache()
from PIL import Image
r = load("moss", offload=False)
ref = Image.open(f"{out}/l000.png").convert("RGB")
for lv in LEVELS[1:]:
    t = time.time()
    r.render(ref, [(f"{style}: {SINGER}", 1.0)], lv / 100, seed=1, size=(1024, 576))[0].save(f"{out}/l{lv:03d}.png")
    print(f"level {lv}%: {time.time() - t:.0f} s", flush=True)
import cv2
me = cv2.imread(photo); feat = face_pass.rec.feature(face_pass.rec.alignCrop(me, face_pass.face(me)))
res = [{"level": lv, "face": face_pass.similarity(cv2.imread(f"{out}/l{lv:03d}.png"), feat)} for lv in LEVELS]
json.dump(res, open(f"{out}/levels.json", "w"), indent=1)
print("levels:", res, flush=True)
