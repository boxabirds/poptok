"""poptok/lean step 4a: the keyframes (on a Spark, morph-spark container, from ~/poptok). Each shot's prompt drawn by
MOSS-Image; where Amber is in it (seen / sings), the face pass makes her her, asked to keep the style's medium (D39:
without it, a photographed face is pasted into the drawing).
    python3 lean/keyframes.py PIECE                 (reads PIECE/lean/shots.json; writes PIECE/lean/keyframes/)"""
import json, os, sys

sys.path.insert(0, "pipeline")
import torch

from workshop import read_env

P = sys.argv[1].rstrip("/"); e = read_env(f"{P}/piece.env")
S = json.load(open(f"{P}/lean/shots.json")); out = f"{P}/lean/keyframes"; os.makedirs(out, exist_ok=True)
from keyframe_model import load
r = load("moss", offload=False)
for i, s in enumerate(S["shots"]):
    if not os.path.exists(f"{out}/{i + 1:02d}.png"):
        r.render(None, [(s["prompt"], 1.0)], 1.0, seed=1, size=(1024, 576))[0].save(f"{out}/{i + 1:02d}.png")
        print(f"keyframe {i + 1}: drawn", flush=True)
del r; torch.cuda.empty_cache()
photo = os.path.normpath(f"{P}/{e['PERFORMER_PHOTO']}")
her = [i for i, s in enumerate(S["shots"]) if s["amber"] != "none"]
faces = json.load(open(f"{out}/faces.json")) if os.path.exists(f"{out}/faces.json") else {}
if her:
    import face_pass
    fix = face_pass.load()
    for i in her:
        n = f"{i + 1:02d}"
        if n in faces:
            continue
        os.replace(f"{out}/{n}.png", f"{out}/{n}-drawn.png") if not os.path.exists(f"{out}/{n}-drawn.png") else None
        im, sim = fix(f"{out}/{n}-drawn.png", photo, seed=1, tries=6, angles=os.path.splitext(photo)[0] + "-angles",
                      style=S["style"])
        im.save(f"{out}/{n}.png"); faces[n] = sim
        print(f"keyframe {i + 1}: face pass, her face {sim}", flush=True)
        json.dump(faces, open(f"{out}/faces.json", "w"), indent=1)
