"""poptok/lean step 4a: the keyframes (on a Spark, morph-spark container, from ~/poptok). Each shot's prompt drawn by
MOSS-Image (NN-drawn.png); where Amber is in it (seen / sings), the face pass makes her her, asked to keep the style's
medium (D39: without it, a photographed face is pasted into the drawing); NN.png is the keyframe. A keyframe is redrawn
only when its prompt changed (prompts.json), and face-passed again only when it was redrawn or newly hers.
    python3 lean/keyframes.py PIECE                 (reads PIECE/lean/shots.json; writes PIECE/lean/keyframes/)"""
import json, os, shutil, sys

sys.path.insert(0, "pipeline")
import torch

from workshop import read_env

P = sys.argv[1].rstrip("/"); e = read_env(f"{P}/piece.env")
S = json.load(open(f"{P}/lean/shots.json")); out = f"{P}/lean/keyframes"; os.makedirs(out, exist_ok=True)
J = lambda n, d: json.load(open(f"{out}/{n}.json")) if os.path.exists(f"{out}/{n}.json") else d
prompts, faces = J("prompts", {}), J("faces", {})
todo = [i for i, s in enumerate(S["shots"]) if prompts.get(f"{i + 1:02d}") != s["prompt"]
        or not os.path.exists(f"{out}/{i + 1:02d}-drawn.png")]
if todo:
    from keyframe_model import load
    r = load("moss", offload=False)
    for i in todo:
        n = f"{i + 1:02d}"
        r.render(None, [(S["shots"][i]["prompt"], 1.0)], 1.0, seed=1, size=(1024, 576))[0].save(f"{out}/{n}-drawn.png")
        prompts[n] = S["shots"][i]["prompt"]; faces.pop(n, None)
        if os.path.exists(f"{out}/{n}.png"):
            os.remove(f"{out}/{n}.png")
        json.dump(prompts, open(f"{out}/prompts.json", "w"), indent=1, ensure_ascii=False)
        print(f"keyframe {i + 1}: drawn", flush=True)
    del r; torch.cuda.empty_cache()
photo = os.path.normpath(f"{P}/{e['PERFORMER_PHOTO']}")
fix = None
for i, s in enumerate(S["shots"]):
    n = f"{i + 1:02d}"
    if s["amber"] == "none":
        faces.pop(n, None)
        if not os.path.exists(f"{out}/{n}.png"):
            shutil.copy(f"{out}/{n}-drawn.png", f"{out}/{n}.png")
        continue
    if n in faces and os.path.exists(f"{out}/{n}.png"):
        continue
    if fix is None:
        import face_pass
        fix = face_pass.load()
    im, sim = fix(f"{out}/{n}-drawn.png", photo, seed=1, tries=6, angles=os.path.splitext(photo)[0] + "-angles",
                  style=S["style"])
    im.save(f"{out}/{n}.png"); faces[n] = sim
    json.dump(faces, open(f"{out}/faces.json", "w"), indent=1)
    print(f"keyframe {i + 1}: face pass, her face {sim}", flush=True)
