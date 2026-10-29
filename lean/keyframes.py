"""poptok/lean step 4a: the keyframes (on a Spark, morph-spark container, from ~/poptok).
  - without Amber: MOSS-Image draws the shot's prompt (NN-drawn.png, seed 1);
  - with Amber (seen / sings): "native" (D46, 2026-10-29): Qwen-Image-Edit draws the whole shot from her photo and the
    shot's words, style first, her expression named; 3 tries (NN-t.png). No stranger and no face graft: grafting her
    face onto a MOSS stranger passed 2 of 10 facebench cases (stamped hair, lost expressions), native 4 of 10, the
    charcoal ones among them (pieces/_bench/review/2026-10-29/facebench).
NN.png is the keyframe: the try the user picked (keyframes/picks.json, {"09": 2}), else the one most like her (SFace).
A shot is redrawn only when its prompt (or, with Amber, its expression) changed (prompts.json).
    python3 lean/keyframes.py PIECE                 (reads PIECE/lean/shots.json; writes PIECE/lean/keyframes/)"""
import json, os, shutil, sys

sys.path.insert(0, "pipeline")
import cv2
import numpy as np
import torch

from workshop import read_env

TRIES = 3
NATIVE = ("Draw the woman in picture 1 as {style}. {picture} The young woman in this picture is the woman in picture "
          "1: keep her face and features exactly. Her expression: {expression}. She wears the clothes described, not "
          "picture 1's, and the setting is the one described, not picture 1's background. No text.")

P = sys.argv[1].rstrip("/"); e = read_env(f"{P}/piece.env")
S = json.load(open(f"{P}/lean/shots.json")); out = f"{P}/lean/keyframes"; os.makedirs(out, exist_ok=True)
J = lambda n, d: json.load(open(f"{out}/{n}.json")) if os.path.exists(f"{out}/{n}.json") else d
prompts, faces, picks = J("prompts", {}), J("faces", {}), J("picks", {})
photo = os.path.normpath(f"{P}/{e['PERFORMER_PHOTO']}")
her = lambda s: s["amber"] != "none"
key = lambda s: s["prompt"] + (f" | native | {s.get('expression')}" if her(s) else "")   # what a drawing depends on
todo = [i for i, s in enumerate(S["shots"]) if prompts.get(f"{i + 1:02d}") != key(s)]


def done(n, s):
    prompts[n] = key(s)
    json.dump(prompts, open(f"{out}/prompts.json", "w"), indent=1, ensure_ascii=False)
    json.dump(faces, open(f"{out}/faces.json", "w"), indent=1)


mo = [i for i in todo if not her(S["shots"][i])]
if mo:
    from keyframe_model import load
    r = load("moss", offload=False)
    for i in mo:
        n = f"{i + 1:02d}"; s = S["shots"][i]
        r.render(None, [(s["prompt"], 1.0)], 1.0, seed=1, size=(1024, 576))[0].save(f"{out}/{n}-drawn.png")
        shutil.copy(f"{out}/{n}-drawn.png", f"{out}/{n}.png"); faces.pop(n, None); done(n, s)
        print(f"keyframe {i + 1}: drawn (MOSS)", flush=True)
    del r; torch.cuda.empty_cache()
na = [i for i in todo if her(S["shots"][i])]
if na:
    import face_pass
    from qwen_edit import load as load_edit
    edit = load_edit()
    me = cv2.imread(photo); feat = face_pass.rec.feature(face_pass.rec.alignCrop(me, face_pass.face(me)))
    for i in na:
        n = f"{i + 1:02d}"; s = S["shots"][i]
        sims = []
        for t in range(TRIES):
            im = edit([photo], NATIVE.format(style=S["style"], picture=s["picture"],
                                             expression=s.get("expression") or "as the picture describes"),
                      seed=1 + 1000 * t, size=(1024, 576)).convert("RGB")
            im.save(f"{out}/{n}-{t + 1}.png")
            sims.append(face_pass.similarity(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), feat))
        faces[n] = sims; picks.pop(n, None)
        done(n, s)
        print(f"keyframe {i + 1}: native, her face " + ", ".join("-" if x is None else f"{x:.2f}" for x in sims), flush=True)
    json.dump(picks, open(f"{out}/picks.json", "w"), indent=1)
for i, s in enumerate(S["shots"]):          # the keyframe: the user's pick, else the try most like her
    n = f"{i + 1:02d}"
    if her(s) and isinstance(faces.get(n), list):
        t = picks.get(n) or 1 + int(np.argmax([-2 if x is None else x for x in faces[n]]))
        shutil.copy(f"{out}/{n}-{t}.png", f"{out}/{n}.png")
