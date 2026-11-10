"""One look across a video: does a shot with Amber look drawn by the same artist as a shot without her? (2026-11-10)
On a Spark (morph-spark container, from ~/poptok). Uses a piece's lean shots (cuckoo: 11 shots, 6 with Amber, one
charcoal style). Two methods are drawn here; two more already exist in PIECE/lean/keyframes/:
  mixed (existing)  shots without her by MOSS (NN-drawn.png), with her by Qwen "native" from her photo (NN-1..3.png)
  qwen  (existing)  shots without her by Qwen on a blank sheet (NN-qwen.png), with her native (NN-1..3.png)
  C     (new)       every shot by MOSS; for shots with her, Qwen redraws MOSS's still with her as the woman, given her
                    still (first, as the canvas) and her photo: MOSS sets every look, Qwen only puts her in            -> OUT/C/NN-t.png
  D     (new)       every shot by Qwen, each given one style still (ANCHOR, Qwen's shot 01) as a style reference;
                    shots with her also get her photo                                          -> OUT/D/NN[-t].png
Shots with her: 3 tries, seeds 1, 1001, 2001 (as lean/keyframes.py), each scored against her photo (SFace).
    python3 lean/consistency/run.py PIECE OUT METHOD [NN ...]       METHOD: C or D
"""
import json, os, sys

sys.path.insert(0, "pipeline")
import cv2
import numpy as np
from PIL import Image

import face_pass
from qwen_edit import load as load_edit

P, OUT, METHOD = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
only = set(sys.argv[4:])
K = f"{P}/lean/keyframes"
S = json.load(open(f"{P}/lean/shots.json"))
env = dict(l.split("=", 1) for l in open(f"{P}/piece.env").read().splitlines() if "=" in l and not l.startswith("#"))
photo = os.path.normpath(f"{P}/{env['PERFORMER_PHOTO'].strip()}")
NOT = env.get("STYLE_NOT", "").strip().strip('"')
avoid = f" Not {NOT}." if NOT else ""
ANCHOR = f"{K}/01-qwen.png"
TRIES = 3

C_PROMPT = ("Picture 1 is the finished drawing: keep it exactly, the same drawing medium and strokes, composition, "
            "setting, light, pose and clothes. Change only the young woman's face, so that she is the woman in picture 2: "
            "her face and features exactly. Her expression: {expression}. Picture 2 is only a reference for her face: "
            "exactly one woman in the picture, not picture 2's clothes, hair style or background. No text.")
D_HER = ("Draw the woman in picture 1 as {style}. {picture} The young woman in this picture is the woman in picture 1: "
         "keep her face and features exactly. Her expression: {expression}. Draw it in exactly the drawing style of "
         "picture 2: its medium, strokes, paper, tones and contrast; picture 2 is only a style reference, not its "
         "content. She wears the clothes described, not picture 1's, and the setting is the one described. Exactly one "
         "woman, never picture 1 itself pasted in. No text.")
D_NONE = ("Picture 1 is only a style reference: draw a new picture in exactly its drawing style (its medium, strokes, "
          "paper, tones and contrast), not its content. Draw: {prompt}. Only the people the description names. No text.")

edit = load_edit()
me = cv2.imread(photo); feat = face_pass.rec.feature(face_pass.rec.alignCrop(me, face_pass.face(me)))


def score(im):
    bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    face_pass.det.setInputSize((bgr.shape[1], bgr.shape[0])); _, fs = face_pass.det.detect(bgr)
    return None if fs is not None and len(fs) > 1 else face_pass.similarity(bgr, feat)   # two women: none (as keyframes.py)


os.makedirs(f"{OUT}/{METHOD}", exist_ok=True)
fp = f"{OUT}/{METHOD}/faces.json"
faces = json.load(open(fp)) if os.path.exists(fp) else {}
for i, s in enumerate(S["shots"]):
    n = f"{i + 1:02d}"
    if only and n not in only:
        continue
    her = s["amber"] != "none"
    expr = s.get("expression") or "as the picture describes"
    if METHOD == "C" and not her:
        continue                                                    # C keeps MOSS's own still (NN-drawn.png)
    if METHOD == "D" and not her:
        if n == "01":
            continue                                                # the anchor itself
        edit([ANCHOR], D_NONE.format(prompt=s["prompt"]) + avoid, seed=1).convert("RGB").save(f"{OUT}/D/{n}.png")
        print(f"D {n}: drawn from the anchor", flush=True); continue
    sims = []
    for t in range(TRIES):
        if METHOD == "C":
            refs, prompt = [f"{K}/{n}-drawn.png", photo], C_PROMPT.format(expression=expr)   # canvas first
        else:
            refs, prompt = [photo, ANCHOR], D_HER.format(style=S["style"], picture=s["picture"], expression=expr)
        im = edit(refs, prompt + avoid, seed=1 + 1000 * t).convert("RGB")
        im.save(f"{OUT}/{METHOD}/{n}-{t + 1}.png"); sims.append(score(im))
    faces[n] = sims
    json.dump(faces, open(fp, "w"), indent=1)
    print(f"{METHOD} {n}: her face " + ", ".join("-" if x is None else f"{x:.2f}" for x in sims), flush=True)
