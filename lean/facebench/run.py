"""Face-pass benchmark (D45), step 3, on a Spark: each case through each face-pass region (head = the current one,
whole, nohair), with the case's expression named and the treatment as the style; and "native": no stranger, no graft,
Qwen-Image-Edit draws the case's whole scene from her photo and the case's words. Every try saved,
OUT/<id>/<method>-<try>.png, scores in OUT/<id>/tries.json.
    python3 lean/facebench/run.py OUT PHOTO [ID ...]"""
import json, os, sys

sys.path.insert(0, "pipeline")
import cv2
import numpy as np

import face_pass
from qwen_edit import load as load_edit

NATIVE = ("Draw the woman in picture 1 as {style}: {scene}. Her expression: {expression}. Keep her face and features "
          "exactly; she wears the clothes described, not picture 1's, and the setting is the one described, not "
          "picture 1's background. No text.")
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json")))
out, photo, only = sys.argv[1], sys.argv[2], set(sys.argv[3:])
edit = load_edit(); fix = face_pass.load(edit)
me = cv2.imread(photo); feat = face_pass.rec.feature(face_pass.rec.alignCrop(me, face_pass.face(me)))


def save(d, method, tries, res):
    res[method] = []
    for t, x in enumerate(tries):
        if x["image"] is not None:
            x["image"].save(f"{d}/{method}-{t + 1}.png")
        res[method].append({"try": t + 1, "sim": x["sim"], "ok": x["ok"]})
    json.dump(res, open(f"{d}/tries.json", "w"), indent=1)
    print(f"{os.path.basename(d)} {method}: " + ", ".join(f"{x['sim']:.2f}" if x["sim"] is not None else "-"
                                                         for x in res[method]), flush=True)


for c in C["cases"]:
    if only and c["id"] not in only:
        continue
    d = f"{out}/{c['id']}"; os.makedirs(d, exist_ok=True)
    res = json.load(open(f"{d}/tries.json")) if os.path.exists(f"{d}/tries.json") else {}
    st = C["styles"][c["style"]]
    for region in ("head", "whole", "nohair"):
        if region not in res:
            fix(f"{out}/{c['id']}.png", photo, seed=1, tries=3, angles=os.path.splitext(photo)[0] + "-angles",
                region=region, style=st, expression=c["expression"], hair=f"{out}/{c['id']}-hair.png")
            save(d, region, fix.tries, res)
    if "native" not in res:
        tries = []
        for t in range(3):
            im = edit([photo], NATIVE.format(style=st, scene=c["scene"].format(woman="her"), expression=c["expression"]),
                      seed=1 + 1000 * t, size=(1024, 576)).convert("RGB")
            tries.append({"image": im, "ok": True,
                          "sim": face_pass.similarity(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), feat)})
        save(d, "native", tries, res)
