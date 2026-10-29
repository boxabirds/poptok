"""Face-pass benchmark (D45) sheet: per case, the drawing, then each region's tries (3 each), labelled with her face
score; a try that moved the face (rejected) is grey. Two views: whole frames, and the heads close up.
    python3 lean/facebench/sheet.py OUT REVIEW_DIR"""
import json, os, sys

import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "pipeline")
from face_pass import face

O, R = sys.argv[1], sys.argv[2]
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json")))["cases"]
REG = ("head", "whole", "nohair", "native")
for view in ("frames", "heads"):
    tw, th = (320, 180) if view == "frames" else (220, 220)
    S = Image.new("RGB", (tw * (1 + 3 * len(REG)), th * len(C)), (90, 90, 90))
    for r, c in enumerate(C):
        src = f"{O}/{c['id']}.png"
        if not os.path.exists(f"{O}/{c['id']}/tries.json"):
            continue
        T = json.load(open(f"{O}/{c['id']}/tries.json")); f = face(cv2.imread(src))

        def tile(p, lab):
            im = Image.open(p).convert("RGB")
            if view == "heads" and f is not None:
                cx, cy, s = f[0] + f[2] / 2, f[1] + f[3] / 2, max(f[2], f[3]) * 3.0
                im = im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))
            im = im.resize((tw, th)); d = ImageDraw.Draw(im)
            d.rectangle((0, 0, d.textlength(lab, font=F) + 8, 20), fill=0); d.text((4, 1), lab, font=F, fill=(255, 255, 0))
            return im
        S.paste(tile(src, f"{c['id']} drawing"), (0, r * th))
        for g, reg in enumerate(REG):
            for x in T.get(reg, []):
                p = f"{O}/{c['id']}/{reg}-{x['try']}.png"
                if os.path.exists(p):
                    S.paste(tile(p, f"{c['id']} {reg}{x['try']} " + (f"{x['sim']:.2f}" if x["sim"] is not None else "-")),
                            ((1 + g * 3 + x["try"] - 1) * tw, r * th))
    S.save(f"{R}/45-facebench-{view}.jpg", quality=85)
print(f"{R}/45-facebench-frames.jpg, {R}/45-facebench-heads.jpg")
