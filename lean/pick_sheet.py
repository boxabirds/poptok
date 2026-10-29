"""Her tries per Amber keyframe (lean/keyframes.py, native), for the user to pick from (D44): per row her photo, then
the tries, each labelled with the keyframe, try and face score (SFace, >= 0.363 = her), and Claude's verdict by eye if
keyframes/verdicts.json has one ({"09": "yn?"}: green tick, red cross, amber ?). The user's picks go in
keyframes/picks.json ({"09": 2}).
    python3 lean/pick_sheet.py PIECE OUT.jpg"""
import json, os, sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "pipeline")
from workshop import read_env

P, OUT = sys.argv[1].rstrip("/"), sys.argv[2]; K = f"{P}/lean/keyframes"; e = read_env(f"{P}/piece.env")
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
M = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
MARK = {"y": ("✔", (0, 200, 0)), "n": ("✘", (230, 0, 0)), "?": ("?", (255, 170, 0))}
S = json.load(open(f"{P}/lean/shots.json"))["shots"]; faces = json.load(open(f"{K}/faces.json"))
V = json.load(open(f"{K}/verdicts.json")) if os.path.exists(f"{K}/verdicts.json") else {}
rows = [(f"{i + 1:02d}", s) for i, s in enumerate(S) if isinstance(faces.get(f"{i + 1:02d}"), list)]
W, H = 512, 288
sheet = Image.new("RGB", (W * 4, H * len(rows)), (90, 90, 90))


def lab(im, t, v=None):
    d = ImageDraw.Draw(im); d.rectangle((0, 0, d.textlength(t, font=F) + 10, 24), fill=0)
    d.text((5, 2), t, font=F, fill=(255, 255, 0))
    if v in MARK:
        g, c = MARK[v]; d.ellipse((W - 80, H - 80, W - 6, H - 6), fill="white")
        d.text((W - 43, H - 43), g, font=M, fill=c, anchor="mm")
    return im


for r, (n, s) in enumerate(rows):
    ph = Image.open(os.path.normpath(f"{P}/{e['PERFORMER_PHOTO']}")).convert("RGB"); ph.thumbnail((W, H))
    t0 = Image.new("RGB", (W, H), (40, 40, 40)); t0.paste(ph, ((W - ph.width) // 2, 0))
    sheet.paste(lab(t0, f"{int(n)}: Amber (her photo)"), (0, r * H))
    for t, sim in enumerate(faces[n]):
        im = Image.open(f"{K}/{n}-{t + 1}.png").convert("RGB").resize((W, H))
        sc = "-" if sim is None else f"{sim:.2f}"
        sheet.paste(lab(im, f"{int(n)}.{t + 1}  {s['amber']}  face {sc}", (V.get(n) or "")[t:t + 1] or None),
                    ((t + 1) * W, r * H))
sheet.save(OUT, quality=88)
print(f"pick sheet: {OUT}")
