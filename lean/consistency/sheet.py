"""Sheet and a rough number for the one-look test (lean/consistency/run.py): one row per method, the piece's shots across.
    python3 lean/consistency/sheet.py PIECE OUT REVIEW_DIR
Rows: A mixed (today's default: MOSS without her, Qwen native with her), B qwen (RENDER=qwen: Qwen for every shot),
C (MOSS for every shot, Qwen puts her into the shots she's in), D (Qwen for every shot, anchored to one style still).
Shots with her show the try most like her (SFace; a try with two faces never wins), as lean/keyframes.py picks.
"Same artist?" (rough): each still described by 5 drawing statistics (mean brightness, contrast, share of near-black,
share of near-white, stroke density = Canny edge share); the gap between the average still with her and the average
still without her, in units of the typical spread within each group. 0 = indistinguishable; higher = two looks.
"""
import json, os, sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P, O, R = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"), sys.argv[3]
K = f"{P}/lean/keyframes"
S = json.load(open(f"{P}/lean/shots.json"))["shots"]
her = [s["amber"] != "none" for s in S]
kfaces = json.load(open(f"{K}/faces.json"))
F = {m: json.load(open(f"{O}/{m}/faces.json")) if os.path.exists(f"{O}/{m}/faces.json") else {} for m in "CD"}


def best(n, faces, pattern):
    sims = faces.get(n)
    if not sims: return None, None
    t = int(np.argmax([-2 if x is None else x for x in sims]))
    return pattern.format(n=n, t=t + 1), sims[t]


ROWS = [("A", "mixed (today)", lambda n, h: best(n, kfaces, K + "/{n}-{t}.png") if h else (f"{K}/{n}-drawn.png", None)),
        ("B", "Qwen for all", lambda n, h: best(n, kfaces, K + "/{n}-{t}.png") if h else (f"{K}/{n}-qwen.png", None)),
        ("C", "MOSS + Qwen puts her in", lambda n, h: best(n, F["C"], O + "/C/{n}-{t}.png") if h else (f"{K}/{n}-drawn.png", None)),
        ("D", "Qwen, one style still", lambda n, h: best(n, F["D"], O + "/D/{n}-{t}.png") if h else
         ((f"{K}/01-qwen.png" if n == "01" else f"{O}/D/{n}.png"), None))]


def stats(f):
    g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (512, 288))
    return np.array([g.mean() / 255, g.std() / 255, (g < 50).mean(), (g > 215).mean(), (cv2.Canny(g, 60, 160) > 0).mean()])


TW, TH, LH = 320, 180, 22
fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
im = Image.new("RGB", (len(S) * TW, 40 + len(ROWS) * (TH + LH + 30)), "white")
d = ImageDraw.Draw(im)
d.text((8, 10), "Same look across shots? Shots with Amber marked ★. Label: row+shot, face score (her photo, 0-1).", font=fb, fill=(0, 0, 0))
res = {}
for r, (key, name, get) in enumerate(ROWS):
    y = 40 + r * (TH + LH + 30)
    feats = {True: [], False: []}; sims = []
    for i, s in enumerate(S):
        n = f"{i + 1:02d}"
        f, sim = get(n, her[i])
        if f and os.path.exists(f):
            im.paste(Image.open(f).convert("RGB").resize((TW, TH)), (i * TW, y + LH + 26))
            feats[her[i]].append(stats(f))
            if sim is not None: sims.append(sim)
        lab = f"{key}{n}{' ★' if her[i] else ''}" + (f"  face {sim:.2f}" if sim is not None else "")
        d.rectangle([i * TW, y + LH + 26, i * TW + 7 + 9 * len(lab), y + LH + 46], fill=(255, 255, 255))
        d.text((i * TW + 4, y + LH + 27), lab, font=fnt, fill=(0, 0, 0))
    gap = None
    if feats[True] and feats[False]:
        a, b = np.array(feats[True]), np.array(feats[False])
        spread = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-6
        gap = float(np.linalg.norm((a.mean(0) - b.mean(0)) / spread) / np.sqrt(a.shape[1]))
    res[key] = {"name": name, "same_artist_gap": None if gap is None else round(gap, 2),
                "face_mean": round(float(np.mean(sims)), 2) if sims else None, "shots_with_her": len(sims)}
    d.text((8, y + 8), f"{key}  {name}    look gap (with vs. without her): "
           f"{'-' if gap is None else f'{gap:.2f}'}    her face, mean: {res[key]['face_mean']}", font=fb, fill=(0, 0, 0))
os.makedirs(R, exist_ok=True)
im.save(f"{R}/one-look-sheet.jpg", quality=88)
json.dump(res, open(f"{R}/one-look.json", "w"), indent=1)
for k, v in res.items(): print(k, v)
