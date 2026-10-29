"""The storyboard as one sheet: every keyframe in order, labelled with its number, time, lines, whether Amber is in it
(and her face score), and the directors' beat, to review the video's overall shape.
    python3 lean/storyboard_sheet.py PIECE OUT.jpg"""
import json, os, sys, textwrap

from PIL import Image, ImageDraw, ImageFont

P, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
S = json.load(open(f"{P}/lean/shots.json"))["shots"]; K = f"{P}/lean/keyframes"
segs = json.load(open([f"{P}/analysis/{x}" for x in os.listdir(f"{P}/analysis") if x.endswith("-true-segments.json")][0]))["segments"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from workshop import read_env
RQ = read_env(f"{P}/piece.env").get("RENDER") == "qwen"     # lean/keyframes.py: Qwen draws every panel
faces = json.load(open(f"{K}/faces.json")) if os.path.exists(f"{K}/faces.json") else {}
picks = json.load(open(f"{K}/picks.json")) if os.path.exists(f"{K}/picks.json") else {}
for n, v in list(faces.items()):            # native: a score per try; the keyframe is the pick, else the best
    if isinstance(v, list):
        t = picks.get(n) or 1 + max(range(len(v)), key=lambda j: -2 if v[j] is None else v[j])
        faces[n] = v[t - 1]
W, H, T, C = 640, 360, 150, 3
sheet = Image.new("RGB", (W * C, (H + T) * ((len(S) + C - 1) // C)), "white")
for i, s in enumerate(S):
    n = f"{i + 1:02d}"; x, y = (i % C) * W, (i // C) * (H + T)
    im = Image.open(f"{K}/{n}.png").convert("RGB").resize((W, H)) if os.path.exists(f"{K}/{n}.png") else Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    who = {"none": "", "seen": "Amber seen", "sings": "Amber SINGS"}[s["amber"]]
    if who and n in faces and faces[n] is not None:
        who += f" (face {faces[n]:.2f})"
    drawn = "Qwen" if s["amber"] != "none" or os.path.exists(f"{K}/{i + 1:02d}-qwen.png") and RQ else "MOSS"      # who drew it (lean/keyframes.py; Qwen: from her photo)
    t = f"{i + 1}  {s['start']:.1f}-{s['end']:.1f}s  {drawn}  {who}"
    d.rectangle((0, 0, d.textlength(t, font=B) + 12, 28), fill=0); d.text((6, 3), t, font=B, fill=(255, 255, 0))
    sheet.paste(im, (x, y))
    d = ImageDraw.Draw(sheet)
    lines = " / ".join(segs[k - 1]["text"] for k in s.get("lines", []) if 0 < k <= len(segs)) or "(instrumental)"
    d.text((x + 8, y + H + 6), textwrap.shorten(f'"{lines}"', 52, placeholder=" ..."), font=B, fill=0)
    for j, l in enumerate(textwrap.wrap(s["beat"], 78)[:5]):
        d.text((x + 8, y + H + 34 + j * 20), l, font=F, fill=(60, 60, 60))
sheet.save(OUT, quality=88)
print(f"storyboard sheet: {OUT}")
