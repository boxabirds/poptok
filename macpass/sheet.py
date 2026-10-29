"""The macpass keyframes as one sheet, in order, each labelled with its number, time, lyric, camera intent and beat.
    python3 macpass/sheet.py PIECE OUT.jpg"""
import json, sys, textwrap

from PIL import Image, ImageDraw, ImageFont

P, OUT = sys.argv[1].rstrip("/"), sys.argv[2]; O = f"{P}/macpass"
B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
S = json.load(open(f"{O}/shots.json"))["shots"]; sb = {s["index"]: s for s in json.load(open(f"{O}/storyboard.json"))["shots"]}
W, H, T, C = 640, 357, 120, 3
sheet = Image.new("RGB", (W * C, (H + T) * ((len(S) + C - 1) // C)), "white")
for k, s in enumerate(S):
    x, y = (k % C) * W, (k // C) * (H + T)
    try:
        im = Image.open(f"{O}/keyframes/keyframe_{s['index']:03d}.png").convert("RGB").resize((W, H))
    except FileNotFoundError:
        im = Image.new("RGB", (W, H), "grey")
    sheet.paste(im, (x, y)); d = ImageDraw.Draw(sheet); b = sb.get(s["index"], {})
    d.text((x + 6, y + H + 4), f"{s['index']}  {s['start_s']:.1f}-{s['end_s']:.1f}s  {s['target_text'][:30]}  [{b.get('camera_intent')}]",
           font=B, fill="black")
    for j, line in enumerate(textwrap.wrap(b.get("beat", ""), 78)[:5]):
        d.text((x + 6, y + H + 30 + j * 17), line, font=F, fill="black")
sheet.save(OUT, quality=88); print(OUT)
