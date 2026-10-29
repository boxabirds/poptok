"""The singer's mouth, close up, every 2nd frame across cuckoo's "can't I come" (2.32-2.96 s into the D38/D39 clip;
Whisper heard "I" at 2.54-2.80 s), one row per clip: to see whether the sung "I" is an open "ah-ee" or an "ooo".
    python3 lean/poc_mouth_strip.py SWEEP_DIR OUT.jpg CLIP=LABEL ...      (CLIP: it_<CLIP>.mp4 in SWEEP_DIR)"""
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "pipeline")
from face_pass import face

F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
O, out, clips = sys.argv[1], sys.argv[2], [a.split("=") for a in sys.argv[3:]]
frames = range(58, 76, 2)
S = Image.new("RGB", (160 * len(frames), 160 * len(clips)))
for r, (c, lab) in enumerate(clips):
    cap = cv2.VideoCapture(f"{O}/it_{c}.mp4")
    for j, fn in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn); ok, im = cap.read(); f = face(im) if ok else None
        if f is None:
            t = Image.new("RGB", (160, 160))
        else:
            mx, my, s = (f[10] + f[12]) / 2, (f[11] + f[13]) / 2, int(f[2] * 0.55)
            cr = im[max(0, int(my - s / 2)):int(my + s / 2), max(0, int(mx - s / 2)):int(mx + s / 2)]
            t = Image.fromarray(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB)).resize((160, 160))
        w = "I" if 2.54 <= fn / 25 < 2.80 else ("come" if fn / 25 >= 2.80 else "can't")
        d = ImageDraw.Draw(t); d.rectangle((0, 0, 160, 18), fill=0)
        d.text((3, 1), f"{r + 1}{'abcdefghi'[j]} {lab} {fn / 25:.2f}s {w}", font=F, fill=(255, 255, 0))
        S.paste(t, (j * 160, r * 160))
S.save(out, quality=88)
