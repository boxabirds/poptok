"""D38 test 2's sheet and video. Sheet: one row per style level: the keyframe, then 6 consecutive clip frames from
the middle of the sung line (judged frame to frame), and the numbers (her face in the keyframe and through the clip,
lip-sync r and its baseline). Video: the six clips in a 3x2 grid, each labelled, with the song.
    python3 lean/poc_sweep_sheet.py SWEEP_DIR REVIEW_DIR"""
import json, os, re, subprocess, sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

O, R = sys.argv[1], sys.argv[2]
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
LV = json.load(open(f"{O}/levels.json"))
w, h = 384, 216


def lab(im, y, t):
    d = ImageDraw.Draw(im); d.rectangle((0, y - 2, d.textlength(t, font=F) + 10, y + 22), fill=0)
    d.text((5, y), t, font=F, fill=(255, 255, 0))


rows, nums = [], []
for r, x in enumerate(LV):
    l = f"{x['level']:03d}"; clip = f"{O}/it_{l}.mp4"
    tiles = [Image.open(f"{O}/k{l}.png" if os.path.exists(f"{O}/k{l}.png") else f"{O}/l{l}.png").convert("RGB").resize((w, h))]
    lab(tiles[0], 3, f"{r + 1}a {x['level']}% keyframe")
    lab(tiles[0], 190, f"her face {x['face']:.2f}" if x["face"] is not None else "no face found")
    sync = open(f"{O}/sync_{l}.txt").read() if os.path.exists(f"{O}/sync_{l}.txt") else ""
    m = re.search(r"best r = ([-0-9.]+) at ([-+0-9]+) ms.*baseline ([0-9.]+)", sync)
    ids = [v["similarity"] for v in json.load(open(f"{O}/id_{l}.json")) if v["similarity"] is not None] \
        if os.path.exists(f"{O}/id_{l}.json") else []
    c = cv2.VideoCapture(clip); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); s = max(0, n // 2 - 3)
    for j in range(6):
        c.set(cv2.CAP_PROP_POS_FRAMES, s + j); ok, fr = c.read()
        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((w, h)) if ok else Image.new("RGB", (w, h))
        lab(im, 3, f"{r + 1}{'bcdefg'[j]} f{s + j}"); tiles.append(im)
    if m:
        lab(tiles[1], 190, f"lip-sync r {float(m.group(1)):.2f} at {m.group(2)} ms (baseline {float(m.group(3)):.2f})")
    if ids:
        lab(tiles[4], 190, f"her face in clip: median {np.median(ids):.2f}, min {min(ids):.2f}")
    rows.append(tiles)
    nums.append({"level": x["level"], "keyframe_face": x["face"], "clip_face_median": float(np.median(ids)) if ids else None,
                 "sync_r": float(m.group(1)) if m else None, "sync_ms": int(m.group(2)) if m else None,
                 "baseline": float(m.group(3)) if m else None})
S = Image.new("RGB", (w * 7, h * len(rows)))
for r, tiles in enumerate(rows):
    for j, t in enumerate(tiles):
        S.paste(t, (j * w, r * h))
S.save(f"{R}/2-sweep.jpg", quality=88)
json.dump(nums, open(f"{R}/2-sweep.json", "w"), indent=1)
clips = [(x["level"], f"{O}/it_{x['level']:03d}.mp4") for x in LV if os.path.exists(f"{O}/it_{x['level']:03d}.mp4")]
if len(clips) == 6:
    ins, fc = [], ""
    for i, (lv, p) in enumerate(clips):
        ins += ["-i", p]
        fc += (f"[{i}:v]scale=640:360,setsar=1,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
               f"text='{i + 1} {lv}%':fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6:x=8:y=8[v{i}];")
    fc += "".join(f"[v{i}]" for i in range(6)) + "xstack=inputs=6:layout=0_0|640_0|1280_0|0_360|640_360|1280_360:shortest=1[o]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-i", f"{O}/mix.wav", "-filter_complex", fc, "-map", "[o]",
                    "-map", "6:a", "-c:v", "libx264", "-crf", "20", "-c:a", "aac", "-shortest", f"{R}/2-sweep.mp4"], check=True)
for x in nums:
    print(x)
