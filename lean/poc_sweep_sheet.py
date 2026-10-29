"""D38 test 2's (and D39's) sheet and videos. Sheet: one row per keyframe (D38's levels, then D39's in levels2.json):
the keyframe, then 6 consecutive clip frames from the middle of the sung line (judged frame to frame), and the numbers
(her face in the keyframe and through the clip, lip-sync r and its baseline). Videos: groups of six clips in a 3x2
grid, each labelled, with the song.
    python3 lean/poc_sweep_sheet.py SWEEP_DIR REVIEW_DIR [NAME [ID,ID,..=VIDEO ...]]
NAME: the sheet's name (default 2-sweep); each ID,..=VIDEO makes REVIEW_DIR/VIDEO.mp4 from those six rows (default: the
six D38 levels as 2-sweep.mp4)."""
import json, os, re, subprocess, sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

O, R = sys.argv[1], sys.argv[2]
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
LV = [{"id": f"{x['level']:03d}", "label": f"{x['level']}% style", "face": x["face"]} for x in json.load(open(f"{O}/levels.json"))]
if os.path.exists(f"{O}/levels2.json"):
    LV = sorted(LV + json.load(open(f"{O}/levels2.json")), key=lambda x: (not x["id"].isdigit(), x["id"]))
NAME = sys.argv[3] if len(sys.argv) > 3 else "2-sweep"
VIDS = [a.split("=") for a in sys.argv[4:]] or [["000,010,025,050,075,100", "2-sweep"]]
w, h = 384, 216


def lab(im, y, t):
    d = ImageDraw.Draw(im); d.rectangle((0, y - 2, d.textlength(t, font=F) + 10, y + 22), fill=0)
    d.text((5, y), t, font=F, fill=(255, 255, 0))


rows, nums = [], []
for r, x in enumerate(LV):
    l = x["id"]; clip = f"{O}/it_{l}.mp4"
    tiles = [Image.open(f"{O}/k{l}.png" if os.path.exists(f"{O}/k{l}.png") else f"{O}/l{l}.png").convert("RGB").resize((w, h))]
    lab(tiles[0], 3, f"{r + 1}a {x['label']}")
    lab(tiles[0], 190, f"her face {x['face']:.2f}" if x["face"] is not None else "no face found")
    sync = open(f"{O}/sync_{l}.txt").read() if os.path.exists(f"{O}/sync_{l}.txt") else ""   # eval_lipsync --window 1
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
    nums.append({"row": r + 1, "id": l, "label": x["label"], "keyframe_face": x["face"], "clip_face_median": float(np.median(ids)) if ids else None,
                 "sync_r": float(m.group(1)) if m else None, "sync_ms": int(m.group(2)) if m else None,
                 "baseline": float(m.group(3)) if m else None})
S = Image.new("RGB", (w * 7, h * len(rows)))
for r, tiles in enumerate(rows):
    for j, t in enumerate(tiles):
        S.paste(t, (j * w, r * h))
S.save(f"{R}/{NAME}.jpg", quality=88)
json.dump(nums, open(f"{R}/{NAME}.json", "w"), indent=1)
row = {x["id"]: (i + 1, x["label"]) for i, x in enumerate(LV)}
for ids, vid in VIDS:
    clips = [(i, f"{O}/it_{i}.mp4") for i in ids.split(",") if os.path.exists(f"{O}/it_{i}.mp4")]
    if len(clips) != 6:
        print(f"{vid}: {len(clips)} of 6 clips, skipped"); continue
    ins, fc = [], ""
    for k, (i, p) in enumerate(clips):
        ins += ["-i", p]
        t = f"{row[i][0]} {row[i][1]}".replace("%", " pc").replace(",", "")    # drawtext: % and , are special
        fc += (f"[{k}:v]scale=640:360,setsar=1,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
               f"text='{t}':fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.6:x=8:y=8[v{k}];")
    fc += "".join(f"[v{k}]" for k in range(6)) + "xstack=inputs=6:layout=0_0|640_0|1280_0|0_360|640_360|1280_360:shortest=1[o]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-i", f"{O}/mix.wav", "-filter_complex", fc, "-map", "[o]",
                    "-map", "6:a", "-c:v", "libx264", "-crf", "20", "-c:a", "aac", "-shortest", f"{R}/{vid}.mp4"], check=True)
for x in nums:
    print(x)
