"""Self-check of a rendered video, frame by frame (not a sparse sheet): white flares, hard jumps, and a contact sheet of
consecutive frames around each event so they can be looked at.

    python3 pipeline/eval_video.py VIDEO.mp4 [--punches PUNCHES.json] [--sheet-dir DIR]
Flare: brightness more than 25 levels over its 4-second rolling median. Jump: frame-to-frame change over 6x the median
step, grouped into events; a jump within 2 frames of a bass-hit pulse (from finish.py's punches file) is expected.
"""
from __future__ import annotations

import argparse, json, os, subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--punches")
ap.add_argument("--sheet-dir")
a = ap.parse_args()
fps = eval(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate",
                           "-of", "csv=p=0", a.video], capture_output=True, text=True).stdout.strip())
W, H = 128, 72
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", a.video, "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                     capture_output=True).stdout
f = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
lum = f.mean((1, 2))
d = np.abs(np.diff(f, axis=0)).mean((1, 2))
win = int(2 * fps)
med = np.array([np.median(lum[max(0, i - win):i + win]) for i in range(len(lum))])


def events(idx, gap):
    ev = []
    for i in idx:
        if ev and i - ev[-1][1] <= gap:
            ev[-1][1] = i
        else:
            ev.append([int(i), int(i)])
    return ev


flares = events(np.where(lum - med > 25)[0], int(fps / 2))
jumps = events(np.where(d > np.median(d) * 6)[0], 3)
punch = []
if a.punches and os.path.exists(a.punches):
    p = json.load(open(a.punches))
    p = p if isinstance(p, list) else p.get("punches", [])
    punch = [x if isinstance(x, (int, float)) else x.get("t", 0) for x in p]
near = lambda e: any(abs(e[0] / fps - t) < 2 / fps + 0.05 for t in punch)
odd = [e for e in jumps if not near(e)]
print(f"{a.video}: {len(f)} frames, {len(f) / fps:.1f} s; median step {np.median(d):.2f}")
print(f"flares: {len(flares)}  " + " ".join(f"{e[0] / fps:.1f}s(+{lum[e[0]:e[1] + 1].max() - med[e[0]]:.0f})" for e in flares))
print(f"jumps: {len(jumps)} ({len(jumps) - len(odd)} on pulses); others at " +
      " ".join(f"{e[0] / fps:.1f}-{e[1] / fps:.1f}s" for e in odd[:30]))
if a.sheet_dir:
    os.makedirs(a.sheet_dir, exist_ok=True)
    for k, e in enumerate((flares + odd)[:12]):
        t = max(0, e[0] / fps - 8 / fps)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", a.video, "-vf",
                        "scale=240:135,tile=8x2", "-frames:v", "1", f"{a.sheet_dir}/event_{k:02d}_{e[0] / fps:.1f}s.jpg"])
    print("sheets:", a.sheet_dir)
