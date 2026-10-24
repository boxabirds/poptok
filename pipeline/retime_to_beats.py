"""Retime a first/last-frame transition stream (ltx_chain.py output with --clips) so each transition's midpoint lands ON the
downbeat where its target keyframe's shot starts, instead of wherever the video model put it.

    python3 pipeline/retime_to_beats.py OUT [--src ltx_frames] [--dst ltx_frames_beat]

Per shot i, k_i = first frame where the clip is more than halfway (in mean abs grey difference) from key i to key i+1.
Anchors: output frame start(i+1) <- source frame start(i) + k_i; output 0 <- 0; output end <- end. Piecewise-linear
time map between anchors, nearest source frame, written as symlinks to the source PNGs.
"""
from __future__ import annotations

import argparse, json, os

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--src", default="ltx_frames")
ap.add_argument("--dst", default="ltx_frames_beat")
ap.add_argument("--fps", type=float, default=24)
ap.add_argument("--clips", action="store_true", help="SRC is a ltx_chain.py stream: use SRC/clips.json (clip j changes "
                "from its first to its last frame and should be halfway at its downbeat)")
a = ap.parse_args()
sh = json.load(open(os.path.join(a.out, "shots.json")))["shots"]
src, dst = os.path.join(a.out, a.src), os.path.join(a.out, a.dst)
os.makedirs(dst, exist_ok=True)
ld = lambda f: np.asarray(Image.open(os.path.join(src, f"{f:05d}.png")).convert("L").resize((128, 72)), np.float32)

start = [int(round(s["start"] * a.fps)) for s in sh]
n = int(round(sh[-1]["end"] * a.fps))
if a.clips:                                                    # (clip start, clip end, target output frame)
    spans = [(c["a"], c["e"], c["downbeat"]) for c in json.load(open(os.path.join(src, "clips.json")))[1:]
             if c.get("change", True)]
else:
    spans = [(start[i], start[i + 1], start[i + 1]) for i in range(len(sh) - 1)]
anchors = [(0, 0)]
for s0, s1, target in spans:
    fr = [ld(f) for f in range(s0, s1)]
    A, B = fr[0], fr[-1]
    prog = [np.abs(x - A).mean() / (np.abs(x - A).mean() + np.abs(x - B).mean() + 1e-6) for x in fr]
    k = next((j for j, v in enumerate(prog) if v > 0.5), len(fr) // 2)
    anchors.append((target, s0 + k))
anchors.append((n - 1, n - 1))
ox, sx = np.array(anchors, float).T
m = np.clip(np.round(np.interp(np.arange(n), ox, sx)).astype(int), 0, n - 1)
for f, s in enumerate(m):
    p = os.path.join(dst, f"{f:05d}.png")
    if os.path.lexists(p):
        os.remove(p)
    os.symlink(os.path.relpath(os.path.join(src, f"{s:05d}.png"), dst), p)
sp = np.diff(sx) / np.maximum(np.diff(ox), 1)
print(f"{n} frames -> {dst}; playback speed per bar: min {sp.min():.2f}x median {np.median(sp):.2f}x max {sp.max():.2f}x")
