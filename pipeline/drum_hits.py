"""Deep drum hits from the drum stem: onsets of low-band (40-250 Hz) spectral flux, with a strength each, and the
passages where they come in quick runs ("tat-tat-tat"). Used to sync bursts of light to the drums (finish.py --bursts)
and to tell the directors' workshop where the drum passages are.

    python3 pipeline/drum_hits.py ANALYSIS STEM [--pct 97]     -> ANALYSIS/drum_hits.json
{"hits": [{"t": s, "strength": 0-1}], "runs": [{"start": s, "end": s, "n": hits}]}
"""
from __future__ import annotations

import argparse, json, os, subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("analysis")
ap.add_argument("stem")
ap.add_argument("--pct", type=float, default=97, help="flux percentile above which a peak counts as a hit")
ap.add_argument("--run-gap", type=float, default=0.6, help="hits closer than this belong to one run")
a = ap.parse_args()
SR, HOP, N = 22050, 256, 1024
f = os.path.join(a.analysis, f"{a.stem}-drums.mp3")
x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                                 capture_output=True, check=True).stdout, np.float32)
win, freqs = np.hanning(N), np.fft.rfftfreq(N, 1 / SR)
band = (freqs > 40) & (freqs < 250)
frames = np.lib.stride_tricks.sliding_window_view(x, N)[::HOP] * win
mag = np.abs(np.fft.rfft(frames, axis=1))[:, band]
flux = np.r_[0, np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)]
t = np.arange(len(flux)) * HOP / SR
thr = np.percentile(flux, a.pct)
peaks = [i for i in range(1, len(flux) - 1) if flux[i] > thr and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]]
hits = []
for i in peaks:
    if not hits or t[i] - hits[-1]["t"] > 0.08:
        hits.append({"t": round(float(t[i]), 3), "strength": float(flux[i])})
top = max(h["strength"] for h in hits) if hits else 1
for h in hits:
    h["strength"] = round(h["strength"] / top, 3)
runs, cur = [], []
for h in hits:
    if cur and h["t"] - cur[-1]["t"] > a.run_gap:
        runs.append(cur); cur = []
    cur.append(h)
if cur:
    runs.append(cur)
runs = [{"start": r[0]["t"], "end": r[-1]["t"], "n": len(r)} for r in runs if len(r) >= 3]
out = os.path.join(a.analysis, "drum_hits.json")
json.dump({"stem": f, "hits": hits, "runs": runs}, open(out, "w"), indent=1)
print(f"{len(hits)} drum hits, {len(runs)} runs of 3+ -> {out}")
