"""Per-frame music curves to drive a camera through a 3D world (2026-10-26, abstract-landscape pieces): loudness, onset
strength and the transients themselves, sampled at the video frame rate, so motion can follow the music directly.

    python3 pipeline/world/music_curves.py SONG OUT.json [--fps 24] [--start S --dur D | --pick D]
--pick D chooses the D-second window with the most transient activity (for a proof of concept).
OUT.json: {"fps", "start", "dur", "energy": [0..1 per frame], "flux": [0..1 per frame], "hits": [frame, ...],
           "hit_strength": [0..1, ...], "tempo": bpm,
           "bands": {"low"|"lowmid"|"mid"|"high": {"hz": [f0, f1], "energy": [...], "flux": [...], "hits": [frame, ...]}}}
"""
from __future__ import annotations

import argparse, json, subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("song")
ap.add_argument("out")
ap.add_argument("--fps", type=float, default=24)
ap.add_argument("--start", type=float, default=0)
ap.add_argument("--dur", type=float, default=0)
ap.add_argument("--pick", type=float, default=0)
a = ap.parse_args()
SR, HOP, N = 22050, 256, 2048
x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", a.song, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                                 capture_output=True, check=True).stdout, np.float32)
frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (0, N)), N)[::HOP]
spec = np.abs(np.fft.rfft(frames * np.hanning(N), axis=1))
logs = np.log1p(spec * 10)
flux = np.maximum(np.diff(logs, axis=0, prepend=logs[:1]), 0).sum(axis=1)       # spectral flux: onset strength
rms = np.sqrt((frames ** 2).mean(axis=1))
t = np.arange(len(flux)) * HOP / SR


def norm(v, lo=5, hi=99):
    a_, b_ = np.percentile(v, lo), np.percentile(v, hi)
    return np.clip((v - a_) / (b_ - a_ + 1e-9), 0, 1)


fl, en = norm(flux), norm(rms)
# transients: local maxima of flux above its 90th percentile, at least 0.12 s apart
thr = np.percentile(flux, 90)
peaks = [i for i in range(1, len(flux) - 1) if flux[i] > thr and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]]
hits = []
for i in peaks:
    if not hits or t[i] - t[hits[-1]] > 0.12:
        hits.append(i)
    elif flux[i] > flux[hits[-1]]:
        hits[-1] = i
# spectral bands, each with its own loudness and its own transients (user, 2026-10-26: "do a spectral analysis -- highs
# and lows"): lows (kick, bass) can push the camera, mids sway it, highs make the light sparkle
freqs = np.fft.rfftfreq(N, 1 / SR)
BANDS = {"low": (20, 150), "lowmid": (150, 600), "mid": (600, 3000), "high": (3000, 11000)}
band = {}
for name, (f0, f1) in BANDS.items():
    sel = (freqs >= f0) & (freqs < f1)
    e = np.sqrt((spec[:, sel] ** 2).mean(axis=1))
    bf = np.maximum(np.diff(logs[:, sel], axis=0, prepend=logs[:1, sel]), 0).sum(axis=1)
    th = np.percentile(bf, 92)
    pk = []
    for i in range(1, len(bf) - 1):
        if bf[i] > th and bf[i] >= bf[i - 1] and bf[i] >= bf[i + 1]:
            if not pk or t[i] - t[pk[-1]] > 0.1:
                pk.append(i)
            elif bf[i] > bf[pk[-1]]:
                pk[-1] = i
    band[name] = (norm(e), norm(bf), pk)
# tempo from the autocorrelation of flux (60-180 bpm)
ac = np.correlate(fl - fl.mean(), fl - fl.mean(), "full")[len(fl) - 1:]
lags = np.arange(len(ac)) * HOP / SR
ok = (lags > 60 / 180) & (lags < 60 / 60)
tempo = 60 / lags[ok][np.argmax(ac[ok])]
start, dur = a.start, a.dur or t[-1]
if a.pick:                       # the window with the most transient activity
    ht = np.array([t[i] for i in hits]); hs = np.array([fl[i] for i in hits])
    best = max(np.arange(0, t[-1] - a.pick, 0.5), key=lambda s: hs[(ht >= s) & (ht < s + a.pick)].sum())
    start, dur = float(best), a.pick
nf = int(round(dur * a.fps))
ft = start + np.arange(nf) / a.fps
out = {"song": a.song, "fps": a.fps, "start": start, "dur": dur, "tempo": round(float(tempo), 1),
       "energy": np.interp(ft, t, en).round(4).tolist(), "flux": np.interp(ft, t, fl).round(4).tolist(),
       "hits": [int(round((t[i] - start) * a.fps)) for i in hits if start <= t[i] < start + dur],
       "hit_strength": [round(float(fl[i]), 3) for i in hits if start <= t[i] < start + dur],
       "bands": {name: {"hz": list(BANDS[name]), "energy": np.interp(ft, t, e).round(4).tolist(),
                        "flux": np.interp(ft, t, bf).round(4).tolist(),
                        "hits": [int(round((t[i] - start) * a.fps)) for i in pk if start <= t[i] < start + dur]}
                 for name, (e, bf, pk) in band.items()}}
json.dump(out, open(a.out, "w"))
print(f"{a.out}: {start:.1f}-{start + dur:.1f} s, {len(out['hits'])} transients, tempo ~{out['tempo']} bpm")
