"""The separate layers of a song, per video frame (2026-10-27, Ice Walk: "there are multiple layers of sounds going ... How
many distinct layers can you identify?"), so each can drive its own visual. Demucs (htdemucs_6s) splits the song into
stems; each layer is then read from the stem it lives in:

    subbass  drums-free bass below 90 Hz: a slow envelope (0..1 per frame)
    pad      the sustained chord (the 'guitar' stem, 80-1200 Hz): its envelope, and the frames where the chord changes
    chime    short tonal notes in 450-800 Hz ('other' stem): one event per note, with its pitch
    drone    the steadiest narrow tone in 700-1400 Hz ('other' stem): a slow envelope (0..1) as it swells and fades
    clicks   broadband ticks above 3 kHz ('drums' stem): one event per tick

    python3 pipeline/world/music_layers.py SONG OUT.json --start S --dur D [--fps 24] [--stems DIR]
Runs where Demucs is (the m2v-pre image). The stems of [S - 10, S + D + 10] are cached in DIR (default: next to OUT).
"""
from __future__ import annotations

import argparse, json, os, subprocess

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, stft

ap = argparse.ArgumentParser()
ap.add_argument("song")
ap.add_argument("out")
ap.add_argument("--start", type=float, required=True)
ap.add_argument("--dur", type=float, required=True)
ap.add_argument("--fps", type=float, default=24)
ap.add_argument("--stems", default="")
a = ap.parse_args()
PAD = 10.0                              # context either side, so the separation is not starved at the window's edges
pre = min(PAD, a.start)
sd = a.stems or os.path.join(os.path.dirname(os.path.abspath(a.out)), "stems")
os.makedirs(sd, exist_ok=True)
win = os.path.join(sd, "win.wav")
if not os.path.exists(os.path.join(sd, "htdemucs_6s", "win", "other.wav")):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a.start - pre), "-t", str(a.dur + pre + PAD), "-i", a.song,
                    "-ac", "2", "-ar", "44100", win], check=True)
    subprocess.run(["python", "-m", "demucs", "-n", "htdemucs_6s", "-o", sd, win], check=True)
HOP, NFFT = 256, 4096
nf = int(round(a.dur * a.fps))


def stem(name):                         # magnitude spectrogram of the window, its frame times in video frames
    x, sr = sf.read(os.path.join(sd, "htdemucs_6s", "win", f"{name}.wav"))
    x = x.mean(axis=1)[int(pre * sr):int((pre + a.dur) * sr)]
    f, t, Z = stft(x, sr, nperseg=NFFT, noverlap=NFFT - HOP)
    return f, t * a.fps, np.abs(Z), sr


def energy(f, A, lo, hi):
    return (A[(f >= lo) & (f < hi)] ** 2).sum(axis=0)


def norm(v, lo=5, hi=95):
    a_, b_ = np.percentile(v, lo), np.percentile(v, hi)
    return np.clip((v - a_) / (b_ - a_ + 1e-12), 0, 1)


def per_frame(t, v):
    return np.interp(np.arange(nf), t, v).round(4).tolist()


def onsets(e, t, sr, pct, gap_s):      # rises of the log energy above its pct-th percentile, at least gap_s apart
    l = np.log(e + 1e-12)
    d = uniform_filter1d(np.maximum(np.diff(l, prepend=l[0]), 0), 3)
    p, _ = find_peaks(d, height=np.percentile(d, pct), distance=max(1, int(gap_s * sr / HOP)))
    return p, d


out = {"song": a.song, "fps": a.fps, "start": a.start, "dur": a.dur}
f, t, A, sr = stem("bass")
out["subbass"] = per_frame(t, norm(uniform_filter1d(np.sqrt(energy(f, A, 30, 90)), int(0.3 * sr / HOP))))

f, t, A, sr = stem("guitar")
S = uniform_filter1d(np.log1p(A[(f >= 80) & (f < 1200)] * 100), int(0.33 * sr / HOP), axis=1)
nov = np.linalg.norm(np.diff(S, axis=1, prepend=S[:, :1]), axis=0)
p, _ = find_peaks(nov, height=np.percentile(nov, 93), distance=int(1.0 * sr / HOP))      # chords last a second or more
out["pad"] = {"energy": per_frame(t, norm(np.sqrt(energy(f, A, 80, 1200)))),
              "changes": [int(round(t[i])) for i in p if round(t[i]) < nf]}

f, t, A, sr = stem("other")
sel = (f >= 700) & (f < 1400)
avg = (A ** 2).mean(axis=1)
dhz = float(f[sel][np.argmax(avg[sel])])
dr = np.sqrt(energy(f, A, dhz - 15, dhz + 15))
out["drone"] = {"hz": round(dhz, 1), "energy": per_frame(t, norm(uniform_filter1d(dr, int(0.5 * sr / HOP)), 5, 98))}
# chime notes: narrow-band 450-800 Hz runs above that band's 80th percentile lasting >= 1.5 frames, one event each
cb = (f >= 450) & (f < 800)
m = energy(f, A, 450, 800)
on = m > np.percentile(m, 80)
notes, i = [], 0
while i < len(on):
    if on[i]:
        j = i
        while j < len(on) and on[j]:
            j += 1
        if t[j - 1] - t[i] >= 1.5 and round(t[i]) < nf:
            notes.append({"frame": int(round(t[i])), "hz": round(float(f[cb][np.argmax(A[cb, i:j].sum(axis=1))]), 1),
                          "strength": round(float(np.sqrt(m[i:j].max() / m.max())), 3)})
        i = j
    else:
        i += 1
out["chime"] = notes

f, t, A, sr = stem("drums")
p, d = onsets(energy(f, A, 3000, 16000), t, sr, 90, 0.08)
out["clicks"] = {"hits": [int(round(t[i])) for i in p if round(t[i]) < nf],
                 "strength": [round(float(v), 3) for i, v in zip(p, norm(d[p], 0, 100)) if round(t[i]) < nf]}
json.dump(out, open(a.out, "w"))
print(f"{a.out}: drone {dhz:.0f} Hz, {len(out['pad']['changes'])} chord changes, {len(notes)} chime notes, "
      f"{len(out['clicks']['hits'])} clicks")
