"""Illustrate what an onset is and how librosa detects it, on a real song -> SVG.

Two stages, because librosa lives in the m2v-pre image and matplotlib on the host:
    python3 tools/onset_svg.py dump SONG.wav DATA.npz [--start 44] [--secs 2]     (m2v-pre image: librosa)
    python3 tools/onset_svg.py draw DATA.npz OUT.svg [--width 728]                 (host: matplotlib)
Panels: 1 the sound wave; 2 spectrogram (64 pitch bands, librosa's mel spectrogram in dB, as onset_strength uses);
3 change since the previous frame, per band (red louder, blue quieter); 4 only the increases kept; 5 added up across
bands = onset strength; 6 its peaks = onsets (librosa.onset.onset_detect). Same frame size as beat_grid.py (hop 256).
"""
from __future__ import annotations

import argparse, base64, html, io, sys, textwrap

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["dump", "draw"]); ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--start", type=float, default=44.0); ap.add_argument("--secs", type=float, default=2.0)
ap.add_argument("--width", type=int, default=728)
a = ap.parse_args()
SR, HOP = 22050, 256

if a.mode == "dump":
    import librosa
    y, _ = librosa.load(a.src, sr=SR, offset=a.start, duration=a.secs)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=64))   # as onset_strength
    d = np.diff(S, axis=1, prepend=S[:, :1])
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, n_mels=64)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP)
    np.savez(a.dst, y=y, S=S, d=d, env=env, on=on, start=a.start, secs=a.secs, name=a.src.split("/")[-1])
    print(f"{len(on)} onsets in {a.secs} s"); sys.exit()

import matplotlib; matplotlib.use("Agg")
import matplotlib.image as mpimg
D = np.load(a.src)
y, S, d, env, on = D["y"], D["S"], D["d"], D["env"], D["on"]
t0, secs, name = float(D["start"]), float(D["secs"]), str(D["name"])
W = a.width; L = 16; PW = W - 32
INK, MUTED, RULE, ACC, ACC2, ON = "#1d2433", "#5b6475", "#d9dde5", "#2f6fdb", "#e0732c", "#c0392b"
out = []; add = out.append; esc = html.escape
nf = S.shape[1]
X = lambda f: L + f / nf * PW


def para(y0, text, size=12.5):
    lines = textwrap.wrap(text, int(PW / (size * 0.53)))
    for k, line in enumerate(lines):
        add(f'<text x="16" y="{y0 + k * size * 1.3:.1f}" font-size="{size}" fill="{MUTED}">{esc(line)}</text>')
    return y0 + len(lines) * size * 1.3


def panel(y0, h, num, title, cap):
    add(f'<text x="16" y="{y0 + 16}" font-size="16" font-weight="700" fill="{INK}">{num}  {esc(title)}</text>')
    top = para(y0 + 36, cap) - 4
    add(f'<rect x="{L}" y="{top}" width="{PW}" height="{h}" fill="#fafbfc" stroke="{RULE}"/>')
    return top


def img(V, x0, y0, w, h, cmap, vmin, vmax):
    b = io.BytesIO(); mpimg.imsave(b, V[::-1], cmap=cmap, format="png", vmin=vmin, vmax=vmax)
    add(f'<image x="{x0}" y="{y0}" width="{w}" height="{h}" preserveAspectRatio="none" style="image-rendering:pixelated" '
        f'href="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}"/>')


def onsets(y0, h, col=ON):
    for f in on:
        add(f'<line x1="{X(f):.1f}" y1="{y0}" x2="{X(f):.1f}" y2="{y0 + h}" stroke="{col}" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>')


def time_axis(y0):
    for k in range(int(secs * 4) + 1):
        s = k * 0.25; xx = L + s / secs * PW
        anc = "start" if k == 0 else "end" if s >= secs else "middle"
        add(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + 4}" stroke="{MUTED}"/>')
        add(f'<text x="{xx:.1f}" y="{y0 + 16}" font-size="10.5" text-anchor="{anc}" fill="{MUTED}">{t0 + s:.2f} s</text>')


add(f'<text x="16" y="30" font-size="20" font-weight="700" fill="{INK}">What an onset is, and how it is found</text>')
yy = para(52, f"An onset is where a new sound starts. {name}, {t0:.0f}–{t0 + secs:.0f} s.", 13) + 6

h = 80; top = panel(yy, h, "1", "The sound wave", "Hits: sudden jumps in loudness")
c = max(1, len(y) // PW); m = np.abs(y).max(); mid = top + h / 2
for i in range(PW):
    s = y[i * c:(i + 1) * c]
    if len(s): add(f'<line x1="{L + i}" y1="{mid - s.max() / m * (h / 2 - 3):.1f}" x2="{L + i}" y2="{mid - s.min() / m * (h / 2 - 3):.1f}" stroke="{ACC}"/>')
onsets(top, h)
yy = top + h + 30

h = 110; top = panel(yy, h, "2", "Spectrogram", "Loudness per pitch band, every 12 ms")
img(S, L, top, PW, h, "Blues", S.max() - 70, S.max()); yy = top + h + 30

lim = np.percentile(np.abs(d), 99)
h = 110; top = panel(yy, h, "3", "Change since last frame", "Red louder, blue quieter")
img(d, L, top, PW, h, "RdBu_r", -lim, lim); yy = top + h + 30

h = 110; top = panel(yy, h, "4", "Keep only increases", "Fading isn't a new sound")
img(np.maximum(d, 0), L, top, PW, h, "Reds", 0, lim); yy = top + h + 30

h = 100; top = panel(yy, h, "5", "Add up across bands", "One number per frame: onset strength")
em = env.max()
pts = " ".join(f"{X(f):.1f},{top + h - 5 - v / em * (h - 12):.1f}" for f, v in enumerate(env))
add(f'<polyline points="{pts}" fill="none" stroke="{ACC2}" stroke-width="1.5"/>')
yy = top + h + 30

h = 100; top = panel(yy, h, "6", "Peaks are onsets", f"{len(on)} onsets in {secs:.0f} seconds")
pts6 = " ".join(f"{X(f):.1f},{top + h - 5 - v / em * (h - 12):.1f}" for f, v in enumerate(env))
add(f'<polyline points="{pts6}" fill="none" stroke="{ACC2}" stroke-width="1" opacity="0.5"/>')
for f in on:
    add(f'<line x1="{X(f):.1f}" y1="{top + 4}" x2="{X(f):.1f}" y2="{top + h}" stroke="{ON}" stroke-width="2"/>')
    add(f'<circle cx="{X(f):.1f}" cy="{top + h - 5 - env[f] / em * (h - 12):.1f}" r="3.2" fill="{ON}"/>')
time_axis(top + h)
yy = top + h + 34
yy = para(yy, "librosa.onset.onset_strength (mel spectrogram in dB, lag 1, mean over 64 bands) and onset_detect. "
              "Generated by tools/onset_svg.py.", 11) + 4

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{yy:.0f}" viewBox="0 0 {W} {yy:.0f}" font-family="Helvetica, Arial, sans-serif">'
       f'<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(out) + "</svg>")
open(a.dst, "w").write(svg)
print(f"wrote {a.dst}: {len(on)} onsets")
