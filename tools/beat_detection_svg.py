"""Illustrate how the beat grid is found (librosa, as pipeline/beat_grid.py), on a real song -> SVG.

    python tools/beat_detection_svg.py SONG.wav OUT.svg [--start 40] [--secs 8]      (m2v-pre image: librosa)
Panels, numbered: 1 sound wave, 2 spectrogram, 3 onset strength, 4 tempo (self-similarity of 3), 5 beats chosen by
dynamic programming, 6 downbeats (our rule, not librosa's). Same settings as beat_grid.py: 22 050 Hz, hop 256,
tightness 100; beats tracked on the whole song, then a window shown.
"""
from __future__ import annotations

import argparse, html, textwrap

import librosa, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("song"); ap.add_argument("out")
ap.add_argument("--start", type=float, default=40.0); ap.add_argument("--secs", type=float, default=8.0)
ap.add_argument("--width", type=int, default=728)
a = ap.parse_args()
SR, HOP = 22050, 256
aud, _ = librosa.load(a.song, sr=SR)
env = librosa.onset.onset_strength(y=aud, sr=SR, hop_length=HOP)
tempo, bf = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, units="frames", tightness=100)
tempo = float(np.atleast_1d(tempo)[0])
bt = librosa.frames_to_time(bf, sr=SR, hop_length=HOP)
phase = int(np.argmax([env[bf][p::4].sum() for p in range(4)]))
down = set(bt[phase::4].round(4))
period = 60 / tempo
ac = librosa.autocorrelate(env, max_size=int(2.2 * SR / HOP)); ac = ac / ac[0]
lags = np.arange(len(ac)) * HOP / SR

t0, t1 = a.start, a.start + a.secs
W, L, R = a.width, 16, 16                     # width; explanations sit above each plot
PW = W - L - R
X = lambda t: L + (t - t0) / (t1 - t0) * PW
esc = html.escape
out = []
add = out.append
INK, MUTED, RULE, ACC, ACC2, BEAT, DOWN = "#1d2433", "#5b6475", "#d9dde5", "#2f6fdb", "#e0732c", "#2f9e6b", "#c0392b"


def wrap(text, size):
    return textwrap.wrap(text, int((W - 32) / (size * 0.53)))


def para(y0, text, size=12.5, col=None):
    """Wrapped text from y0 (baseline of the first line); returns the y below it."""
    for k, line in enumerate(wrap(text, size)):
        add(f'<text x="16" y="{y0 + k * size * 1.3:.1f}" font-size="{size}" fill="{col or MUTED}">{esc(line)}</text>')
    return y0 + len(wrap(text, size)) * size * 1.3


def panel(y0, h, num, title, lines):
    """Title and explanation above the plot; returns the plot's top."""
    add(f'<text x="16" y="{y0 + 16}" font-size="16" font-weight="700" fill="{INK}">{num}  {esc(title)}</text>')
    top = para(y0 + 36, " ".join(lines)) - 4
    add(f'<rect x="{L}" y="{top}" width="{PW}" height="{h}" fill="#fafbfc" stroke="{RULE}"/>')
    return top


def time_axis(y0):
    for s in range(int(np.ceil(t0)), int(t1) + 1):
        add(f'<line x1="{X(s):.1f}" y1="{y0}" x2="{X(s):.1f}" y2="{y0 + 5}" stroke="{MUTED}"/>')
        anc = "start" if s <= t0 else "end" if s >= t1 else "middle"
        add(f'<text x="{X(s):.1f}" y="{y0 + 18}" font-size="11" text-anchor="{anc}" fill="{MUTED}">{s} s</text>')


H = {1: 110, 2: 150, 3: 130, 4: 140, 5: 160, 6: 120}
GAP = 46
add(f'<text x="16" y="30" font-size="20" font-weight="700" fill="{INK}">How the beat grid is found (librosa)</text>')
y = para(52, f'{a.song.split("/")[-1]}, {t0:.0f}–{t1:.0f} s · {tempo:.0f} BPM', 13) + 6

# 1 waveform
h = H[1]; y = panel(y, h, "1", "The sound wave", ["Raw audio samples over time"])
i0, i1 = int(t0 * SR), int(t1 * SR); seg = aud[i0:i1]
cols = PW; chunk = len(seg) // cols; mx = np.abs(seg).max() + 1e-9
mid = y + h / 2
pts = []
for c in range(cols):
    s = seg[c * chunk:(c + 1) * chunk]
    add(f'<line x1="{L + c}" y1="{mid - s.max() / mx * (h / 2 - 6):.1f}" x2="{L + c}" y2="{mid - s.min() / mx * (h / 2 - 6):.1f}" stroke="{ACC}" stroke-width="1"/>')
y += h + GAP

# 2 spectrogram
h = H[2]; y = panel(y, h, "2", "Spectrogram", ["Energy in each pitch band"])
M = librosa.power_to_db(librosa.feature.melspectrogram(y=aud[i0:i1], sr=SR, hop_length=HOP, n_mels=64), ref=np.max)
nb, nf = M.shape; cw = PW / nf; bh = h / nb
for f in range(0, nf, 3):
    for b in range(nb):
        v = (M[b, f] + 80) / 80
        if v < 0.15: continue
        r, g, bl = int(255 - 200 * v), int(255 - 150 * v), int(255 - 40 * v)
        add(f'<rect x="{L + f * cw:.1f}" y="{y + h - (b + 1) * bh:.1f}" width="{3 * cw + .3:.1f}" height="{bh + .3:.1f}" fill="rgb({r},{g},{bl})"/>')
y += h + GAP

# 3 onset strength
h = H[3]; y = panel(y, h, "3", "Onset strength", ["Where new sounds start"])
f0, f1 = int(t0 * SR / HOP), int(t1 * SR / HOP); e = env[f0:f1]; em = e.max() + 1e-9
pts = " ".join(f"{X((f0 + k) * HOP / SR):.1f},{y + h - 6 - v / em * (h - 14):.1f}" for k, v in enumerate(e))
add(f'<polyline points="{pts}" fill="none" stroke="{ACC2}" stroke-width="1.4"/>')
time_axis(y + h)
y += h + GAP

# 4 tempo
h = H[4]; y = panel(y, h, "4", "Tempo", [f"Best repeat: {period:.2f} s = {tempo:.0f} BPM"])
lx = lambda s: L + s / 2.2 * PW
pts = " ".join(f"{lx(s):.1f},{y + h - 8 - max(v, 0) * (h - 18):.1f}" for s, v in zip(lags, ac) if s <= 2.2)
add(f'<polyline points="{pts}" fill="none" stroke="{ACC}" stroke-width="1.4"/>')
for m, lab in ((1, "1 beat"), (2, "2 beats"), (3, "3 beats")):
    s = period * m
    if s <= 2.2:
        add(f'<line x1="{lx(s):.1f}" y1="{y + 4}" x2="{lx(s):.1f}" y2="{y + h}" stroke="{BEAT}" stroke-dasharray="4 3"/>')
        add(f'<text x="{lx(s) + 4:.1f}" y="{y + 16}" font-size="10.5" fill="{BEAT}">{lab} ({s:.2f} s)</text>')
for s in np.arange(0, 2.21, 0.5):
    add(f'<text x="{lx(s):.1f}" y="{y + h + 16}" font-size="11" text-anchor="{"start" if s == 0 else "middle"}" fill="{MUTED}">{s:.1f} s{" delay" if s == 0 else ""}</text>')
y += h + GAP

# 5 beats (dynamic programming)
h = H[5]; y = panel(y, h, "5", "Beats", ["Strong onsets, evenly spaced"])
pts = " ".join(f"{X((f0 + k) * HOP / SR):.1f},{y + h - 6 - v / em * (h - 30):.1f}" for k, v in enumerate(e))
add(f'<polyline points="{pts}" fill="none" stroke="{ACC2}" stroke-width="1" opacity="0.55"/>')
wb = [t for t in bt if t0 <= t <= t1]
for i, t in enumerate(wb):
    add(f'<line x1="{X(t):.1f}" y1="{y + 14}" x2="{X(t):.1f}" y2="{y + h}" stroke="{BEAT}" stroke-width="2"/>')
    if i:
        xm = (X(t) + X(wb[i - 1])) / 2
        add(f'<path d="M{X(wb[i - 1]) + 2:.1f},{y + 10} L{X(t) - 2:.1f},{y + 10}" stroke="{MUTED}" stroke-width="0.8"/>')
        add(f'<text x="{xm:.1f}" y="{y + 7}" font-size="9.5" text-anchor="middle" fill="{MUTED}">{t - wb[i - 1]:.2f}</text>')
time_axis(y + h)
y += h + GAP

# 6 downbeats
h = H[6]; y = panel(y, h, "6", "Downbeats", ["Count in fours (our rule)"])
n = 0
for t in bt:
    if not (t0 <= t <= t1): continue
    k = (list(bt).index(t) - phase) % 4 + 1
    isd = k == 1
    add(f'<line x1="{X(t):.1f}" y1="{y + 22}" x2="{X(t):.1f}" y2="{y + h}" stroke="{DOWN if isd else BEAT}" stroke-width="{3 if isd else 1.5}"/>')
    add(f'<text x="{X(t):.1f}" y="{y + 17}" font-size="{14 if isd else 11}" font-weight="{700 if isd else 400}" text-anchor="middle" fill="{DOWN if isd else BEAT}">{k}</text>')
time_axis(y + h)
y = para(y + h + 44, "librosa onset_strength + beat_track · step 6: beat_grid.py", 11) + 4
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{y}" viewBox="0 0 {W} {y}" '
       f'font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>' + "".join(out) + "</svg>")
open(a.out, "w").write(svg)
print(f"wrote {a.out}: tempo {tempo:.1f}, {len(wb)} beats in window, downbeat phase {phase}")
