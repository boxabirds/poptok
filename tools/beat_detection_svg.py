"""Illustrate how the beat grid is found (librosa, as pipeline/beat_grid.py), on a real song -> SVG.

    python tools/beat_detection_svg.py SONG.wav OUT.svg [--start 40] [--secs 8]      (m2v-pre image: librosa)
Panels, numbered: 1 sound wave, 2 spectrogram, 3 onset strength, 4 tempo (self-similarity of 3), 5 beats chosen by
dynamic programming, 6 downbeats (our rule, not librosa's). Same settings as beat_grid.py: 22 050 Hz, hop 256,
tightness 100; beats tracked on the whole song, then a window shown.
"""
from __future__ import annotations

import argparse, html

import librosa, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("song"); ap.add_argument("out")
ap.add_argument("--start", type=float, default=40.0); ap.add_argument("--secs", type=float, default=8.0)
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
W, L, R = 1200, 250, 30                      # width, left text column, right margin
PW = W - L - R
X = lambda t: L + (t - t0) / (t1 - t0) * PW
esc = html.escape
out = []
add = out.append
INK, MUTED, RULE, ACC, ACC2, BEAT, DOWN = "#1d2433", "#5b6475", "#d9dde5", "#2f6fdb", "#e0732c", "#2f9e6b", "#c0392b"


def panel(y0, h, num, title, lines):
    add(f'<rect x="{L}" y="{y0}" width="{PW}" height="{h}" fill="#fafbfc" stroke="{RULE}"/>')
    add(f'<text x="16" y="{y0 + 20}" font-size="17" font-weight="700" fill="{INK}">{num}  {esc(title)}</text>')
    for k, s in enumerate(lines):
        add(f'<text x="16" y="{y0 + 42 + k * 16}" font-size="12.5" fill="{MUTED}">{esc(s)}</text>')


def time_axis(y0):
    for s in range(int(np.ceil(t0)), int(t1) + 1):
        add(f'<line x1="{X(s):.1f}" y1="{y0}" x2="{X(s):.1f}" y2="{y0 + 5}" stroke="{MUTED}"/>')
        add(f'<text x="{X(s):.1f}" y="{y0 + 18}" font-size="11" text-anchor="middle" fill="{MUTED}">{s} s</text>')


H = {1: 110, 2: 150, 3: 130, 4: 140, 5: 160, 6: 120}
GAP = 34
y = 70
add(f'<text x="16" y="34" font-size="22" font-weight="700" fill="{INK}">How the beat grid is found (librosa)</text>')
add(f'<text x="16" y="56" font-size="13" fill="{MUTED}">{esc(a.song.split("/")[-1])}, {t0:.0f}–{t1:.0f} s of the song. '
    f'Same settings as pipeline/beat_grid.py. Result: {tempo:.1f} beats per minute, a beat every {period:.2f} s.</text>')

# 1 waveform
h = H[1]; panel(y, h, "1", "The sound wave", ["The song as numbers: 22,050 samples", "a second. Loud hits show as tall", "spikes, but the beat isn't marked."])
i0, i1 = int(t0 * SR), int(t1 * SR); seg = aud[i0:i1]
cols = PW; chunk = len(seg) // cols; mx = np.abs(seg).max() + 1e-9
mid = y + h / 2
pts = []
for c in range(cols):
    s = seg[c * chunk:(c + 1) * chunk]
    add(f'<line x1="{L + c}" y1="{mid - s.max() / mx * (h / 2 - 6):.1f}" x2="{L + c}" y2="{mid - s.min() / mx * (h / 2 - 6):.1f}" stroke="{ACC}" stroke-width="1"/>')
y += h + GAP

# 2 spectrogram
h = H[2]; panel(y, h, "2", "Spectrogram", ["Energy in 64 pitch bands (low at the", "bottom), every 12 ms. A drum hit is a", "sudden bright vertical stripe."])
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
h = H[3]; panel(y, h, "3", "Onset strength", ["For each moment: how much louder each", "band just got, added up. Peaks = new", "sounds starting (drum hits, plucks)."])
f0, f1 = int(t0 * SR / HOP), int(t1 * SR / HOP); e = env[f0:f1]; em = e.max() + 1e-9
pts = " ".join(f"{X((f0 + k) * HOP / SR):.1f},{y + h - 6 - v / em * (h - 14):.1f}" for k, v in enumerate(e))
add(f'<polyline points="{pts}" fill="none" stroke="{ACC2}" stroke-width="1.4"/>')
time_axis(y + h)
y += h + GAP

# 4 tempo
h = H[4]; panel(y, h, "4", "Tempo", ["Compare curve 3 with itself shifted", "by each delay. It matches best when", f"shifted by one beat: {period:.2f} s = {tempo:.0f} per minute."])
lx = lambda s: L + s / 2.2 * PW
pts = " ".join(f"{lx(s):.1f},{y + h - 8 - max(v, 0) * (h - 18):.1f}" for s, v in zip(lags, ac) if s <= 2.2)
add(f'<polyline points="{pts}" fill="none" stroke="{ACC}" stroke-width="1.4"/>')
for m, lab in ((1, "1 beat"), (2, "2 beats"), (3, "3 beats")):
    s = period * m
    if s <= 2.2:
        add(f'<line x1="{lx(s):.1f}" y1="{y + 4}" x2="{lx(s):.1f}" y2="{y + h}" stroke="{BEAT}" stroke-dasharray="4 3"/>')
        add(f'<text x="{lx(s) + 4:.1f}" y="{y + 16}" font-size="11.5" fill="{BEAT}">{lab} ({s:.2f} s)</text>')
for s in np.arange(0, 2.21, 0.5):
    add(f'<text x="{lx(s):.1f}" y="{y + h + 16}" font-size="11" text-anchor="middle" fill="{MUTED}">{s:.1f} s delay</text>')
y += h + GAP

# 5 beats (dynamic programming)
h = H[5]; panel(y, h, "5", "Beats", ["Strong AND evenly spaced: the chain", "of moments that lands on peaks of", f"curve 3 but stays close to {period:.2f} s", "apart. A beat can fall on a quiet", "moment to keep the pulse."])
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
h = H[6]; panel(y, h, "6", "Downbeats (our rule)", ["Count in fours (assumes 4/4). Of the 4", "ways to start counting, keep the one", "whose 1s land on the strongest onsets.", "librosa doesn't do this step."])
n = 0
for t in bt:
    if not (t0 <= t <= t1): continue
    k = (list(bt).index(t) - phase) % 4 + 1
    isd = k == 1
    add(f'<line x1="{X(t):.1f}" y1="{y + 22}" x2="{X(t):.1f}" y2="{y + h}" stroke="{DOWN if isd else BEAT}" stroke-width="{3 if isd else 1.5}"/>')
    add(f'<text x="{X(t):.1f}" y="{y + 17}" font-size="{14 if isd else 11}" font-weight="{700 if isd else 400}" text-anchor="middle" fill="{DOWN if isd else BEAT}">{k}</text>')
time_axis(y + h)
y += h + 40
add(f'<text x="16" y="{y - 8}" font-size="11.5" fill="{MUTED}">Steps 3–5: librosa.onset.onset_strength, librosa.beat.beat_track (tempo, then dynamic programming; tightness 100). '
    f'Step 6: pipeline/beat_grid.py. Generated by tools/beat_detection_svg.py.</text>')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{y}" viewBox="0 0 {W} {y}" '
       f'font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>' + "".join(out) + "</svg>")
open(a.out, "w").write(svg)
print(f"wrote {a.out}: tempo {tempo:.1f}, {len(wb)} beats in window, downbeat phase {phase}")
