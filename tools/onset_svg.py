"""Zoom on onset picking: which bars of the onset-strength curve librosa calls onsets, and why the others aren't -> SVG.

Two stages (librosa in the m2v-pre image, drawing on the host):
    python3 tools/onset_svg.py dump SONG.wav DATA.json [--at 91.05] [--secs 0.3]     (m2v-pre image)
    python3 tools/onset_svg.py draw DATA.json OUT.svg [--width 728]
librosa.onset.onset_detect's rule (util.peak_pick, defaults at 22 050 Hz, hop 256), with the curve scaled to 0-1 over
the song. A bar is an onset if it:
  1 is at or above the dashed line: the average of the bars from 100 ms before to 100 ms after it, + 0.07;
  2 is at least as tall as the 2 bars before it (still rising or at the top);
  3 comes more than 2 bars (~30 ms) after the last onset.
dump reproduces the rule bar by bar, checks it against onset_detect over the whole song, and stores the window.
"""
from __future__ import annotations

import argparse, html, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["dump", "draw"]); ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--at", type=float, default=91.05); ap.add_argument("--secs", type=float, default=0.3)
ap.add_argument("--width", type=int, default=728)
a = ap.parse_args()
SR, HOP = 22050, 256

if a.mode == "dump":
    import librosa, numpy as np
    y, _ = librosa.load(a.src, sr=SR)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    ref = set(librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP).tolist())
    e = (env - env.min()) / (env.max() - env.min())                       # onset_detect's normalisation
    pre_max, post_max = int(0.03 * SR // HOP), int(0.00 * SR // HOP + 1)
    pre_avg, post_avg = int(0.10 * SR // HOP), int(0.10 * SR // HOP + 1)
    wait, delta, n = int(0.03 * SR // HOP), 0.07, len(e)
    line = np.array([e[max(0, i - pre_avg):min(n, i + post_avg)].mean() + delta for i in range(n)])
    rising = np.array([e[i] == e[max(0, i - pre_max):min(n, i + post_max)].max() for i in range(n)])
    why, last = [], -10 ** 9
    for i in range(n):
        if e[i] < line[i]: why.append("below")
        elif not rising[i]: why.append("falling")
        elif i <= last + wait: why.append("too soon")
        else: why.append("onset"); last = i
    mine = {i for i, w in enumerate(why) if w == "onset"}
    print(f"reproduced onset_detect: {len(mine & ref)}/{len(ref)} agree, {len(mine - ref)} extra")
    f0 = int(round(a.at * SR / HOP)); f1 = f0 + int(round(a.secs * SR / HOP))
    json.dump({"name": a.src.split("/")[-1], "t0": f0 * HOP / SR, "dt": HOP / SR, "e": e[f0:f1].round(4).tolist(),
               "line": line[f0:f1].round(4).tolist(), "why": why[f0:f1],
               "agree": f"{len(mine & ref)}/{len(ref)}"}, open(a.dst, "w"), indent=1)
    print("window:", " ".join(w[0] for w in why[f0:f1]))
    sys.exit()

D = json.load(open(a.src))
e, line, why, dt, t0 = D["e"], D["line"], D["why"], D["dt"], D["t0"]
W = a.width; L, R = 40, 16; PW = W - L - R; PH = 280
INK, MUTED, RULE = "#1d2433", "#5b6475", "#d9dde5"
COL = {"onset": "#2f9e6b", "falling": "#e8a87c", "too soon": "#9db8e0", "below": "#c9ced8"}
LAB = {"onset": "onset", "falling": "fails 2: falling", "too soon": "fails 3: too soon", "below": "fails 1: below line"}
out = []; add = out.append; esc = html.escape
add(f'<text x="16" y="30" font-size="20" font-weight="700" fill="{INK}">Which bars are onsets?</text>')
add(f'<text x="16" y="52" font-size="12.5" fill="{MUTED}">Bars: how much louder the sound just got, every 12 ms.</text>')
add(f'<text x="16" y="78" font-size="13.5" font-weight="700" fill="{INK}">A bar is an onset if it:</text>')
rules = ["1  is above the dashed line", "2  is as tall as the 2 bars before it", "3  comes 30 ms after the last onset"]
for k, r in enumerate(rules):
    add(f'<text x="16" y="{98 + k * 19}" font-size="13" fill="{INK}">{esc(r)}</text>')
add(f'<text x="16" y="{98 + 3 * 19 + 2}" font-size="12" fill="{MUTED}">Dashed line: moving average (100 ms each side) + 0.07</text>')
ly = 98 + 4 * 19 + 14
for k, key in enumerate(["onset", "below", "falling", "too soon"]):
    lx = 16 + k * 172
    add(f'<rect x="{lx}" y="{ly - 10}" width="14" height="12" fill="{COL[key]}"/>')
    add(f'<text x="{lx + 20}" y="{ly}" font-size="12" fill="{MUTED}">{esc(LAB[key])}</text>')
T = ly + 26
n = len(e); bw = PW / n
X = lambda i: L + (i + 0.5) * bw
top_v = max(max(e), max(line)) * 1.15
Y = lambda v: T + PH - v / top_v * PH
add(f'<rect x="{L}" y="{T}" width="{PW}" height="{PH}" fill="#fafbfc" stroke="{RULE}"/>')
for i, v in enumerate(e):
    add(f'<rect x="{L + i * bw + 1:.1f}" y="{Y(v):.1f}" width="{bw - 2:.1f}" height="{T + PH - Y(v):.1f}" fill="{COL[why[i]]}"/>')
    if why[i] == "onset":
        yl = min(Y(x) for x in e[i:i + 3]) - 8                           # above this bar and the next two
        add(f'<text x="{X(i):.1f}" y="{yl:.1f}" font-size="12.5" font-weight="700" text-anchor="middle" fill="{COL["onset"]}">onset</text>')
        add(f'<line x1="{X(i):.1f}" y1="{yl + 3:.1f}" x2="{X(i):.1f}" y2="{Y(v) - 2:.1f}" stroke="{COL["onset"]}" stroke-width="1.2"/>')
pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(line))
add(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="6 4"/>')
for s in range(0, n, 8):
    add(f'<text x="{X(s):.1f}" y="{T + PH + 15}" font-size="10.5" text-anchor="middle" fill="{MUTED}">{t0 + s * dt:.2f} s</text>')
for v in (0, 0.5):
    if v <= top_v: add(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" font-size="10.5" text-anchor="end" fill="{MUTED}">{v:.1f}</text>')
yy = T + PH + 38
add(f'<text x="16" y="{yy}" font-size="11" fill="{MUTED}">{esc(D["name"])}, {t0:.2f}–{t0 + n * dt:.2f} s · librosa onset_detect, '
    f'rule reproduced ({D["agree"]} onsets in the song) · tools/onset_svg.py</text>')
yy += 14
open(a.dst, "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{yy}" viewBox="0 0 {W} {yy}" '
                       f'font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>' + "".join(out) + "</svg>")
print(f"wrote {a.dst}")
