"""Zoom on onset picking: three candidate peaks in the onset-strength curve, each tested "strong enough?" -> SVG.

Two stages (librosa in the m2v-pre image, matplotlib-free drawing on the host):
    python3 tools/onset_svg.py dump SONG.wav DATA.json          (m2v-pre image)
    python3 tools/onset_svg.py draw DATA.json OUT.svg [--width 728]
librosa.onset.onset_detect's rule (util.peak_pick, defaults at 22 050 Hz, hop 256): the curve is scaled to 0-1 over the
song; a frame is an onset if it is the highest point from 30 ms before to just after it, stands at least 0.07 above the
average of the 100 ms either side, and comes 30 ms or more after the last onset. dump reproduces that rule, checks it
against onset_detect; the picture marks true peaks (highest within ~35 ms) with the same bar, and picks the first short stretch with three candidate peaks, at least one of them rejected.
"""
from __future__ import annotations

import argparse, html, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["dump", "draw"]); ap.add_argument("src"); ap.add_argument("dst")
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
    wait, delta = int(0.03 * SR // HOP), 0.07
    n = len(e)
    avg = np.array([e[max(0, i - pre_avg):min(n, i + post_avg)].mean() for i in range(n)])
    is_max = np.array([e[i] == e[max(0, i - pre_max):min(n, i + post_max)].max() for i in range(n)])
    mine, last = set(), -10 ** 9                                           # librosa's rule, to check the reproduction
    for i in range(n):
        if is_max[i] and e[i] >= avg[i] + delta and i > last + wait:
            mine.add(i); last = i
    agree = len(mine & ref) / max(len(ref), 1)
    print(f"reproduced onset_detect: {len(mine & ref)}/{len(ref)} onsets agree ({agree:.1%}); {len(mine - ref)} extra")
    # for the picture: true peaks (highest within +-3 frames, ~35 ms), each tested against the same bar
    cands = [(i, bool(e[i] >= avg[i] + delta)) for i in range(3, n - 3) if e[i] == e[i - 3:i + 4].max() and e[i] > 0.03]
    best = None
    for k in range(len(cands) - 2):
        tri = cands[k:k + 3]
        span = (tri[2][0] - tri[0][0]) * HOP / SR
        oks = [o for _, o in tri]
        if span <= 0.9 and oks.count(True) == 2 and tri[0][0] * HOP / SR > 30 \
                and e[[i for i, o in tri if not o][0]] > 0.07:                # the "no" is a visible bump
            m = min(abs(e[i] - avg[i] - delta) for i, _ in tri)             # every answer clear, not a hair's breadth
            f0_, f1_ = tri[0][0] - 5, tri[2][0] + 6
            quiet_edges = max(e[f0_:tri[0][0] - 1].max(), e[tri[2][0] + 2:f1_].max()) < 0.5 * min(e[i] for i, _ in tri if e[i] >= avg[i] + delta)
            if m > 0.03 and quiet_edges and (best is None or m > best[0]): best = (m, tri)
    _, tri = best
    f0, f1 = tri[0][0] - 5, tri[2][0] + 6
    json.dump({"name": a.src.split("/")[-1], "t0": f0 * HOP / SR, "dt": HOP / SR, "e": e[f0:f1].round(4).tolist(),
               "avg": avg[f0:f1].round(4).tolist(), "delta": delta, "agree": agree,
               "cands": [[i - f0, o, round(float(e[i]), 3), round(float(avg[i] + delta), 3)] for i, o in tri]},
              open(a.dst, "w"), indent=1)
    print(f"window {f0 * HOP / SR:.2f}-{f1 * HOP / SR:.2f} s; candidates {[(round(i * HOP / SR, 2), o) for i, o in tri]}")
    sys.exit()

D = json.load(open(a.src))
e, avg, delta, cands, dt, t0 = D["e"], D["avg"], D["delta"], D["cands"], D["dt"], D["t0"]
W = a.width; L, R, T = 40, 16, 120; PW = W - L - R; PH = 300
INK, MUTED, RULE, BAR, THR, YES, NO = "#1d2433", "#5b6475", "#d9dde5", "#e0732c", "#5b6475", "#2f9e6b", "#c0392b"
n = len(e); bw = PW / n
X = lambda i: L + (i + 0.5) * bw
top_v = max(max(e), max(a_ + delta for a_ in avg)) * 1.12
Y = lambda v: T + PH - v / top_v * PH
out = []; add = out.append; esc = html.escape
add(f'<text x="16" y="30" font-size="20" font-weight="700" fill="{INK}">Is this onset strong enough?</text>')
add(f'<text x="16" y="52" font-size="13" fill="{MUTED}">{esc(D["name"])}, {t0:.2f}–{t0 + n * dt:.2f} s · one bar = 12 ms</text>')
# legend
lx = 16
for col, lab, dash in ((BAR, "onset strength", ""), (THR, "local average + 0.07", "5 4")):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{lx}" y1="78" x2="{lx + 22}" y2="78" stroke="{col}" stroke-width="{8 if not dash else 2}"{da}/>')
    add(f'<text x="{lx + 28}" y="82" font-size="12" fill="{MUTED}">{lab}</text>'); lx += 200
add(f'<rect x="{L}" y="{T}" width="{PW}" height="{PH}" fill="#fafbfc" stroke="{RULE}"/>')
cset = {c[0]: c for c in cands}
for i, v in enumerate(e):
    col = (YES if cset[i][1] else NO) if i in cset else BAR
    add(f'<rect x="{L + i * bw + 1:.1f}" y="{Y(v):.1f}" width="{bw - 2:.1f}" height="{T + PH - Y(v):.1f}" fill="{col}" opacity="{1 if i in cset else 0.55}"/>')
pts = " ".join(f"{X(i):.1f},{Y(v + delta):.1f}" for i, v in enumerate(avg))
add(f'<polyline points="{pts}" fill="none" stroke="{THR}" stroke-width="2" stroke-dasharray="5 4"/>')
for k, (i, ok, v, thr) in enumerate(cands):
    letter = "ABC"[k]; col = YES if ok else NO
    add(f'<line x1="{X(i) - 10:.1f}" y1="{Y(thr):.1f}" x2="{X(i) + 10:.1f}" y2="{Y(thr):.1f}" stroke="{col}" stroke-width="2.5"/>')
    yl = min(min(Y(x) for x in e[max(0, i - 3):i + 4]), Y(thr)) - 30
    add(f'<text x="{X(i):.1f}" y="{yl:.1f}" font-size="15" font-weight="700" text-anchor="middle" fill="{col}">{letter}: {"yes" if ok else "no"}</text>')
    add(f'<text x="{X(i):.1f}" y="{yl + 15:.1f}" font-size="11" text-anchor="middle" fill="{MUTED}">{v:.2f} vs {thr:.2f}</text>')
for s in range(0, n, 10):
    add(f'<text x="{X(s):.1f}" y="{T + PH + 16}" font-size="10.5" text-anchor="middle" fill="{MUTED}">{t0 + s * dt:.2f} s</text>')
for v in (0, 0.5, 1.0):
    if v <= top_v: add(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" font-size="10.5" text-anchor="end" fill="{MUTED}">{v:.1f}</text>')
yy = T + PH + 44
for k, (i, ok, v, thr) in enumerate(cands):
    add(f'<text x="16" y="{yy + k * 18}" font-size="12.5" fill="{YES if ok else NO}">{"ABC"[k]}: {"clears the bar → onset" if ok else "below the bar → not an onset"}</text>')
yy += 3 * 18 + 14
add(f'<text x="16" y="{yy}" font-size="11" fill="{MUTED}">Rule: librosa onset_detect (peak above local average + 0.07); '
    f'shown at each peak. tools/onset_svg.py</text>')
yy += 14
open(a.dst, "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{yy}" viewBox="0 0 {W} {yy}" '
                       f'font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>' + "".join(out) + "</svg>")
print(f"wrote {a.dst}")
