"""Illustrate how the stem splitter works (Demucs mdx_extra, as pipeline/analyse.py), on a real song -> SVG.

    python3 tools/demucs_svg.py ANALYSIS_DIR STEM OUT.svg --mix SONG.wav [--start 40] [--secs 8] [--width 728]   (host)
Panels: 1 the mix (wave, spectrogram); 2 one model: two branches meeting in the middle; 3 four models averaged;
4 the four stems the pipeline made; 5 check: the stems added up against the mix; 6 how it learned.
Model facts (read from the checkpoint, demucs 4.0.1 get_model("mdx_extra")): a bag of 4 HDemucs (Hybrid Demucs v3)
models, 83.6 M parameters each, equal weights; each with a waveform branch and a spectrogram branch (FFT 4096), 6
layers deep; 44.1 kHz stereo. The stems are MP3s, which start ~25 ms late (the encoder's delay): they are lined up
with the mix by cross-correlation before the check.
"""
from __future__ import annotations

import argparse, base64, html, io, subprocess, textwrap

import matplotlib; matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ana"); ap.add_argument("stem"); ap.add_argument("out"); ap.add_argument("--mix", required=True)
ap.add_argument("--start", type=float, default=40.0); ap.add_argument("--secs", type=float, default=8.0)
ap.add_argument("--width", type=int, default=728)
a = ap.parse_args()
SR, PAD = 44100, 0.5          # full rate: at 22 kHz the 25 ms MP3 delay falls between samples and the check reads 24%, not 3%
t0, t1 = a.start, a.start + a.secs


def load(f, pad=0.0):
    x = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t0 - pad), "-t", str(a.secs + 2 * pad), "-i", f, "-ac", "1",
                        "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(x, np.float32)


def spec(x, n=4096, hop=1024):
    w = np.hanning(n)
    S = np.array([np.abs(np.fft.rfft(x[i:i + n] * w)) for i in range(0, len(x) - n, hop)]).T[:n // 8:2]   # 0-5.5 kHz
    return 20 * np.log10(S + 1e-6)


names = ["vocals", "drums", "bass", "other"]
mix = load(a.mix)
raw = {k: load(f"{a.ana}/{a.stem}-{k}.mp3", PAD) for k in names}
tot = sum(raw.values()); p = int(PAD * SR); n = len(mix)
lag = max(range(0, 2 * p), key=lambda d: float(np.dot(mix[:n - 2 * p], tot[d:d + n - 2 * p])))   # MP3 start delay
stems = {k: v[lag:lag + n] for k, v in raw.items()}
n = min(n, *(len(v) for v in stems.values())); mix = mix[:n]; stems = {k: v[:n] for k, v in stems.items()}
total = sum(stems.values()); resid = mix - total
res_pct = 100 * np.sqrt(np.mean(resid ** 2)) / np.sqrt(np.mean(mix ** 2))
Smix = spec(mix); hi = Smix.max(); lo = hi - 80

W = a.width; L = 16; PW = W - 32
INK, MUTED, RULE, ACC, ACC2 = "#1d2433", "#5b6475", "#d9dde5", "#2f6fdb", "#e0732c"
COL = {"vocals": "#c0392b", "drums": "#2f6fdb", "bass": "#2f9e6b", "other": "#8e44ad"}
out = []; add = out.append; esc = html.escape


def para(y0, text, size=12.5):
    lines = textwrap.wrap(text, int(PW / (size * 0.53)))
    for k, line in enumerate(lines):
        add(f'<text x="16" y="{y0 + k * size * 1.3:.1f}" font-size="{size}" fill="{MUTED}">{esc(line)}</text>')
    return y0 + len(lines) * size * 1.3


def panel(y0, h, num, title, cap, framed=True):
    add(f'<text x="16" y="{y0 + 16}" font-size="16" font-weight="700" fill="{INK}">{num}  {esc(title)}</text>')
    top = para(y0 + 36, cap) - 4
    if framed: add(f'<rect x="{L}" y="{top}" width="{PW}" height="{h}" fill="#fafbfc" stroke="{RULE}"/>')
    return top


def wave(x, x0, y0, w, h, col, scale):
    c = max(1, len(x) // int(w)); mid = y0 + h / 2
    for i in range(int(w)):
        s = x[i * c:(i + 1) * c]
        if len(s): add(f'<line x1="{x0 + i}" y1="{mid - s.max() / scale * (h / 2 - 3):.1f}" x2="{x0 + i}" y2="{mid - s.min() / scale * (h / 2 - 3):.1f}" stroke="{col}"/>')


def img(S, x0, y0, w, h):
    v = np.clip((S - lo) / (hi - lo), 0, 1)[::-1]; b = io.BytesIO()
    mpimg.imsave(b, v, cmap="Blues", format="png", vmin=0, vmax=1)
    add(f'<image x="{x0}" y="{y0}" width="{w}" height="{h}" preserveAspectRatio="none" href="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}"/>')


def box(x, y, w, h, label, fill, sub="", fs=11):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{fill}" stroke="{INK}" stroke-width="0.7"/>')
    add(f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + (-1 if sub else 4):.1f}" font-size="{fs}" text-anchor="middle" fill="{INK}">{esc(label)}</text>')
    if sub: add(f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + 12:.1f}" font-size="9.5" text-anchor="middle" fill="{MUTED}">{esc(sub)}</text>')


def arrow(x1, y1, x2, y2):
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#ah)"/>')


add(f'<text x="16" y="30" font-size="20" font-weight="700" fill="{INK}">How the stem splitter works (Demucs)</text>')
y = para(52, f"{a.stem}, {t0:.0f}–{t1:.0f} s · model mdx_extra", 13) + 6
sc = np.abs(mix).max()

# 1 the mix
h = 150; top = panel(y, h, "1", "The mix", "One stream: wave, then spectrogram")
wave(mix, L, top + 4, PW, 56, ACC, sc); img(Smix, L, top + 64, PW, h - 66)
y = top + h + 30

# 2 one model
h = 210; top = panel(y, h, "2", "One model, two branches", "Squeeze down, meet, expand back")
cx = L + 10; iw = 74
box(cx, top + 16, iw, 30, "wave", "#e8f0fd"); box(cx, top + h - 46, iw, 30, "spectrogram", "#e8f0fd", fs=10)
ex = cx + iw + 16; lw, gap = 24, 4
mw = 84; ox_w = 70
dx0 = L + PW - 10 - ox_w - 14 - 6 * (lw + gap)
mx = (ex + 6 * (lw + gap) + dx0) / 2 - mw / 2
for i in range(6):
    sc_ = 1 - i * 0.11
    for yy in (top + 16, top + h - 46):
        box(ex + i * (lw + gap), yy + (1 - sc_) * 15, lw, 30 * sc_, str(i + 1), "#eef2f7", fs=9.5)
        box(dx0 + i * (lw + gap), yy + (1 - (0.45 + i * 0.11)) * 15, lw, 30 * (0.45 + i * 0.11), str(6 - i), "#eef2f7", fs=9.5)
arrow(cx + iw, top + 31, ex - 2, top + 31); arrow(cx + iw, top + h - 31, ex - 2, top + h - 31)
box(mx, top + h / 2 - 24, mw, 48, "shared middle", "#fde9dc", "seconds at once", fs=10.5)
ee = ex + 6 * (lw + gap) - gap
arrow(ee, top + 31, mx + 14, top + h / 2 - 26); arrow(ee, top + h - 31, mx + 14, top + h / 2 + 26)
arrow(mx + mw - 14, top + h / 2 - 26, dx0, top + 31); arrow(mx + mw - 14, top + h / 2 + 26, dx0, top + h - 31)
add(f'<path d="M{ex + 12},{top + 14} C{ex + 120},{top - 6} {dx0 + 60},{top - 6} {dx0 + 5 * (lw + gap) + 12},{top + 14}" fill="none" stroke="{ACC}" stroke-dasharray="4 3"/>')
add(f'<text x="{(ex + dx0 + 5 * (lw + gap)) / 2 + 12:.1f}" y="{top - 6}" font-size="9.5" text-anchor="middle" fill="{ACC}">shortcuts: fine detail</text>')
ox = L + PW - 10 - ox_w
for k, nm in enumerate(names):
    yy = top + 52 + k * 27
    box(ox, yy, ox_w, 22, nm, "#ffffff", fs=10.5); add(f'<rect x="{ox}" y="{yy}" width="4" height="22" fill="{COL[nm]}"/>')
de = dx0 + 6 * (lw + gap) - gap
arrow(de, top + 31, ox + 20, top + 50); arrow(de, top + h - 31, ox + 20, top + 52 + 4 * 27 - 3)
add(f'<text x="{ox + ox_w / 2}" y="{top + h - 6}" font-size="9.5" text-anchor="middle" fill="{MUTED}">both branches added</text>')
y = top + h + 30

# 3 bag of 4
h = 74; top = panel(y, h, "3", "Four models, averaged", "Averaging evens out mistakes", framed=False)
bw = (PW - 150) / 4 - 8
for k in range(4):
    box(L + k * (bw + 8), top + 8, bw, 34, f"model {k + 1}", "#eef2f7", "84 M numbers")
    add(f'<line x1="{L + k * (bw + 8) + bw / 2:.1f}" y1="{top + 42}" x2="{L + k * (bw + 8) + bw / 2:.1f}" y2="{top + 56}" stroke="{MUTED}" stroke-width="1.2"/>')
add(f'<line x1="{L + bw / 2:.1f}" y1="{top + 56}" x2="{L + 3 * (bw + 8) + bw / 2:.1f}" y2="{top + 56}" stroke="{MUTED}" stroke-width="1.2"/>')
arrow(L + 3 * (bw + 8) + bw / 2, top + 56, L + PW - 136, top + 56)
box(L + PW - 136, top + 41, 80, 30, "average", "#fde9dc", fs=10.5)
arrow(L + PW - 56, top + 56, L + PW - 40, top + 56)
add(f'<text x="{L + PW - 36}" y="{top + 60}" font-size="10.5" fill="{INK}">stems</text>')
y = top + h + 34

# 4 stems
rowh = 62; h = rowh * 4 + 6; top = panel(y, h, "4", "The four stems", "The pipeline's own files")
for k, nm in enumerate(names):
    yy = top + 3 + k * rowh
    add(f'<rect x="{L + 3}" y="{yy}" width="4" height="{rowh - 6}" fill="{COL[nm]}"/>')
    add(f'<text x="{L + 12}" y="{yy + 13}" font-size="11" font-weight="700" fill="{COL[nm]}">{nm}</text>')
    wave(stems[nm], L + 12, yy + 16, 200, rowh - 22, COL[nm], sc)
    img(spec(stems[nm]), L + 222, yy, PW - 225, rowh - 6)
y = top + h + 30

# 5 check
h = 90; top = panel(y, h, "5", "Check: add them up", f"Leftover: {res_pct:.0f}% of the mix")
half = PW / 2
wave(total, L + 3, top + 14, half - 8, h - 18, ACC, sc); wave(resid, L + half + 3, top + 14, half - 8, h - 18, ACC2, sc)
add(f'<text x="{L + 6}" y="{top + 12}" font-size="10" fill="{MUTED}">sum of 4 stems</text>')
add(f'<text x="{L + half + 6}" y="{top + 12}" font-size="10" fill="{MUTED}">mix − sum, same scale</text>')
y = top + h + 30

# 6 training
h = 92; top = panel(y, h, "6", "How it learned", "Songs with known separate tracks", framed=False)
steps = [("known tracks", ""), ("add up", "= a mix"), ("separate", "4 guesses"), ("compare", "with truth"), ("adjust", "the numbers")]
sw = (PW - 4 * 18) / 5
for k, (t, s) in enumerate(steps):
    box(L + k * (sw + 18), top + 6, sw, 40, t, "#eef2f7", s, fs=10.5)
    if k: arrow(L + k * (sw + 18) - 16, top + 26, L + k * (sw + 18) - 2, top + 26)
x4, x2 = L + 4 * (sw + 18) + sw / 2, L + 2 * (sw + 18) + sw / 2
add(f'<path d="M{x4:.1f},{top + 46} C{x4:.1f},{top + 72} {x2:.1f},{top + 72} {x2:.1f},{top + 48}" fill="none" stroke="{MUTED}" marker-end="url(#ah)"/>')
add(f'<text x="{(x4 + x2) / 2:.1f}" y="{top + 78}" font-size="10" text-anchor="middle" fill="{MUTED}">repeat millions of times</text>')
y = top + h + 14
y = para(y, f"Model read from the checkpoint (demucs 4.0.1). Stems from pipeline/analyse.py, lined up with the mix "
            f"({(lag - p) / SR * 1000:.0f} ms MP3 delay). Generated by tools/demucs_svg.py.", 11) + 4

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{y:.0f}" viewBox="0 0 {W} {y:.0f}" font-family="Helvetica, Arial, sans-serif">'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs><rect width="100%" height="100%" fill="#ffffff"/>' + "".join(out) + "</svg>")
open(a.out, "w").write(svg)
print(f"wrote {a.out}; MP3 delay {(lag - p) / SR * 1000:.1f} ms; leftover {res_pct:.1f}% of mix level")
