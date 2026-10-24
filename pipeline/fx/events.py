"""Event times for effects, from the piece's analysis.

    {"source": "drum_hits",                       analysis/drum_hits.json (pipeline/drum_hits.py)
     "pattern": "triplet" | "runs" | "all" | "isolated_triple",
                                                   triplet: a strong low hit then two more ~0.48 s and ~0.63 s apart;
                                                   runs: hits inside runs of 3+; all: every hit;
                                                   isolated_triple: three loud hits in "band" (Hz, default 2000-8000)
                                                   "gap" s apart (default [0.1, 0.25]) with nothing as loud within
                                                   0.4 s either side, each at least "level" (0-1) of the loudest such
                                                   triple (the noble-hunt tat-tat-tat: a sharp snare/tom triple)
     "times": [75.58, 136.21],                     or: fixed times (s), overriding detection (strength 1)
     "min_strength": 0.55,                         the first (or every) hit's strength floor, 0-1
     "after_lyric": "noble hunt",                  optional: only events from WINDOW[0] s before a line containing
     "window": [-0.5, 4.0]}                        this text starts to WINDOW[1] s after it ends (lyric timings)
Returns [(t, strength)] sorted by time.
"""
from __future__ import annotations

import glob, json, os


def isolated_triples(analysis: str, band=(2000, 8000), gap=(0.1, 0.25), level=0.35) -> list:
    import subprocess
    import numpy as np
    stem = sorted(glob.glob(os.path.join(analysis, "*-drums.mp3")))[0]
    sr, hop, n = 22050, 256, 1024
    x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", stem, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                                     capture_output=True, check=True).stdout, np.float32)
    fr = np.fft.rfftfreq(n, 1 / sr)
    b = (fr > band[0]) & (fr < band[1])
    m = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n), axis=1))[:, b]
    fl = np.r_[0, np.maximum(0, np.diff(m, axis=0)).sum(1)]
    t = np.arange(len(fl)) * hop / sr
    pk = [(float(t[i]), float(fl[i])) for i in range(1, len(fl) - 1) if fl[i] >= fl[i - 1] and fl[i] >= fl[i + 1]]
    top = max(v for _, v in pk)
    cand = []
    for lvl in (0.15,):
        P = [p for p in pk if p[1] >= lvl * top]
        for k in range(len(P) - 2):
            a, bb, c = P[k], P[k + 1], P[k + 2]
            if gap[0] <= bb[0] - a[0] <= gap[1] and gap[0] <= c[0] - bb[0] <= gap[1]:
                weakest = min(a[1], bb[1], c[1])
                near = [p for p in P if (a[0] - 0.4 < p[0] < a[0] - 0.05 or c[0] + 0.05 < p[0] < c[0] + 0.4)
                        and p[1] >= 0.5 * weakest]
                if not near:
                    cand.append((a, bb, c, weakest))
    if not cand:
        return []
    ref = max(w for *_, w in cand)
    ev = []
    for a, bb, c, w in cand:
        if w >= level * ref:
            ev += [(round(p[0], 3), round(p[1] / top, 3)) for p in (a, bb, c)]
    return ev


def lyric_filter(ev: list, spec: dict, analysis: str, whole: int = 1) -> list:
    """Keep events (in groups of WHOLE, judged by the group's first) inside a window around lines containing
    spec["after_lyric"]: from WINDOW[0] s before the line starts to WINDOW[1] s after it ends."""
    if not spec.get("after_lyric"):
        return ev
    seg_p = [p for p in sorted(glob.glob(os.path.join(analysis, "*-true-segments.json")))]
    segs = json.load(open(seg_p[0]))["segments"] if seg_p else []
    lo, hi = spec.get("window", [-0.5, 4.0])
    wins = [(s["start"] + lo, s["end"] + hi) for s in segs if spec["after_lyric"].lower() in s["text"].lower()]
    keep = []
    for k in range(0, len(ev), whole):
        if any(a <= ev[k][0] <= b for a, b in wins):
            keep += ev[k:k + whole]
    return keep


def select(spec: dict, analysis: str) -> list:
    if spec.get("times"):
        return sorted((float(t), 1.0) for t in spec["times"])
    pat = spec.get("pattern", "all")
    if pat == "isolated_triple":
        ev = isolated_triples(analysis, tuple(spec.get("band", (2000, 8000))), tuple(spec.get("gap", (0.1, 0.25))),
                              float(spec.get("level", 0.35)))
        return sorted(set(lyric_filter(ev, spec, analysis, whole=3)))
    hits = json.load(open(os.path.join(analysis, "drum_hits.json")))
    hs, runs = hits["hits"], hits["runs"]
    pat, floor = spec.get("pattern", "all"), float(spec.get("min_strength", 0.3))
    ev = []
    if pat == "triplet":
        for i, a in enumerate(hs):
            if a["strength"] < floor:
                continue
            b = next((h for h in hs[i + 1:] if 0.38 <= h["t"] - a["t"] <= 0.58), None)
            c = next((h for h in hs if b and 0.5 <= h["t"] - b["t"] <= 0.72), None)
            if b and c:
                ev += [(a["t"], a["strength"]), (b["t"], b["strength"]), (c["t"], c["strength"])]
    elif pat == "runs":
        ev = [(h["t"], h["strength"]) for h in hs if h["strength"] >= floor
              and any(r["start"] - 1e-3 <= h["t"] <= r["end"] + 1e-3 for r in runs)]
    else:
        ev = [(h["t"], h["strength"]) for h in hs if h["strength"] >= floor]
    ev = lyric_filter(ev, spec, analysis, whole=3 if pat == "triplet" else 1)
    return sorted(set(ev))
