"""A review sheet with one row per music layer, so each layer's visual can be judged (user, 2026-10-27: "when you do a POC
include a sample that has all 7 layers otherwise I can't evaluate"). Each row shows three frames around one event of that
layer (before / on / after), labelled with the row number, the layer and the frame.

    python3 pipeline/world/layer_sheet.py RENDER_DIR CURVES.json LAYERS.json OUT.jpg [--tag 12] [--start S]
RENDER_DIR holds 0001.png .. NNNN.png (frame f of the clip is file f); --start S: file 1 is song frame S (a dry run
from the middle of the song), the labels stay song frames.
"""
from __future__ import annotations

import argparse, json, os, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("render")
ap.add_argument("curves")
ap.add_argument("layers")
ap.add_argument("out")
ap.add_argument("--tag", default="")
ap.add_argument("--start", type=int, default=1)
a = ap.parse_args()
C, LY = json.load(open(a.curves)), json.load(open(a.layers))
N = len([f for f in os.listdir(a.render) if f.endswith(".png")])
OF = a.start - 1                        # the events and curves moved to the rendered frames
C["tweets"]["hits"] = [h - OF for h in C["tweets"]["hits"]]
C["bands"]["low"]["hits"] = [h - OF for h in C["bands"]["low"]["hits"]]
LY["pad"]["changes"] = [h - OF for h in LY["pad"]["changes"]]
LY["chime"] = [dict(n, frame=n["frame"] - OF) for n in LY["chime"]]
LY["clicks"]["hits"] = [h - OF for h in LY["clicks"]["hits"]]
LY["subbass"] = LY["subbass"][OF:]; LY["drone"]["energy"] = LY["drone"]["energy"][OF:]


def ok(f):                              # event frame f (0-based, as in the JSONs) is file f + 1
    return max(1, min(N, f + 1))


def mid(hits, lo=10):
    if not hits:
        return N // 2                   # an event well inside the clip, so its before/after frames exist
    hs = [h for h in hits if lo <= h <= N - 12]
    return hs[len(hs) // 2] if hs else hits[0]


def extremes(v):                        # the quietest and loudest frames
    i0 = min(range(N), key=lambda i: v[min(i, len(v) - 1)]); i1 = max(range(N), key=lambda i: v[min(i, len(v) - 1)])
    return i0, i1


IN = lambda hs: [h for h in hs if 0 <= h < N]   # events inside the rendered frames (a dry run renders only the start)
tw = IN(C["tweets"]["hits"]); t0 = max(tw, key=lambda h: sum(1 for x in tw if abs(x - h) < 8)) if tw else 1
kick = IN(C["bands"]["low"]["hits"]); k0 = mid(kick)
ch = IN(LY["pad"]["changes"]) or [N // 2]; c0 = mid(ch)
ph = IN([n["frame"] for n in LY["chime"]]); n0 = mid(ph)
x0 = max(((s_, h) for s_, h in zip(LY["clicks"]["strength"], LY["clicks"]["hits"]) if 0 <= h < N), default=(0, 1))[1]
s_lo, s_hi = extremes(LY["subbass"]); d_lo, d_hi = extremes(LY["drone"]["energy"])
rows = [("1 heartbeat (camera surge; motion, see video)", [k0 - 6, k0, k0 + 8]),
        ("2 swifts (tweet)", [t0, t0 + 3, t0 + 7]),
        ("3 sub-bass: ground drift (quiet / loud / loud+6)", [s_lo, s_hi, s_hi + 6]),
        ("4 chord change: cloud shadows", [c0 - 2, c0 + 14, c0 + 30]),
        ("5 chime: crystal lit", [n0 - 1, n0 + 1, n0 + 6]),
        ("6 drone: sun halo (quiet / loud / loud+6)", [d_lo, d_hi, d_hi + 6]),
        ("7 click: crack glint", [x0 - 1, x0, x0 + 2])]
ins, fc, st, lay = [], "", "", []
for r, (name, fr) in enumerate(rows):
    for c, f in enumerate(fr):
        i = r * 3 + c; f = ok(f)
        ins += ["-i", os.path.join(a.render, f"{f:04d}.png")]
        txt = (f"{a.tag} " if a.tag else "") + (f"{name}  f{f + OF}" if c == 0 else f"f{f + OF}")
        txt = txt.replace(":", "\\:").replace("'", "")
        fc += (f"[{i}]scale=640:-1,drawtext=text='{txt}':fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.65:"
               f"x=8:y=8[v{i}];")
        st += f"[v{i}]"; lay.append(f"{c * 640}_{r * 360}")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex",
                f"{fc}{st}xstack=inputs={len(rows) * 3}:layout={'|'.join(lay)}[o]", "-map", "[o]", "-q:v", "3", a.out],
               check=True)
print(a.out, {name.split(" ")[1]: fr for name, fr in rows})
