"""Shot list + keyframes: one shot per bar (downbeat to downbeat), from full-metadata<tag>.json and beats.json.

    ./dr.sh python3 pipeline/keyframes.py SONG.wav --style "..." --out OUT [--analysis DIR] [--size 1024x576]

One shot per downbeat. Each shot uses the music2vid prompt active at its start (visual_description before the first
lyric line), plus the style. Keyframes by MOSS-Image (text-to-image), seed = 1000 + shot index.
Writes OUT/shots.json and OUT/keys/shot_XXX.png.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import analysis_dir


ap = argparse.ArgumentParser()
ap.add_argument("song")
ap.add_argument("--style", required=True)
ap.add_argument("--size", default="1024x576")
ap.add_argument("--out", required=True)
ap.add_argument("--analysis", default=None)
ap.add_argument("--model", default="moss")
ap.add_argument("--plan-only", action="store_true")
ap.add_argument("--replan", action="store_true", help="rebuild shots.json even if it exists (it may hold scenes.py edits)")
ap.add_argument("--meta", default="full-metadata.json", help="e.g. full-metadata-true.json (master-lyric prompts)")
ap.add_argument("--keys", default="keys")
ap.add_argument("--skip-cast", action="store_true", help="leave shots that show the performer (cast scene / sing / "
                "part, from prompts_per_line.py --cues) to amber_keys.py")
ap.add_argument("--from-end", action="store_true", help="work backwards and stop at the first key that exists: a second "
                "box can help a running job from the other end (keys synced between them)")
ap.add_argument("--max-shots", type=int, default=0, help="render only the first N (benchmarks)")
a = ap.parse_args()
song = Path(a.song); ana = analysis_dir(song, a.analysis)
meta = json.load(open(ana / a.meta)); beats = json.load(open(ana / "beats.json"))
dur = float(meta["duration"])
segs = meta["full_transcript"]["segments"]
down = [0.0] + [t for t in beats["downbeats"] if t > 0.3] + [dur]          # a shot boundary at every downbeat


def seg_at(t):
    cur = {"prompt": meta["visual_description"]}
    for s in segs:
        if s["start"] <= t + 0.25:                                           # a line starting just after the cut counts
            cur = s
    return cur


composed = meta.get("composed", False)        # treatment.py prompts already lead with the look: append no style
style = "" if composed else a.style
shots = []
for i in range(len(down) - 1):
    s = seg_at(down[i])
    shots.append({"i": i, "start": round(down[i], 4), "end": round(down[i + 1], 4), "prompt": s["prompt"],
                  **({"word": s["word"]} if s.get("word") else {}),
                  **({"cast": s["cast"], "cast_note": s.get("cast_note", "")} if s.get("cast") else {})})
os.makedirs(os.path.join(a.out, a.keys), exist_ok=True)
_sp = os.path.join(a.out, "shots.json")
if os.path.exists(_sp) and not a.replan:          # keep a shot list edited by scenes.py apply (prompts, casts)
    shots = json.load(open(_sp))["shots"]
    print(f"{len(shots)} shots (existing shots.json kept; --replan to rebuild it)")
    if a.plan_only:
        sys.exit(0)
else:
  json.dump({"style": style, "beats": beats["beats"], "downbeats": beats["downbeats"], "duration": dur, "shots": shots,
           **({"treatment": meta["treatment"]} if "treatment" in meta else {})},
          open(os.path.join(a.out, "shots.json"), "w"), indent=1)
  print(f"{len(shots)} shots, {len(set(s['prompt'] for s in shots))} distinct prompts")
  if a.plan_only:
      sys.exit(0)

from keyframe_model import load
W, H = (int(x) for x in a.size.split("x"))
r = load(a.model, offload=os.environ.get("OFFLOAD", "0") == "1")      # no CPU offload: run where the model fits
CAST = ("scene", "sing", "part")
_rr = os.path.join(a.out, "reroll.json")                                 # check_keys.py --reroll: a new seed per attempt
REROLL = json.load(open(_rr)) if os.path.exists(_rr) else {}
seed = lambda i: 1000 + i + 7919 * REROLL.get(str(i), 0)
for s in (shots[::-1] if a.from_end else shots[:a.max_shots or None]):
    if a.skip_cast and s.get("cast") in CAST:
        continue
    p = os.path.join(a.out, a.keys, f"shot_{s['i']:03d}.png")
    if os.path.exists(p):
        if a.from_end:
            break                                                            # met the other box's keys
        continue
    t = time.time()
    im, _ = r.render(None, [(f"{s['prompt']}, {style}" if style else s["prompt"], 1.0)], 1.0, seed=seed(s["i"]), size=(W, H))
    im.save(p)
    print(f"shot {s['i']:3d} {s['start']:6.2f}s {time.time()-t:.0f}s  {s['prompt'][:70]}", flush=True)
