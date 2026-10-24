"""Put the MASTER lyrics on Whisper's clock: words from the owner's lyrics file, timing from Whisper's word timestamps.
Whisper's own text is never used downstream (it misheard e.g. "PHP" as "D.H.P.", "GOG goin'" as "I'll go in Barbie").

    python3 pipeline/align_lyrics.py SONG.wav LYRICS.txt [--analysis DIR]
Reads  ANALYSIS/<song>-medium.json (whisper-timestamped, with per-word start/end)
Writes ANALYSIS/<song>-true-segments.json: {"segments": [{"start", "end", "text", "matched"}]}, one per lyric
line, same shape as <song>-medium-segments.json so gen_prompts_local.py can use it (--segments).

Alignment: the token streams of the master lyrics and of Whisper's words are matched with difflib; each master line
takes the start of its first matched token and the end of its last. Lines with no matched token are placed by linear
interpolation between their neighbours. "matched" is the fraction of the line's tokens Whisper also heard.
"""
from __future__ import annotations

import argparse, json, os, re, sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import analysis_dir

ap = argparse.ArgumentParser()
ap.add_argument("song")
ap.add_argument("lyrics")
ap.add_argument("--analysis", default=None)
a = ap.parse_args()
song = Path(a.song); ana = analysis_dir(song, a.analysis)
toks = lambda s: re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))

# {...} is Amber/casting direction (see docs/pocs/22-amber-investigation.md), not sung words; [Section] lines aren't sung
lines = [l for l in (re.sub(r"\{[^}]*\}", "", x).strip() for x in open(a.lyrics)) if l and not re.match(r"^\[.*\]$", l)]
mt, owner = [], []
for i, l in enumerate(lines):
    for t in toks(l):
        mt.append(t); owner.append(i)
wt, wtime = [], []
for seg in json.load(open(ana / f"{song.stem}-medium.json"))["segments"]:
    for w in seg.get("words", []):
        for t in toks(w["text"]):
            wt.append(t); wtime.append((w["start"], w["end"]))

hit = [None] * len(mt)
for b in SequenceMatcher(None, mt, wt, autojunk=False).get_matching_blocks():
    for k in range(b.size):
        hit[b.a + k] = wtime[b.b + k]
seg = []
for i, l in enumerate(lines):
    idx = [k for k, o in enumerate(owner) if o == i]
    got = [hit[k] for k in idx if hit[k]]
    seg.append({"start": got[0][0] if got else None, "end": got[-1][1] if got else None, "text": l,
                "matched": round(len(got) / max(len(idx), 1), 2)})
known = [i for i, s in enumerate(seg) if s["start"] is not None]
_ve = []


def vocal_end() -> float:
    """When the vocal stem last carries singing (RMS above 10% of its loud level), for trailing unheard lines."""
    if not _ve:
        import subprocess
        import numpy as np
        v = ana / f"{song.stem}-vocals.mp3"
        x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", str(v), "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
                                         capture_output=True).stdout, np.float32) if v.exists() else np.zeros(1, np.float32)
        r = np.sqrt(np.mean(x[: len(x) // 800 * 800].reshape(-1, 800) ** 2, axis=1)) if len(x) >= 800 else np.zeros(1)
        on = np.nonzero(r > 0.1 * np.percentile(r, 95))[0]
        _ve.append(float(on[-1] + 1) * 0.1 if len(on) else 0.0)
    return _ve[0]
for i, s in enumerate(seg):                                   # place unmatched lines between matched neighbours
    if s["start"] is None:
        lo = max([k for k in known if k < i], default=None); hi = min([k for k in known if k > i], default=None)
        if lo is not None and hi is not None:
            slot = (seg[hi]["start"] - seg[lo]["end"]) / (hi - lo)     # an equal share of the gap each (lines Whisper
            s["start"] = seg[lo]["end"] + (i - lo - 1) * slot            # missed, e.g. stacked voices, were zero-length)
            s["end"] = s["start"] + slot
        elif lo is not None and vocal_end() > seg[lo]["end"] + 1:   # trailing lines nobody heard (e.g. a final chorus):
            tail = [k for k in range(lo + 1, len(seg))]                # share out the time up to the end of the vocals
            slot = (vocal_end() - seg[lo]["end"]) / len(tail)
            s["start"] = seg[lo]["end"] + (i - lo - 1) * slot; s["end"] = s["start"] + slot
        else:
            s["start"] = s["end"] = seg[lo]["end"] if lo is not None else seg[hi]["start"]
for i in range(len(seg) - 1):                                 # monotonic, no overlaps
    seg[i + 1]["start"] = max(seg[i + 1]["start"], seg[i]["start"])
    seg[i]["end"] = max(seg[i]["end"], seg[i]["start"])
out = ana / f"{song.stem}-true-segments.json"
json.dump({"source": str(a.lyrics), "segments": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()}
                                                  for s in seg]}, open(out, "w"), indent=1)
weak = [s for s in seg if s["matched"] < 0.5]
print(f"{len(seg)} master lines aligned; {len(weak)} with <50% of words heard (timing interpolated/approximate) -> {out}")
for s in seg[:6] + weak[:6]:
    print(f"  {s['start']:6.2f}-{s['end']:6.2f}  m={s['matched']:.2f}  {s['text'][:70]}")
