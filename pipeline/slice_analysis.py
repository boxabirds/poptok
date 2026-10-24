"""Cut a song's analysis to a time window, for a short test piece (poptok --test): stems trimmed, word / line / beat
timings shifted so the window starts at 0. Nothing is recomputed, so the test uses exactly the full song's analysis.

    python3 pipeline/slice_analysis.py SRC_ANALYSIS SRC_STEM DST_ANALYSIS DST_STEM START DURATION
"""
import json, os, subprocess, sys

src, sstem, dst, dstem, st, dur = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6])
os.makedirs(dst, exist_ok=True)
inside = lambda t: st <= t < st + dur
for stem in ("vocals", "drums", "bass", "other"):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(st), "-t", str(dur), "-i",
                    f"{src}/{sstem}-{stem}.mp3", f"{dst}/{dstem}-{stem}.mp3"], check=True)
w = json.load(open(f"{src}/{sstem}-medium.json"))                      # Whisper with word timings
segs = []
for sg in w["segments"]:
    words = [dict(x, start=x["start"] - st, end=x["end"] - st) for x in sg.get("words", []) if inside(x["start"])]
    if words:
        segs.append(dict(sg, start=words[0]["start"], end=words[-1]["end"], words=words))
json.dump(dict(w, segments=segs), open(f"{dst}/{dstem}-medium.json", "w"), indent=1)
json.dump({"segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs]},
          open(f"{dst}/{dstem}-medium-segments.json", "w"), indent=1)
t = json.load(open(f"{src}/{sstem}-true-segments.json"))               # master lyrics on the song's clock
json.dump(dict(t, segments=[dict(s, start=s["start"] - st, end=s["end"] - st) for s in t["segments"]
                            if st - 0.5 <= s["start"] < st + dur - 0.5]), open(f"{dst}/{dstem}-true-segments.json", "w"), indent=1)
b = json.load(open(f"{src}/beats.json"))
json.dump(dict(b, beats=[round(x - st, 4) for x in b["beats"] if inside(x)],
               downbeats=[round(x - st, 4) for x in b["downbeats"] if inside(x)]), open(f"{dst}/beats.json", "w"), indent=1)
if os.path.exists(f"{src}/metadata.json"):
    subprocess.run(["cp", f"{src}/metadata.json", f"{dst}/metadata.json"], check=True)
print(f"analysis cut to {st:.2f}-{st + dur:.2f}s -> {dst}")
