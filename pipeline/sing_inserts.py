"""Sung inserts (POC 22 Q2/Q3): the performer's lip-synced performance cut into a finished video wherever the cues say
"sing". Each run of consecutive sung shots becomes one lip-synced clip (InfiniteTalk by default: it articulates the
words correctly where LongCat 1.5 lost the r's, user review 2026-10-24), driven by the vocal stem for exactly
that stretch of the song and starting from the run's first casting still (so it keeps the shot's place and look). The
runs start and end on shot boundaries, which are downbeats, and the clip starts with the song, so it stays in sync.

    python3 pipeline/sing_inserts.py plan OUT ANALYSIS STEM [--performer "..."]   -> OUT/sing/insert_K.{wav,json}
    (render: one LongCat run per insert on a Spark, in docker/longcat; see bin/sing_inserts.sh)
    python3 pipeline/sing_inserts.py splice OUT VIDEO.mp4 OUT.mp4                -> the video with the inserts cut in
"""
from __future__ import annotations

import argparse, json, math, os, subprocess, sys


def runs(shots):
    out, cur = [], None
    for s in shots:
        if s.get("cast") == "sing":
            if cur and abs(cur["end"] - s["start"]) < 0.05:
                cur["end"] = s["end"]
                cur["shots"].append(s["i"])
            else:
                cur = {"start": s["start"], "end": s["end"], "shots": [s["i"]], "prompt": s["prompt"]}
                out.append(cur)
    return out


def segments(dur, first=93 / 25, more=80 / 25):           # LongCat: 93 frames, then 80 new per segment, at 25 fps
    return 1 + max(0, math.ceil((dur - first) / more))


def plan(a):
    sh = json.load(open(os.path.join(a.out, "shots.json")))["shots"]
    scp = os.path.join(a.out, "scenes.json")
    if os.path.exists(scp):                           # scenes.py: a scene's lipsync flag decides, not just its cue
        on = set()
        for sc in json.load(open(scp))["scenes"]:
            if sc.get("lipsync"):
                on.update(range(sc["bars"][0], sc["bars"][1] + 1))
        sh = [dict(s, cast="sing" if s["i"] in on else ("scene" if s.get("cast") == "sing" else s.get("cast"))) for s in sh]
    d = os.path.join(a.out, "sing")
    os.makedirs(d, exist_ok=True)
    vocals = os.path.join(a.analysis, f"{a.stem}-vocals.mp3")
    rs = runs(sh)
    for k, r in enumerate(rs):
        dur = r["end"] - r["start"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{r['start']:.3f}", "-t", f"{dur:.3f}", "-i", vocals,
                        "-ac", "1", "-ar", "16000", os.path.join(d, f"insert_{k}.wav")], check=True)
        subprocess.run(["cp", os.path.join(a.out, "keys", f"shot_{r['shots'][0]:03d}.png"), os.path.join(d, f"insert_{k}.png")],
                       check=True)
        scene = r["prompt"].split(". ")[0]
        prompt = f"{a.performer} sings with feeling, lips forming the words, small natural head movements. {scene}."
        json.dump({"prompt": prompt, "cond_image": f"/work/insert_{k}.png", "cond_audio": {"person1": f"/work/insert_{k}.wav"}},
                  open(os.path.join(d, f"insert_{k}.json"), "w"), indent=1)                  # LongCat
        json.dump({"prompt": prompt, "cond_video": f"/work/insert_{k}.png", "cond_audio": {"person1": f"/work/insert_{k}.wav"}},
                  open(os.path.join(d, f"it_insert_{k}.json"), "w"), indent=1)               # InfiniteTalk
        r.update(k=k, duration=round(dur, 3), segments=segments(dur))
    json.dump(rs, open(os.path.join(d, "inserts.json"), "w"), indent=1)
    for r in rs:
        print(f"insert {r['k']}: {r['start']:.2f}-{r['end']:.2f} s ({r['duration']:.1f} s, {r['segments']} segments), "
              f"shots {r['shots'][0]}-{r['shots'][-1]}")


def clip_for(d, r, engine="infinitetalk"):
    if engine == "infinitetalk":
        return os.path.join(d, f"it_out_{r['k']}.mp4")
    n = r["segments"]
    return os.path.join(d, f"out_{r['k']}", "ai2v_demo_1.mp4" if n == 1 else f"video_continue_{n}.mp4")


def splice(a):
    d = os.path.join(a.out, "sing")
    rs = [r for r in json.load(open(os.path.join(d, "inserts.json"))) if os.path.exists(clip_for(d, r, a.engine))]
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height,r_frame_rate",
                            "-of", "csv=p=0", a.video], capture_output=True, text=True).stdout.strip().split(",")
    W, H, fps = probe[0], probe[1], probe[2]
    inputs, parts, t, n = ["-i", a.video], [], 0.0, 0
    for r in rs:
        inputs += ["-i", clip_for(d, r, a.engine)]
    for j, r in enumerate(rs, 1):
        parts.append(f"[0:v]trim={t:.3f}:{r['start']:.3f},setpts=PTS-STARTPTS[p{n}]"); n += 1
        parts.append(f"[{j}:v]fps={fps},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,trim=duration={r['end'] - r['start']:.3f},"
                     f"setpts=PTS-STARTPTS[p{n}]"); n += 1
        t = r["end"]
    parts.append(f"[0:v]trim=start={t:.3f},setpts=PTS-STARTPTS[p{n}]"); n += 1
    graph = ";".join(parts) + ";" + "".join(f"[p{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[v]", "-map", "0:a",
                    "-c:v", "libx264", "-crf", "18", "-c:a", "copy", a.dst], check=True)
    print(f"wrote {a.dst}: {len(rs)} sung inserts cut in")


ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["plan", "splice"])
ap.add_argument("out")
ap.add_argument("x")
ap.add_argument("y", nargs="?")
ap.add_argument("--performer", default="A young woman with long wavy blond hair")
ap.add_argument("--engine", default="infinitetalk", choices=["infinitetalk", "longcat"])
a = ap.parse_args()
if a.step == "plan":
    a.analysis, a.stem = a.x, a.y
    plan(a)
else:
    a.video, a.dst = a.x, a.y
    splice(a)
