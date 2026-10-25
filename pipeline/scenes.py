"""Scene metadata: the scene (a run of bars sharing one keyframe, i.e. one shot of the LTX chain) as the unit you edit.
OUT/scenes.json sits next to the run's shots.json / shots_merged.json and is the place to change a scene: its prompt,
the performer cue, how it moves, how the previous scene turns into it, notes; then `apply` re-renders only what changed.

    python3 pipeline/scenes.py build OUT [--cues CUED_LYRICS --segments TRUE_SEGMENTS.json]   -> OUT/scenes.json
    python3 pipeline/scenes.py frame OUT 1379 [2037 ...]          which scene each video frame is in (24 fps)
    python3 pipeline/scenes.py transitions OUT [--llm URL]        LLM writes transition_in (A->B morph) for each change
    python3 pipeline/scenes.py apply OUT                          edited scenes -> shots files; deletes their stills and
                                                                  the video frames that depend on them (see below)
A scene in scenes.json:
  id, bars [first, last], start/end (s), frames [first, last], section, lyric, cast, cast_note,
  prompt        the keyframe prompt (edit it to redraw the still)
  transition_in how the previous scene becomes this one, for the Transition LoRA (written by `transitions`, editable)
  motion        movement within the scene (camera, subject), for the clips that continue it
  lipsync       true: a sung insert replaces it (sing_inserts.py)
  reroll        true: redraw the still with a new seed, same prompt
  notes, locked (locked scenes are never redrawn)
`apply` compares each scene with the copy saved at the last build/apply (scenes.applied.json). A changed prompt / cast /
reroll deletes that scene's still (keys/shot_<first bar>.png, bumping reroll.json) and the chain frames from the clip
that turns into it up to the next scene change after it (the chain starts each clip from the previous clip's last
frame, but each change ends on its keyframe, so later scenes are unaffected). A changed transition_in or motion deletes
just the clips that use it. Re-run ./poptok to redraw only those.
"""
from __future__ import annotations

import argparse, copy, glob, json, os, re, sys, urllib.request

FPS = 24
EDIT = ("prompt", "cast", "cast_note", "reroll")
MOVE = ("transition_in", "motion")


def runs(shots):
    out = []
    for s in shots:
        if out and s["prompt"] == out[-1]["prompt"]:
            out[-1]["bars"][1] = s["i"]
            out[-1]["end"] = s["end"]
        else:
            out.append({"bars": [s["i"], s["i"]], "start": s["start"], "end": s["end"], "prompt": s["prompt"],
                        "cast": s.get("cast", "none"), "cast_note": s.get("cast_note", "")})
    return out


def clips(out):
    p = os.path.join(out, "ltx_chain", "clips.json")
    return json.load(open(p)) if os.path.exists(p) else None


def build(a):
    plan = json.load(open(os.path.join(a.out, "shots_merged.json")))
    segs = json.load(open(a.segments))["segments"] if a.segments else []
    old = {s["id"]: s for s in json.load(open(os.path.join(a.out, "scenes.json")))["scenes"]} \
        if os.path.exists(os.path.join(a.out, "scenes.json")) else {}
    order = section_map(a.cues) if a.cues else []
    seg_sec, j = {}, 0                                # each lyric segment's section, matched in order
    for x in segs:
        k = next((i for i in range(j, len(order)) if order[i][0] == x["text"]), None)
        if k is not None:
            seg_sec[id(x)], j = order[k][1], k + 1
    scenes = []
    for k, r in enumerate(runs(plan["shots"])):
        sid = f"s{k:03d}"
        lyric = " / ".join(x["text"] for x in segs if x["start"] < r["end"] - 0.1 and x["end"] > r["start"] + 0.1)
        sc = {"id": sid, "bars": r["bars"], "start": round(r["start"], 3), "end": round(r["end"], 3),
              "frames": [int(round(r["start"] * FPS)), int(round(r["end"] * FPS)) - 1],
              "section": next((seg_sec[id(x)] for x in segs if x["start"] < r["end"] - 0.1
                               and x["end"] > r["start"] + 0.1 and id(x) in seg_sec), ""),
              "lyric": lyric, "cast": r["cast"], "cast_note": r["cast_note"], "prompt": r["prompt"],
              "keyframe": f"keys/shot_{r['bars'][0]:03d}.png", "transition_in": "", "motion": "",
              "lipsync": r["cast"] == "sing", "reroll": False, "notes": "", "locked": False}
        for f in ("transition_in", "motion", "notes", "locked", "lipsync"):     # keep what was written before
            if sid in old and old[sid].get("bars") == sc["bars"] and old[sid].get(f) not in (None, ""):
                sc[f] = old[sid][f]
        scenes.append(sc)
    doc = {"piece": a.out, "fps": FPS, "duration": plan["duration"], "style": plan.get("style", ""), "scenes": scenes}
    json.dump(doc, open(os.path.join(a.out, "scenes.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(doc, open(os.path.join(a.out, "scenes.applied.json"), "w"), indent=1, ensure_ascii=False)
    print(f"wrote {a.out}/scenes.json: {len(scenes)} scenes over {len(plan['shots'])} bars")


def section_map(path):
    """[(sung line, section)] in order (a repeated chorus line belongs to each chorus in turn)."""
    m, sec = [], ""
    for raw in open(path).read().split("\n"):
        t = raw.strip()
        h = re.match(r"^\[([^\]:]*)\]", t)
        if h and not re.match(r"^\[(singer|amber|performer)\s*:", t, re.I) and not h.group(1).startswith("("):   # [(silence)]: a direction
            sec = h.group(1).strip()
            continue
        if t and not t.startswith("[") and not t.startswith("(instrumental"):   # workshop gap shots aren't sung
            m.append((re.sub(r"\{[^}]*\}", "", t).strip(), sec))
    return m


def frame(a):
    doc = json.load(open(os.path.join(a.out, "scenes.json")))
    for n in a.frames:
        sc = doc["scenes"]
        k = next((k for k, s in enumerate(sc) if s["frames"][0] <= n <= s["frames"][1]), None)
        if k is None:
            print(f"frame {n}: outside the video")
            continue
        s = sc[k]
        if k + 1 < len(sc) and n >= sc[k + 1]["frames"][0] - round(a.lead * FPS):
            t = sc[k + 1]
            print(f"frame {n} ({n / FPS:.1f} s): the transition {s['id']} -> {t['id']} ({t['section']} | {t['lyric'][:50]})")
            continue
        print(f"frame {n} ({n / FPS:.1f} s): {s['id']} bars {s['bars'][0]}-{s['bars'][1]} [{s['cast']}] "
              f"{s['section']} | {s['lyric'][:60]} | {s['prompt'][:90]}")


TRANSITION = """You write the transition for an AI video model that morphs one shot into the next in a single continuous
take (no cut, no cross-fade, no flash). Describe, in 2-3 sentences, how shot A physically transforms into shot B:
which shapes, lights or objects in A become which ones in B, and how the camera moves while it happens. Concrete and
visual; present tense; describe what IS seen. Under 60 words.
Make the change a physical EVENT that carries A into B: something moves, bursts, sweeps or is pushed through (a gust
scatters petals across the frame and they settle as the new scene; glass shatters and the shards become the new
objects; the camera pushes through an object into the next place; a figure walks through and the room changes
behind them). Name what in A becomes what in B. Never a plain fade or dissolve: a quiet object slowly turning into
another quiet object makes the video model cross-fade (POC 27, the oil drawer -> chair tests, 2026-10-24).

Song line(s) over shot B: {lyric}
Shot A: {a}
Shot B: {b}

Return ONLY the transition description."""


def llm(prompt, url):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0.6, "max_tokens": 400}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return " ".join(json.loads(r.read())["choices"][0]["message"]["content"].split())


def transitions(a):
    p = os.path.join(a.out, "scenes.json")
    doc = json.load(open(p))
    sc = doc["scenes"]
    for k in range(1, len(sc)):
        if sc[k]["transition_in"] and not a.force:
            continue
        sc[k]["transition_in"] = llm(TRANSITION.format(lyric=sc[k]["lyric"] or "(instrumental)", a=sc[k - 1]["prompt"],
                                                       b=sc[k]["prompt"]), a.llm)
        print(f"{sc[k]['id']}: {sc[k]['transition_in'][:110]}", flush=True)
        json.dump(doc, open(p, "w"), indent=1, ensure_ascii=False)


def apply(a):
    p, q = os.path.join(a.out, "scenes.json"), os.path.join(a.out, "scenes.applied.json")
    doc, base = json.load(open(p)), json.load(open(q))
    old = {s["id"]: s for s in base["scenes"]}
    cl = clips(a.out) or []
    rr_p = os.path.join(a.out, "reroll.json")
    rr = json.load(open(rr_p)) if os.path.exists(rr_p) else {}
    plans = {n: json.load(open(os.path.join(a.out, n))) for n in ("shots.json", "shots_merged.json")}
    sc = doc["scenes"]
    drop_frames, n_keys = set(), 0

    def clip_range(lo_bar, hi_bar):                   # chain clips (by bar) from lo_bar up to hi_bar, inclusive
        for c in cl:
            if lo_bar <= c["j"] <= hi_bar:
                drop_frames.update(range(c["a"], c["e"]))

    for k, s in enumerate(sc):
        o = old.get(s["id"], {})
        if s.get("locked"):
            continue
        nxt = sc[k + 1]["bars"][0] if k + 1 < len(sc) else 10 ** 6
        after = sc[k + 2]["bars"][0] - 1 if k + 2 < len(sc) else 10 ** 6
        if any(s.get(f) != o.get(f) for f in EDIT) or s.get("reroll"):
            for name, plan in plans.items():
                for b in range(s["bars"][0], s["bars"][1] + 1):
                    plan["shots"][b]["prompt"] = s["prompt"]
                    plan["shots"][b]["cast"], plan["shots"][b]["cast_note"] = s["cast"], s.get("cast_note", "")
            key = os.path.join(a.out, s["keyframe"])
            if os.path.exists(key):
                os.remove(key)
                n_keys += 1
            if s.get("reroll"):
                rr[str(s["bars"][0])] = rr.get(str(s["bars"][0]), 0) + 1
                s["reroll"] = False
            clip_range(s["bars"][0], after)           # into this scene, through it, into and through the next
        elif s.get("transition_in") != o.get("transition_in"):
            clip_range(s["bars"][0], s["bars"][0])
        if s.get("motion") != o.get("motion"):
            clip_range(s["bars"][0] + 1, nxt - 1)
    for name, plan in plans.items():
        json.dump(plan, open(os.path.join(a.out, name), "w"), indent=1)
    json.dump(rr, open(rr_p, "w"), indent=1)
    gone = 0
    for d in ("ltx_chain", "ltx_chain_beat"):
        for f in drop_frames:
            fp = os.path.join(a.out, d, f"{f:05d}.png")
            if os.path.exists(fp):
                os.remove(fp)
                gone += 1
    if n_keys:
        for m in ("keys/.done",):
            mp = os.path.join(a.out, m)
            if os.path.exists(mp):
                os.remove(mp)
    for f in glob.glob(os.path.join(a.out, "*.mp4")):
        pass                                          # finished videos are kept; ./poptok writes a new one
    json.dump(doc, open(p, "w"), indent=1, ensure_ascii=False)
    json.dump(doc, open(q, "w"), indent=1, ensure_ascii=False)
    print(f"applied: {n_keys} stills to redraw, {len(drop_frames)} video frames to re-render ({gone} files removed)")


ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["build", "frame", "transitions", "apply"])
ap.add_argument("out")
ap.add_argument("frames", nargs="*", type=int)
ap.add_argument("--cues")
ap.add_argument("--segments")
ap.add_argument("--llm", default="http://127.0.0.1:8091")
ap.add_argument("--force", action="store_true", help="transitions: rewrite existing ones too")
ap.add_argument("--lead", type=float, default=0.9, help="frame: the chain's transition lead (ltx_chain.py --lead)")
a = ap.parse_args()
{"build": build, "frame": frame, "transitions": transitions, "apply": apply}[a.step](a)
