"""poptok/lean step 2 (D38 test 3): the directors' storyboard. From the brief, the song's length and each sung line's
time, the directors decide the keyframes: each one's time span, the line(s) it covers (or none: an instrumental
stretch), what happens (the beat), and whether Amber is in it (none / seen / sings). They decide the scene / line mix:
one line per keyframe, several lines in one, or several keyframes in one line. Nothing about style or camera lenses:
the visualiser (step 3) turns each beat into a picture.

    python3 lean/directors.py PIECE [--llm http://127.0.0.1:8091]   (via bin/workshop.sh, which starts the LLM:
    WORKSHOP_PY=lean/directors.py bin/workshop.sh PIECE)
Writes PIECE/lean/storyboard.json and storyboard.md. The storyboard is checked (spans from 0 to the song's end, no
gaps or overlaps, 2-10 s each, "sings" only over a sung line); a failing one is sent back once with the problems named.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from workshop import as_json, llm, read_env

PROMPT = """You are the directors of a music video. Read the artist's brief, the lyrics with the time each line is
sung, and plan the video as a storyboard of KEYFRAMES: each keyframe is one picture that the video model animates for
its time span.

THE ARTIST'S BRIEF (follow it; what it says must happen, happens):
---
{brief}
---

THE SONG: {dur:.1f} seconds long. The lines and when they are sung (seconds):
{timeline}

THE PERFORMER: {performer}. She can appear in the video (a photo of her is used), and she can sing a line on camera
(lip-synced, her face close enough to see her mouth). You decide where she appears and where she sings; where she
sings, it should mean the most.

Decide the keyframes: how many, and how they map to the lines. One keyframe can cover one line, several lines, or part
of a line; instrumental stretches need keyframes too. Each keyframe 2-10 seconds. Together they cover the whole song
from 0 to {dur:.1f} with no gaps and no overlaps, in order.

For each keyframe write the BEAT: 1-2 sentences of what we see happen, concrete: who, where, doing what, and how the
camera sees it (close, wide, from above ...). Do not describe a drawing style, medium or colours (that is decided
elsewhere). Do not put lyrics as words on screen.

Return ONLY JSON:
{{"story": "3-5 sentences: the story the video tells, start to end",
  "performer_look": "12-25 words: how she looks in every keyframe she is in (clothes, hair), reused word for word",
  "keyframes": [{{"start": 0.0, "end": 4.5, "lines": [line numbers covered, or [] for instrumental],
                 "beat": "...", "amber": "none" or "seen" or "sings"}}]}}"""


def check(sb, dur, segs):
    """The problems with a storyboard, as sentences for the directors (empty: it passes)."""
    k, p = sb.get("keyframes") or [], []
    if not k:
        return ["there are no keyframes"]
    if abs(k[0]["start"]) > 0.05:
        p.append(f"the first keyframe starts at {k[0]['start']}, not 0")
    if abs(k[-1]["end"] - dur) > 0.5:
        p.append(f"the last keyframe ends at {k[-1]['end']}, not at the song's end {dur:.1f}")
    for i, (a, b) in enumerate(zip(k, k[1:])):
        if abs(b["start"] - a["end"]) > 0.05:
            p.append(f"keyframes {i + 1} and {i + 2}: {'a gap' if b['start'] > a['end'] else 'an overlap'} between "
                     f"{a['end']} and {b['start']}")
    for i, x in enumerate(k):
        d = x["end"] - x["start"]
        if not 2 - 0.05 <= d <= 10 + 0.05:
            p.append(f"keyframe {i + 1} is {d:.1f} s long (2-10 s)")
        if x.get("amber") not in ("none", "seen", "sings"):
            p.append(f"keyframe {i + 1}: amber must be none, seen or sings")
        if x.get("amber") == "sings" and not any(s["start"] < x["end"] and s["end"] > x["start"] for s in segs):
            p.append(f"keyframe {i + 1} has her singing where no line is sung")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    a = ap.parse_args()
    P = a.piece.rstrip("/"); e = read_env(f"{P}/piece.env")
    stem = os.path.splitext(os.path.basename(e["SONG"]))[0]
    segs = json.load(open(f"{P}/analysis/{stem}-true-segments.json"))["segments"]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                f"{P}/{e['SONG']}"], capture_output=True, text=True).stdout)
    timeline = "\n".join(f"  line {i + 1}: {s['start']:.1f}-{s['end']:.1f}  {s['text']}" for i, s in enumerate(segs))
    prompt = PROMPT.format(brief=open(f"{P}/inputs/brief.txt").read().strip(), dur=dur, timeline=timeline,
                           performer=e.get("PERFORMER") or "the singer")
    sb, probs = {}, ["no answer"]
    for attempt in range(2):
        sb = as_json(llm(prompt, a.llm, max_tokens=6000, temperature=0.7))
        probs = check(sb, dur, segs)
        print(f"directors: attempt {attempt + 1}: {len(sb.get('keyframes') or [])} keyframes, "
              f"{len(probs)} problems" + "".join(f"\n  - {x}" for x in probs), flush=True)
        if not probs:
            break
        prompt += ("\n\nYour previous storyboard:\n" + json.dumps(sb) + "\nIt has these problems; fix them and return "
                   "the whole storyboard again:\n" + "\n".join(f"- {x}" for x in probs))
    out = f"{P}/lean"; os.makedirs(out, exist_ok=True)
    sb["checks"] = probs
    json.dump(sb, open(f"{out}/storyboard.json", "w"), indent=1, ensure_ascii=False)
    md = [f"# Storyboard: {os.path.basename(P)} (poptok/lean directors)", "", f"**Story:** {sb.get('story', '')}", "",
          f"**Her look:** {sb.get('performer_look', '')}", "",
          "**Checks:** " + ("all passed (spans 0 to the end, no gaps or overlaps, 2-10 s, sings only over a sung line)"
                            if not probs else "FAILED: " + "; ".join(probs)), "",
          "| No. | Time | Lines | Amber | Beat |", "|---|---|---|---|---|"]
    for i, x in enumerate(sb.get("keyframes") or []):
        ls = " / ".join(segs[n - 1]["text"] for n in x.get("lines", []) if 0 < n <= len(segs)) or "(instrumental)"
        md.append(f"| {i + 1} | {x['start']:.1f}-{x['end']:.1f} | {ls} | {x.get('amber')} | {x.get('beat', '')} |")
    open(f"{out}/storyboard.md", "w").write("\n".join(md) + "\n")
    print(f"directors: wrote {out}/storyboard.md")


if __name__ == "__main__":
    main()
