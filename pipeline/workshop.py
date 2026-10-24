"""Directors' workshop: from lyrics (and the song's analysis) to a visual treatment and a shot-by-shot storyboard, by
simulating a design workshop between three music-video directors with the local LLM. The hand-run version is
docs/workshops/slippery-road.md (Slippery Road v3); this makes it a pipeline step.

    python3 pipeline/workshop.py PIECE [--llm URL] [--force] [--previous OLD_WORKSHOP.md]
Reads PIECE/piece.env's LYRICS (and SONG's analysis if present: duration, sections, tempo), PIECE/inputs/brief.txt
(optional: anything the user wants: artist, mood, references, review notes from earlier versions) and whether a
performer photo is set (PERFORMER_PHOTO). Writes, in PIECE/inputs/:
  workshop.md            the conversation, the result, and the plausibility fixes (for the user to read)
  visual-treatment.txt   the treatment (TREATMENT)
  lyrics-cued.txt        the lyrics annotated line by line ({mode: shot note}; CUES), the sung words unchanged
Three LLM passes: (1) the conversation; (2) the result as JSON (concept, treatment, rules, a mode + shot note per line);
(3) a physical-plausibility review of every shot note (things a camera could never see: e.g. "seen from above, her
reflection the right way up"), whose fixes are applied.

The directors are fictional composites standing for traditions, not real people. The capabilities and limits they're
told about are the pipeline's real ones (keep CAPABILITIES current as the pipeline changes).
"""
from __future__ import annotations

import argparse, json, os, re, shlex, sys, urllib.request

DIRECTORS = """- Mara Okonkwo: the handmade tradition (one physical conceit per song, in-camera tricks, objects with feelings).
- Teo Lindqvist: the observational tradition (long takes, restraint, the uncanny in the ordinary, the camera as a
  witness).
- Iris Varga: the graphic tradition (colour and light as the emotional system, motifs that rhyme across the cut)."""

CAPABILITIES = """What the machine can do (and can't):
- One still per bar (about 2-3 s), drawn by an image model from a one-paragraph prompt; near-duplicate bars merge
  into one scene. Strong on night, rain, light, trains, objects, landscapes and strong styles; weak on precise hands,
  crowds of faces, readable text.
- Video: ONE continuous take that turns each scene into the next (no cuts). A change works when it's a physical EVENT
  that carries A into B (glass shatters and the shards become the next objects, a gust scatters things that settle as
  the next scene, the camera pushes through an object into the next place, a figure walks through and the room changes
  behind them). A quiet object slowly becoming another quiet object becomes a cross-fade.
- The performer (only if a photo is provided) can appear from one photo: her face and hair hold; poses can go wrong.
  Modes per line: none (her world without her), pov (through her eyes, unseen), part (a hand, her back, a reflection,
  never the full face), scene (acting, not singing), sing (lip-synced). Lip-sync costs about 11-28 minutes of compute
  per chorus, so use it for at most two or three passages where singing on camera means something.
- Physically possible only: every body part belongs to a visible body (hands attached to arms, five fingers), no
  duplicate people unless the story says so, reflections only where a real reflection would be (never "seen from above
  with her reflection the right way up"), vehicles on rails/roads, nothing floating unless the idea is explicitly
  surreal. Avoid full-body shots where legs might vanish: medium or close shots, or wide shots with the person small.
- Things that move or appear must come from somewhere in frame (no trains or people from nowhere)."""

CONVERSATION = """Simulate a design workshop between three of the world's best music-video directors, fictional
composites who each stand for a tradition:
{directors}

They are planning the music video for a song. Fixed: the lyrics and the music. Open: the visual treatment, where the
performer appears (if at all) and where she sings, and how every line looks.

The song: {song}
Performer: {performer}
Brief from the artist (follow it): {brief}

{capabilities}

The lyrics (section headers in brackets; notes in parentheses after a header are the songwriter's directions about
voices, follow them):
{lyrics}

Write the workshop as a conversation (names in bold, then what they say), about 1200-1600 words. They should: start
from what the song means (the reading) before style; argue and build on each other; agree on one concept, a small set
of visual rules (look, light, camera, how changes happen), where the performer appears and sings, and then walk
through the song section by section, saying what we see on each line and how each change happens as an event. Every
shot they agree must be physically possible and filmable, as described above. End with a one-paragraph summary of
what they agreed.{previous}"""

PREVIOUS = """

This is a FOLLOW-UP workshop. The same directors met before on this song; their previous workshop (conversation and
agreed result) is below. Since then the artist's brief has changed (the brief above is the new one). Open by
recalling what they agreed and what the new brief changes; then keep what still serves the song and the new brief,
change what it requires, and say explicitly what they keep, change and drop, and why.

--- previous workshop ---
{text}
--- end of previous workshop ---"""

RESULT = """From this directors' workshop, extract the agreed result as JSON.

The workshop:
{conversation}

The lyrics with line numbers (only sung lines are numbered; headers are shown for context):
{numbered}

Performer photo available: {has_photo}. Allowed modes: {modes}.

Return JSON only:
{{"concept": "one or two sentences",
 "treatment": "the visual treatment for the image model: look, light, palette, camera, how changes happen; 60-120 words, no story",
 "rules": ["short visual rules"],
 "lines": [{{"n": line number, "mode": one of the allowed modes, "note": "what we see on this line: one concrete, filmable, physically possible shot, 10-30 words"}}]}}
Every numbered line must appear exactly once in "lines"."""

REVIEW = """You check the shot list of a music video for physical INCOHERENCE: things that break the video's own
world, as opposed to surreal or metaphorical imagery the concept asks for (a glowing furnace in a chest, a corridor of
dust, rain turning to petals are fine: the image model can draw them). Flag only:
- anatomy: extra or detached body parts, hands not attached to arms, a person in two places at once
- impossible camera geometry: a reflection that couldn't exist from that camera angle (e.g. "seen from above, her
  reflection in the puddle the right way up"), a full-body shot of someone whose legs are hidden
- continuity: people, vehicles or objects that appear from nowhere, vehicles off their rails or roads (unless the
  shot is explicitly dreamlike), the camera somewhere it can't be within the same take without a stated move
The concept: {concept}

Shots:
{shots}

List ONLY lines that need a change; leave every coherent line out. Be terse: "problem" at most 15 words, "note" is the
replacement shot in 10-30 words (a description of what we see, never commentary). Return JSON only:
{{"fixes": [{{"n": line number, "problem": "...", "note": "..."}}]}}Empty list if all are fine."""


def read_env(path: str) -> dict:
    """KEY=VALUE pairs from a piece.env (values may be quoted; comments after # outside quotes are ignored)."""
    d = {}
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        try:
            v = shlex.split(v, comments=True)
            d[k.strip()] = v[0] if v else ""
        except ValueError:
            d[k.strip()] = v.strip()
    return d


def llm(prompt: str, url: str, max_tokens: int = 3000, temperature: float = 0.7) -> str:
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def as_json(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {}


HEADER = re.compile(r"^\[?\s*((final|last|pre|post)[- ]?)?(verse|chorus|bridge|outro|intro|pre-chorus|hook|refrain|"
                    r"break|middle 8|interlude)\b[^\]]*\]?\s*$", re.I)


def parse_lyrics(path: str):
    """[(kind, text)] with kind 'header' or 'line'; {...} notes stripped."""
    out = []
    for raw in open(path).read().split("\n"):
        t = re.sub(r"\{[^}]*\}", "", raw).strip()
        if not t:
            continue
        if HEADER.match(t) or (t.startswith("[") and t.endswith("]")):
            out.append(("header", t.strip("[] ")))
        else:
            out.append(("line", t))
    return out


def song_facts(piece: str, e: dict) -> str:
    song = e.get("SONG", "")
    stem = os.path.splitext(os.path.basename(song))[0]
    ana = os.path.join(piece, "analysis")
    facts = []
    seg = os.path.join(ana, f"{stem}-true-segments.json")
    if os.path.exists(seg):
        s = json.load(open(seg))["segments"]
        facts.append(f"{s[-1]['end'] / 60:.1f} minutes; the first sung line at {s[0]['start']:.0f} s")
    beats = os.path.join(ana, "beats.json")
    if os.path.exists(beats):
        b = json.load(open(beats))
        if b.get("tempo"):
            facts.append(f"about {float(b['tempo']):.0f} BPM")
    if not facts and song and os.path.exists(os.path.join(piece, song)):
        try:
            import subprocess
            d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                      os.path.join(piece, song)], capture_output=True, text=True).stdout)
            facts.append(f"{d / 60:.1f} minutes")
        except Exception:
            pass
    hp = os.path.join(ana, "drum_hits.json")
    if os.path.exists(hp):
        runs = json.load(open(hp))["runs"]
        if runs:
            facts.append("quick runs of deep drum hits (tat-tat-tat) at " +
                         ", ".join(f"{r['start']:.0f}-{r['end']:.0f} s" for r in runs))
    return "; ".join(facts) or "(no analysis yet)"


def drum_lines(piece: str, e: dict, lyr) -> set:
    """Indices (into lyr) of sung lines during which a run of deep drum hits falls (analysis/drum_hits.json)."""
    stem = os.path.splitext(os.path.basename(e.get("SONG", "")))[0]
    hp, sp = os.path.join(piece, "analysis", "drum_hits.json"), os.path.join(piece, "analysis", f"{stem}-true-segments.json")
    if not (os.path.exists(hp) and os.path.exists(sp)):
        return set()
    runs, segs = json.load(open(hp))["runs"], json.load(open(sp))["segments"]
    idx = [j for j, (k, _) in enumerate(lyr) if k == "line"]
    out = set()
    for n, s in enumerate(segs[:len(idx)]):
        nxt = segs[n + 1]["start"] if n + 1 < len(segs) else s["end"] + 4
        if any(r["start"] < nxt and r["end"] > s["start"] for r in runs):
            out.add(idx[n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    ap.add_argument("--force", action="store_true", help="redo: the existing workshop is archived (workshop.prev-*.md) "
                    "and passed in as the previous workshop, so the directors build on it")
    ap.add_argument("--previous", help="a previous workshop.md to build on (a follow-up workshop)")
    a = ap.parse_args()
    P = a.piece.rstrip("/")
    e = read_env(os.path.join(P, "piece.env"))
    inp = os.path.join(P, "inputs")
    os.makedirs(inp, exist_ok=True)
    out_md, out_tr, out_cues = (os.path.join(inp, n) for n in ("workshop.md", "visual-treatment.txt", "lyrics-cued.txt"))
    if os.path.exists(out_md):
        if not a.force:
            print(f"{out_md} exists (use --force to redo)")
            return
        import time as _t                                 # keep the previous workshop and build on it
        arch = os.path.join(inp, f"workshop.prev-{_t.strftime('%Y%m%d-%H%M%S')}.md")
        os.rename(out_md, arch)
        a.previous = a.previous or arch
    lyr = parse_lyrics(os.path.join(P, e["LYRICS"]))
    has_photo = bool(e.get("PERFORMER_PHOTO"))
    performer = (e.get("PERFORMER") or "the singer") + (" (a photo is provided: she can appear on screen)" if has_photo
                                                        else " (no photo: she is never seen; the video is her world)")
    modes = ["none", "pov", "part", "scene", "sing"] if has_photo else ["none", "pov"]
    brief_p = os.path.join(inp, "brief.txt")
    brief = open(brief_p).read().strip() if os.path.exists(brief_p) else "(none)"
    drums = drum_lines(P, e, lyr)                     # lines with quick runs of deep drum hits ("tat-tat-tat")
    lyrics_txt = "\n".join(f"[{t}]" if k == "header" else t + ("   (deep drum hits: tat-tat-tat)" if j in drums else "")
                           for j, (k, t) in enumerate(lyr))
    print("workshop: conversation", flush=True)
    prev = ""
    if a.previous:
        txt = open(a.previous).read()
        txt = txt.split("## Plausibility fixes")[0]                       # the conversation and the result
        prev = PREVIOUS.format(text=txt[-24000:])                         # keep within the context window
        print(f"workshop: follow-up to {a.previous}", flush=True)
    conv = llm(CONVERSATION.format(directors=DIRECTORS, song=song_facts(P, e), performer=performer, brief=brief,
                                   capabilities=CAPABILITIES, lyrics=lyrics_txt, previous=prev), a.llm, max_tokens=4000)
    numbered, n = [], 0
    for k, t in lyr:
        if k == "header":
            numbered.append(f"[{t}]")
        else:
            n += 1
            numbered.append(f"{n}. {t}")
    print("workshop: result", flush=True)
    res = {}
    for attempt in range(3):
        try:
            res = as_json(llm(RESULT.format(conversation=conv, numbered="\n".join(numbered), has_photo=has_photo,
                                            modes=", ".join(modes)), a.llm, max_tokens=6000, temperature=0.2))
            got = {int(x["n"]) for x in res.get("lines", [])}
            if got >= set(range(1, n + 1)):
                break
            print(f"  result missing lines {sorted(set(range(1, n + 1)) - got)[:10]}; retry", flush=True)
        except (ValueError, KeyError, TypeError) as ex:
            print(f"  result unparsable ({ex}); retry", flush=True)
    lines = {int(x["n"]): x for x in res.get("lines", [])}
    for i in range(1, n + 1):                         # anything still missing: her world, no note
        lines.setdefault(i, {"n": i, "mode": "none", "note": ""})
        if lines[i].get("mode") not in modes:
            lines[i]["mode"] = "none"
    print("workshop: plausibility review", flush=True)
    shots = "\n".join(f"{i}. [{lines[i]['mode']}] {lines[i]['note']}" for i in range(1, n + 1))
    fixes = []
    try:
        fixes = as_json(llm(REVIEW.format(concept=res.get("concept", ""), shots=shots), a.llm, max_tokens=1500, temperature=0.1)).get("fixes", [])
    except ValueError:
        pass
    kept = []
    for f in fixes:                                   # accept only short, real replacement shots
        try:
            i = int(f.get("n", 0))
        except (TypeError, ValueError):
            continue
        note = " ".join(str(f.get("note", "")).split())
        if i in lines and note and len(note.split()) <= 45 and not re.search(r"no fix|is coherent|likely fine|allowed|"
                                                                           r"correct(ed)? shot:|\bprompt\b", note, re.I):
            lines[i]["note"] = note
            kept.append(dict(f, note=note))
    fixes = kept
    # write the annotated lyrics: the original sung words, a {mode: note} per line; headers kept as they were
    width = max(len(t) for k, t in lyr if k == "line") + 4
    cued, i = [], 0
    for k, t in lyr:
        if k == "header":
            cued.append(("" if not cued else "\n") + f"[{t}]")
        else:
            i += 1
            note = lines[i]["note"].replace("{", "(").replace("}", ")").strip()
            cued.append(f"{t:<{width}}{{{lines[i]['mode']}: {note}}}" if note else f"{t:<{width}}{{{lines[i]['mode']}}}")
    open(out_cues, "w").write("\n".join(cued) + "\n")
    open(out_tr, "w").write(res.get("treatment", "").strip() + "\n")
    md = [f"# Directors' workshop: {os.path.basename(P)}", "",
          "Simulated by `pipeline/workshop.py` with the local LLM. The directors are fictional composites standing for "
          "traditions, not real people.", "", "**Directors**", DIRECTORS, "", f"**Song:** {song_facts(P, e)}  ",
          f"**Performer:** {performer}  ", f"**Brief:** {brief}", "", "## The conversation", "", conv, "",
          "## The result", "", f"**Concept:** {res.get('concept', '')}", "", f"**Treatment** (`inputs/visual-treatment.txt`): "
          f"{res.get('treatment', '')}", "", "**Rules**", *[f"- {r}" for r in res.get("rules", [])], "",
          "**Storyboard** (`inputs/lyrics-cued.txt`)", "", "| # | Line | Mode | Shot |", "|---|---|---|---|"]
    i = 0
    for k, t in lyr:
        if k == "header":
            md.append(f"| | **{t}** | | |")
        else:
            i += 1
            md.append(f"| {i} | {t} | {lines[i]['mode']} | {lines[i]['note']} |")
    md += ["", "## Plausibility fixes", ""] + ([f"- line {f.get('n')}: {f.get('problem', '')} → {f.get('note', '')}"
                                                for f in fixes] or ["- none needed"])
    open(out_md, "w").write("\n".join(md) + "\n")
    print(f"workshop: wrote {out_md}, {out_tr}, {out_cues} ({n} lines, {len(fixes)} plausibility fixes)")


if __name__ == "__main__":
    main()
