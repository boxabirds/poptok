"""Directors' workshop: from lyrics (and the song's analysis) to a visual treatment and a shot-by-shot storyboard, by
simulating a design workshop between three music-video directors with the local LLM. The hand-run version is
docs/workshops/slippery-road.md (Slippery Road v3); this makes it a pipeline step.

    python3 pipeline/workshop.py PIECE [--llm URL] [--force] [--previous OLD_WORKSHOP.md]
Reads PIECE/piece.env's LYRICS and SONG's analysis (required: the lyric timings, PIECE/analysis/<song>-true-segments.json,
so the directors know when each line is sung and how long the instrumental passages are; ./poptok runs the analysis
first), PIECE/inputs/brief.txt
(optional: anything the user wants: artist, mood, references, review notes from earlier versions) and whether a
performer photo is set (PERFORMER_PHOTO). Writes, in PIECE/inputs/:
  workshop.md            the conversation, the result, and the plausibility fixes (for the user to read)
  visual-treatment.txt   the treatment (TREATMENT)
  lyrics-cued.txt        the lyrics annotated line by line ({mode: shot note}; CUES), the sung words unchanged, plus a
                         shot for every slot of the instrumental passages: "(instrumental M:SS.S) {mode: note}"
Instrumental passages of 6 s or more (intro, gaps, outro) get slots of about 6-8 s each, so no still has to carry a long
stretch alone (noble-hunt v1, 2026-10-24: a line followed by a 25 s gap became one 28 s scene, and the video chain
looped back to its still clip after clip). If the piece has an effects config (FX), the times its effects fire are
given to the directors (the effect adds the light; the shots don't need to).
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
- Things that move or appear must come from somewhere in frame (no trains or people from nowhere).
- Restraint with spectacle: explosions, fire and flashes are strong images; a few in a whole video at most, where the
  brief or the song asks (noble-hunt v1, 2026-10-24: explosions in 22 of 33 scenes, "a Michael Bay heaven").
- One performer: when the lyrics or the brief give her a role (a soldier, a traveller), that figure IS her, never a
  second person beside or behind her."""

CONVERSATION = """Simulate a design workshop between three of the world's best music-video directors, fictional
composites who each stand for a tradition:
{directors}

They are planning the music video for a song. Fixed: the lyrics and the music. Open: the visual treatment, where the
performer appears (if at all) and where she sings, and how every line looks.

The song: {song}
Performer: {performer}
Brief from the artist (follow it): {brief}

{capabilities}

The lyrics with their timings (M:SS.S from the start of the song; section headers in brackets; notes in parentheses
after a header are the songwriter's directions about voices, follow them). Instrumental passages are marked with their
length and the shot slots (G1, G2, ...) that must fill them: each slot is one shot of about 6-8 s. Plan them as part of
the video (what happens between the lines, how it moves the story on), never as one image held for the whole gap:
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

The lyrics with line numbers and instrumental slots (only sung lines are numbered; slots are G1, G2, ...; headers
are shown for context):
{numbered}

Performer photo available: {has_photo}. Allowed modes: {modes}.

Return JSON only:
{{"concept": "one or two sentences",
 "treatment": "the visual treatment for the image model: look, light, palette, camera, how changes happen; 60-120 words, no story. If the brief fixes the look, start with the brief's look words unchanged. Never describe the performer (this text styles every shot, most without her)",
 "rules": ["short visual rules"],
 "lines": [{{"n": line number, "mode": one of the allowed modes, "note": "what we see on this line: one concrete, filmable, physically possible shot, 10-30 words"}}]}}
Every numbered line and every slot (G1, G2, ...) must appear exactly once in "lines" ("n" is the line number or the
slot id, e.g. "G3"); slots are instrumental, so their mode is never "sing". Mode "sing" only where the note shows her face singing on
camera, and only in the two or three passages the directors chose for singing; a shot of an object, a room, her back,
a silhouette or her hands is none, part or scene, even inside a sung chorus (Chronophobia workshop 2026-10-24: 26 of 45 lines
were "sing", most of them not shots of her singing)."""

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
        facts.append(f"{mmss(song_duration(piece, e))} long; the first sung line at {mmss(s[0]['start'])}, the last ends "
                     f"at {mmss(s[-1]['end'])}")
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
    fxm = fx_moments(piece, e)
    if fxm:                 # the effects step adds these; listing every drum run made the directors draw explosions
        facts.append(f"{len(fxm)} bursts of light are added by the effects step at " + ", ".join(mmss(t) for t in fxm)
                     + " (the shots there need no explosion of their own; elsewhere none is needed)")
    hp = os.path.join(ana, "drum_hits.json")
    if os.path.exists(hp) and not fxm:
        runs = json.load(open(hp))["runs"]
        if runs:
            facts.append("quick runs of deep drum hits (tat-tat-tat) at " +
                         ", ".join(f"{r['start']:.0f}-{r['end']:.0f} s" for r in runs))
    return "; ".join(facts) or "(no analysis yet)"


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def song_duration(piece: str, e: dict) -> float:
    import subprocess
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 os.path.join(piece, e["SONG"])], capture_output=True, text=True).stdout)


def timings(piece: str, e: dict, lyr):
    """Each sung line's (start, end) from the analysis, in order, and the instrumental slots: [(id, start, end)] for
    every gap of 6 s or more (intro, between lines, outro), cut into shots of about 6-8 s."""
    stem = os.path.splitext(os.path.basename(e.get("SONG", "")))[0]
    sp = os.path.join(piece, "analysis", f"{stem}-true-segments.json")
    if not os.path.exists(sp):
        sys.exit(f"workshop: no lyric timings ({sp}); run the analysis first (./poptok does it before the workshop)")
    segs = json.load(open(sp))["segments"]
    n_lines = sum(k == "line" for k, _ in lyr)
    if len(segs) != n_lines:
        print(f"workshop: {len(segs)} timed segments for {n_lines} lyric lines; timings matched in order", flush=True)
    se = [(float(s["start"]), float(s["end"])) for s in segs[:n_lines]]
    dur = song_duration(piece, e)
    gaps = [(0.0, se[0][0])] + [(se[i][1], se[i + 1][0]) for i in range(len(se) - 1)] + [(se[-1][1], dur)]
    slots, g = [], 0
    for a, b in gaps:
        if b - a < 6:
            continue
        k = max(1, round((b - a) / 7))
        for j in range(k):
            g += 1
            slots.append((f"G{g}", a + j * (b - a) / k, a + (j + 1) * (b - a) / k))
    return se, slots


def fx_moments(piece: str, e: dict) -> list:
    """The times the piece's effects fire (one per cluster of events within 1 s), from its FX config."""
    if not e.get("FX"):
        return []
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fx.events import select
    out = []
    for cfg in json.load(open(os.path.join(piece, e["FX"]))):
        for t, _ in select(cfg["events"], os.path.join(piece, "analysis")):
            if not out or t - out[-1] > 1.0:
                out.append(t)
    return out


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
    ap.add_argument("--dry-run", action="store_true", help="print the timed lyrics and slots the directors get; no LLM")
    a = ap.parse_args()
    P = a.piece.rstrip("/")
    e = read_env(os.path.join(P, "piece.env"))
    inp = os.path.join(P, "inputs")
    os.makedirs(inp, exist_ok=True)
    out_md, out_tr, out_cues = (os.path.join(inp, n) for n in ("workshop.md", "visual-treatment.txt", "lyrics-cued.txt"))
    if os.path.exists(out_md) and not a.dry_run:
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
    drums = set() if e.get("FX") else drum_lines(P, e, lyr)   # lines with quick runs of deep drum hits ("tat-tat-tat")
    se, slots = timings(P, e, lyr)
    items, li = [], 0          # the lyrics in order, with the instrumental slots where they fall
    pend = list(slots)
    for j, (k, t) in enumerate(lyr):
        st = se[li][0] if li < len(se) else None
        if k == "header":                              # a gap before a section goes before its header
            while pend and st is not None and pend[0][1] < st:
                items.append(("slot", pend.pop(0), j))
            items.append(("header", t, j))
            continue
        while pend and st is not None and pend[0][1] < st:
            items.append(("slot", pend.pop(0), j))
        items.append(("line", t, j))
        li += 1
    items += [("slot", x, None) for x in pend]

    def gap_label(i):          # "(instrumental 2:24.6-2:46.7, 22 s: G5 G6 G7)" before the first slot of a gap
        _, (gid, a0, _b), _j = items[i]
        if i > 0 and items[i - 1][0] == "slot":
            return None
        run = []
        for it in items[i:]:
            if it[0] != "slot":
                break
            run.append(it[1])
        return f"(instrumental {mmss(run[0][1])}-{mmss(run[-1][2])}, {run[-1][2] - run[0][1]:.0f} s: slots " + \
            ", ".join(f"{x[0]} at {mmss(x[1])}" for x in run) + ")"
    rows, li = [], 0
    for i, (k, t, j) in enumerate(items):
        if k == "header":
            rows.append(f"[{t}]")
        elif k == "slot":
            lab = gap_label(i)
            if lab:
                rows.append(lab)
        else:
            a0, b0 = se[li] if li < len(se) else (0, 0)
            rows.append(f"{mmss(a0)}-{mmss(b0)}  {t}" + ("   (deep drum hits: tat-tat-tat)" if j in drums else ""))
            li += 1
    lyrics_txt = "\n".join(rows)
    if a.dry_run:
        print(f"song: {song_facts(P, e)}\n\n{lyrics_txt}")
        return
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
    for k, t, j in items:
        if k == "header":
            numbered.append(f"[{t}]")
        elif k == "slot":
            numbered.append(f"{t[0]}. (instrumental, {mmss(t[1])}-{mmss(t[2])})")
        else:
            n += 1
            numbered.append(f"{n}. {t}")
    ids = [str(i) for i in range(1, n + 1)] + [x[0] for x in slots]
    print("workshop: result", flush=True)
    res = {}
    for attempt in range(3):
        try:
            res = as_json(llm(RESULT.format(conversation=conv, numbered="\n".join(numbered), has_photo=has_photo,
                                            modes=", ".join(modes)), a.llm, max_tokens=6000, temperature=0.2))
            got = {str(x["n"]).strip() for x in res.get("lines", [])}
            if got >= set(ids):
                break
            print(f"  result missing {[i for i in ids if i not in got][:10]}; retry", flush=True)
        except (ValueError, KeyError, TypeError) as ex:
            print(f"  result unparsable ({ex}); retry", flush=True)
    lines = {str(x["n"]).strip(): x for x in res.get("lines", [])}
    for i in ids:                                     # anything still missing: her world, no note
        lines.setdefault(i, {"n": i, "mode": "none", "note": ""})
        if lines[i].get("mode") not in modes or (i.startswith("G") and lines[i].get("mode") == "sing"):
            lines[i]["mode"] = "none" if lines[i].get("mode") not in modes else "scene"
    print("workshop: plausibility review", flush=True)
    shots = "\n".join(f"{i}. [{lines[i]['mode']}] {lines[i]['note']}" for i in ids)
    fixes = []
    try:
        fixes = as_json(llm(REVIEW.format(concept=res.get("concept", ""), shots=shots), a.llm, max_tokens=1500, temperature=0.1)).get("fixes", [])
    except ValueError:
        pass
    kept = []
    for f in fixes:                                   # accept only short, real replacement shots
        i = str(f.get("n", "")).strip()
        note = " ".join(str(f.get("note", "")).split())
        if i in lines and note and len(note.split()) <= 45 and not re.search(r"no fix|is coherent|likely fine|allowed|"
                                                                           r"correct(ed)? shot:|\bprompt\b", note, re.I):
            lines[i]["note"] = note
            kept.append(dict(f, note=note))
    fixes = kept
    # write the annotated lyrics: the original sung words, a {mode: note} per line; headers kept as they were
    width = max(len(t) for k, t in lyr if k == "line") + 4
    cued, i = [], 0
    for k, t, j in items:
        if k == "header":
            cued.append(("" if not cued else "\n") + f"[{t}]")
            continue
        if k == "slot":
            key, t = t[0], f"(instrumental {mmss(t[1])})"
        else:
            i += 1
            key = str(i)
        note = lines[key]["note"].replace("{", "(").replace("}", ")").strip()
        cued.append(f"{t:<{width}}{{{lines[key]['mode']}: {note}}}" if note else f"{t:<{width}}{{{lines[key]['mode']}}}")
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
    for k, t, j in items:
        if k == "header":
            md.append(f"| | **{t}** | | |")
        elif k == "slot":
            md.append(f"| {t[0]} | *(instrumental {mmss(t[1])}-{mmss(t[2])})* | {lines[t[0]]['mode']} | {lines[t[0]]['note']} |")
        else:
            i += 1
            md.append(f"| {i} | {mmss(se[i - 1][0]) if i - 1 < len(se) else ''} {t} | {lines[str(i)]['mode']} | {lines[str(i)]['note']} |")
    md += ["", "## Plausibility fixes", ""] + ([f"- line {f.get('n')}: {f.get('problem', '')} → {f.get('note', '')}"
                                                for f in fixes] or ["- none needed"])
    open(out_md, "w").write("\n".join(md) + "\n")
    print(f"workshop: wrote {out_md}, {out_tr}, {out_cues} ({n} lines, {len(slots)} instrumental slots, "
          f"{len(fixes)} plausibility fixes)")


if __name__ == "__main__":
    main()
