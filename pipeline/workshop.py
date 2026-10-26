"""Directors' workshop: from lyrics (and the song's analysis) to a visual treatment and a shot-by-shot storyboard, by
simulating a design workshop between three music-video directors with the local LLM. The hand-run version is
docs/workshops/slippery-road.md (Slippery Road v3); this makes it a pipeline step.

    python3 pipeline/workshop.py PIECE [--stage story|storyboard|both] [--llm URL] [--force] [--previous OLD.md]
Two stages, two pipeline steps (pipeline/steps.ini), like a film's treatment and its step outline:
  story       the conversation, then the TREATMENT: the whole story in prose (logline, who she is and what she wants,
              the arc's movements against the song's sections, opening and closing image) -> story.md, plus the look
              (visual-look.workshop.txt), universe.workshop.txt and workshop.md. People may edit story.md.
  storyboard  from story.md (the authority; the conversation as background): 4-8 scenes, each staging part of the
              arc, and a shot per line -> lyrics-cued.txt, story-scenes.json (each scene's arc movement, beats and
              times, read by prompts_per_line.py and scenes.py), storyboard.md. Rebuilt whenever story.md changes.
Before 2026-10-25 one pass wrote the scenes straight from the conversation and "visual-treatment.txt" held the look:
nothing stated the arc the scenes add up to, and the scene ids were lost after the storyboard.
Reads PIECE/piece.env's LYRICS and SONG's analysis (required: the lyric timings, PIECE/analysis/<song>-true-segments.json,
so the directors know when each line is sung and how long the instrumental passages are; ./poptok runs the analysis
first), PIECE/inputs/brief.txt
(optional: anything the user wants: artist, mood, references, review notes from earlier versions) and whether a
performer photo is set (PERFORMER_PHOTO). Writes, in PIECE/inputs/:
  workshop.md            the conversation and the agreed result (for the user to read)
  story.md               the treatment: the story, start to finish (stage story; editable)
  universe.workshop.txt  the universe the directors chose (line 1: a few words; then what every frame is made of)
  visual-look.workshop.txt  their look. Since 2026-10-26 both are only proposals: the universe step
                         (pipeline/universe.py) offers them among its candidates and writes universe.txt (STYLE_WORD +
                         STYLE_BRIEF) and visual-look.txt (LOOK) from the chosen universe
  storyboard.md          the scenes and shots (stage storyboard), story-scenes.json the same scenes with their times
  lyrics-cued.txt        the lyrics annotated line by line ({mode: shot note}; CUES), the sung words unchanged, plus a
                         shot for every slot of the instrumental passages: "(instrumental M:SS.S) {mode: note}"
Instrumental passages of 6 s or more (intro, gaps, outro) get slots of about 6-8 s each, so no still has to carry a long
stretch alone (noble-hunt v1, 2026-10-24: a line followed by a 25 s gap became one 28 s scene, and the video chain
looped back to its still clip after clip). If the piece has an effects config (FX), the times its effects fire are
given to the directors (the effect adds the light; the shots don't need to).
The directors plan a STORY first (who, what she wants, what happens, 4-8 scenes each held in one place over several
lines), and every line is the next moment of its scene, not a new image (Chronophobia director's cut, 2026-10-25, the
user: "real videos tell a story; this comes across as a slide show": every line was a new image, a chain of objects
turning into other objects, with no one doing anything).
LLM passes: (1) the conversation; (2a) the treatment and the look as JSON, checked for a whole story (a logline, 3-6
movements, enough prose, an ending that isn't the opening); (2b) the storyboard from the treatment (scenes, a scene,
mode + shot note per line), checked for shape; (3) a physical-plausibility review of every shot note (things a camera could never see: e.g. "seen from above, her
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
- Video: ONE continuous take (no cuts). Within a scene the take simply goes on: the camera moves, the action
  continues, each still is the next moment in the same place (adjacent stills of one scene make a smooth, long shot).
  Between scenes, a change works when it's a physical EVENT that carries A into B (glass shatters and the shards become the next objects, a gust scatters things that settle as
  the next scene, the camera pushes through an object into the next place, a figure walks through and the room changes
  behind them). A quiet object slowly becoming another quiet object becomes a cross-fade. Scene changes are a
  handful per song, not one per line: a new image on every line, each object turning into the next, plays as a slide
  show (Chronophobia director's cut, 2026-10-25).
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

The lyrics with their timings (M:SS.S from the start of the song; section headers in brackets; a hint in parentheses
after a header is the songwriter's direction for that section (who is there, what the action is, the voices): follow
it; a bracketed line in parentheses, e.g. [(silence)], is a stage direction at that point). Instrumental passages are marked with their
length and the shot slots (G1, G2, ...) that must fill them: each slot is one shot of about 6-8 s. Plan them as part of
the video (what happens between the lines, how it moves the story on), never as one image held for the whole gap:
{lyrics}

Write the workshop as a conversation (names in bold, then what they say), about 1400-1800 words. They should: start
from what the song means (the reading); then agree THE STORY the video tells, because real music videos tell a story
and a video that illustrates each line with a new image is a slide show. Who is in it, what she wants or is going
through, what she DOES (people acting, not objects transforming), what changes from beginning to end (the ending is
not the opening), where the turn comes (often the bridge or the last chorus). Tell it in 4-8 SCENES, each in one place
and lasting a section or several lines (15-45 s), so the camera stays with a situation while it plays out; a chorus may
come back to a place that has changed. The story runs alongside the lyrics, it doesn't spell out each line's words.
Before going scene by scene, one of them tells the whole story back in a paragraph, start to finish (the TREATMENT, as
screenwriters use the word), and they settle its arc: who she is and what she wants, what stands in her way, the
movements of the story against the song's sections, where the turn comes, and how the last image answers the first.
The scenes then stage that story, each one a part of the arc. Every scene has a SHAPE: how it begins, the turn (what happens or changes in it), and how it ends differently from how
it began; every line moves it on, and no line just restates the scene's mood ("the light keeps fading, the colour
softer" on line after line is filler). Anyone the song sings to ("you") is set up before they matter: seen, or a trace
of them, in an earlier scene, not appearing from nowhere in the last one. Their plan will be checked for this: a scene
without a beginning, a turn and an end, or lines that repeat the same moment, is sent back to them.
Then agree the UNIVERSE STYLE: the medium and world the whole
video lives in, true for every frame. Photographic realism is only one option; argue for at least three different
universes (for example oil impasto, stained glass, charcoal on paper, paper cut-out, claymation, cyanotype,
hand-tinted photographs, woodcut) and choose the one that serves the song's meaning, saying what everything in it is
made of, how light behaves, and what it can and can't show. If the brief fixes the look, that is the universe: keep
it and don't debate it. Then agree how CONCRETE or ABSTRACT the video is: literal people and places, stylised, or
symbolic. Decide it for the whole video and again for every shot: the same idea can be shown many ways (a thousand
people as a crowd of faces, as footprints multiplying in snow, as shadows on a wall, as a river of lanterns, as a
pattern of umbrellas from above), and the right level depends on the moment and on what the machine draws well. Then
argue and build on each other; agree on one concept, a small set of visual rules (light,
camera, how changes happen), where the performer appears and sings, and then walk
through the story scene by scene: where it is, what happens in it line by line (each line the next moment of the same
action, the camera moving with it), and how each change to the next scene happens as an event. Every
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

STORY = """From this directors' workshop, extract what they agreed as JSON: the look, and the story as a TREATMENT (as
screenwriters use the word: the whole story told in prose, start to finish, before any scene list).

The workshop:
{conversation}

The lyrics with their timings and sections:
{lyrics}

Return JSON only:
{{"concept": "one or two sentences",
 "universe": {{"name": "the universe style in 2-4 words, e.g. oil impasto", "brief": "50-90 words: what everything in every frame is made of, its surfaces and texture, how light behaves, the palette; written as rules true for every frame (if the brief fixes the look, its words)"}},
 "look": "the visual look for the image model: look, light, palette, camera, how changes happen; 60-120 words, no story. If the brief fixes the look, start with the brief's look words unchanged. Never describe the performer (this text styles every shot, most without her)",
 "rules": ["short visual rules"],
 "abstraction": "one or two sentences: how concrete or abstract the video is overall, and where it moves along that scale",
 "logline": "the story in one sentence: who, what she does, how it ends",
 "character": "who she is and what she wants or is going through, 1-2 sentences",
 "obstacle": "what stands in her way or weighs on her, 1 sentence",
 "arc": [{{"movement": "its role in the story, e.g. setup, pressure, turn, resolution", "sections": "the song sections it spans, e.g. Intro-Verse 1", "what_happens": "1-2 sentences"}}],
 "treatment": "the whole story in present-tense prose, 250-450 words, start to finish: what she does, what changes, where the turn falls against the song, how it ends; concrete and visual, places and actions named; no camera jargon, no lyrics quoted",
 "opening_image": "the first image, one sentence",
 "closing_image": "the last image, one sentence, answering the first"}}
3-5 movements in song order covering the whole song; the turn falls where the workshop put it (often the bridge or the
last chorus); the closing image is not the opening image.{problem}"""

RESULT = """Break this music video's TREATMENT (its story, below: the authority) into scenes, and a shot for every line.
The directors' workshop that produced it is given for its ideas about places, shots and where she sings; where it
disagrees with the treatment, follow the treatment.

The treatment:
{story}

The workshop:
{conversation}

The lyrics with line numbers and instrumental slots (only sung lines are numbered; slots are G1, G2, ...; headers
are shown for context):
{numbered}

Performer photo available: {has_photo}. Allowed modes: {modes}.

Return JSON only:
{{"story": {{"scenes": [{{"id": "S1", "movement": "the movement of the treatment's arc it belongs to", "from": "the sentence of the treatment this scene stages, quoted", "place": "where, 5-15 words, the same for the whole scene", "action": "what happens in this scene, 15-40 words", "begins": "the situation at its start", "turn": "what happens or changes in it", "ends": "how it ends, different from its start"}}]}},
 "lines": [{{"n": line number, "scene": "S1", "mode": one of the allowed modes, "level": "concrete" | "stylised" | "abstract", "note": "what we see on this line: the next moment of its scene's action, continuing from the line before (same place, the action moving on, the camera moving with it); one filmable, physically possible shot, 10-30 words, shown at that level"}}]}}
4-8 scenes, in song order, together telling the whole treatment from its opening image to its closing image; each
covers a run of consecutive lines and slots (a section or several lines), never a single line. Every scene has its beginning, turn and end, and its lines go through them in order; each line's note
says what is NEW in that moment (never the previous line's note reworded).{problem}
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
        if HEADER.match(t) or (t.startswith("[") and t.endswith("]")) or re.match(r"^\(.*\)$", t):   # (silence): a direction
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


def shape_problem(res: dict, order: dict) -> str:
    """'' if every scene has a shape (a beginning, a turn, an end) and its lines move on; otherwise what's wrong.
    Filler: a note sharing most of its words with the one before it in the same scene (chronophobia-story,
    2026-10-25: "the colour softer, more faded" on 6 lines running, one scene held 75 s on one pair of hands)."""
    scenes = [x for x in (res.get("story") or {}).get("scenes", []) if isinstance(x, dict)]
    flat = [str(x.get("id", "")) for x in scenes if not all(str(x.get(k, "")).strip() for k in ("begins", "turn", "ends"))]
    if flat:
        return f"scenes {', '.join(flat)} have no beginning, turn and end"
    words = lambda t: set(re.findall(r"[a-z']{4,}", str(t).lower()))
    ls = sorted((x for x in res.get("lines", []) if isinstance(x, dict)), key=lambda x: order.get(str(x.get("n")).strip(), 0))
    filler = [str(b.get("n")) for a, b in zip(ls, ls[1:]) if a.get("scene") == b.get("scene") and words(b.get("note"))
              and len(words(a.get("note")) & words(b.get("note"))) > 0.5 * len(words(b.get("note")))]
    if len(filler) > max(2, len(ls) // 10):
        return f"lines {', '.join(filler[:12])} repeat the moment before them instead of moving the scene on"
    return ""


def flat(v) -> str:
    """A JSON field as one line of text (the LLM sometimes returns a dict of parts instead of a string)."""
    if isinstance(v, dict):
        v = " ".join(str(x) for x in v.values())
    return " ".join(str(v or "").split())


def story_problem(st: dict) -> str:
    """'' if the treatment tells a whole story: a logline, 3-6 movements, enough prose, an ending that isn't the opening."""
    arc = [x for x in st.get("arc", []) if isinstance(x, dict)]
    if not flat(st.get("logline")):
        return "no logline"
    if not 3 <= len(arc) <= 6:
        return f"{len(arc)} movements in the arc (3-5, covering the whole song)"
    if len(flat(st.get("treatment")).split()) < 150:
        return "the treatment is too short to tell the whole story (250-450 words of prose)"
    if flat(st.get("opening_image")).lower() == flat(st.get("closing_image")).lower():
        return "the closing image is the opening image"
    return ""


def story_md(name: str, st: dict) -> str:
    arc = [x for x in st.get("arc", []) if isinstance(x, dict)]
    return "\n".join([
        f"# Treatment: {name}", "",
        "The story the video tells, from the directors' workshop (`pipeline/workshop.py --stage story`; the conversation is "
        "in `workshop.md`). Edit it freely: the storyboard (scenes and a shot per line) is built from this file, and "
        "`./poptok` rebuilds it, and everything after it, when this file changes.", "",
        f"**Logline:** {flat(st.get('logline'))}", "",
        f"**Who:** {flat(st.get('character'))}  ", f"**In her way:** {flat(st.get('obstacle'))}", "",
        "## Treatment", "", flat(st.get("treatment")), "",
        "## Arc", "", *[f"{k}. **{flat(x.get('movement')).capitalize()}** ({flat(x.get('sections'))}): "
                         f"{flat(x.get('what_happens'))}" for k, x in enumerate(arc, 1)], "",
        f"**Opening image:** {flat(st.get('opening_image'))}  ", f"**Closing image:** {flat(st.get('closing_image'))}", ""])


def section(md: str, start: str, end: str) -> str:
    """The text of a markdown file between two headings (all of it if the first is missing)."""
    return md.split(start, 1)[1].split(end, 1)[0].strip() if start in md else md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    ap.add_argument("--stage", choices=["story", "storyboard", "both"], default="both",
                    help="story: the conversation and the treatment (workshop.md, story.md, universe.workshop.txt, visual-look.workshop.txt); "
                    "storyboard: scenes and a shot per line from story.md (lyrics-cued.txt, story-scenes.json, storyboard.md)")
    ap.add_argument("--force", action="store_true", help="redo the story: the existing workshop is archived "
                    "(workshop.prev-*.md) and passed in as the previous workshop, so the directors build on it")
    ap.add_argument("--previous", help="a previous workshop.md to build on (a follow-up workshop)")
    ap.add_argument("--dry-run", action="store_true", help="print the timed lyrics and slots the directors get; no LLM")
    a = ap.parse_args()
    P = a.piece.rstrip("/")
    e = read_env(os.path.join(P, "piece.env"))
    e["SONG"] = e.get("SONG") or "outputs/song.wav"    # no SONG: the one ./poptok generated (MUSIC_STYLE)
    e["LYRICS"] = e.get("LYRICS") or "inputs/lyrics.txt"
    inp = os.path.join(P, "inputs")
    os.makedirs(inp, exist_ok=True)
    out_md, out_look, out_cues, out_uni, out_story, out_board, out_scenes = (os.path.join(inp, n) for n in (
        "workshop.md", "visual-look.workshop.txt", "lyrics-cued.txt", "universe.workshop.txt", "story.md", "storyboard.md",
        "story-scenes.json"))
    do_story, do_board = a.stage in ("story", "both"), a.stage in ("storyboard", "both")
    if do_story and not a.dry_run:
        if os.path.exists(out_story) and not a.force:
            print(f"{out_story} exists (use --force to redo the story)")
            do_story = False
        elif os.path.exists(out_md):                     # keep the previous workshop and build on it
            import time as _t
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
    order, ln = {}, 0                                 # each line's and slot's position in the song
    for k, t, j in items:
        if k == "line":
            ln += 1
        if k != "header":
            order[t[0] if k == "slot" else str(ln)] = len(order)

    if do_story:            # (1) the conversation, (2a) the treatment and the look
        print("workshop: conversation", flush=True)
        prev = ""
        if a.previous:
            txt = open(a.previous).read()
            txt = txt.split("## Plausibility fixes")[0]                       # the conversation and the result
            prev = PREVIOUS.format(text=txt[-24000:])                         # keep within the context window
            print(f"workshop: follow-up to {a.previous}", flush=True)
        conv = llm(CONVERSATION.format(directors=DIRECTORS, song=song_facts(P, e), performer=performer, brief=brief,
                                       capabilities=CAPABILITIES, lyrics=lyrics_txt, previous=prev), a.llm, max_tokens=4000)
        print("workshop: treatment", flush=True)
        st, problem = {}, ""
        for attempt in range(3):
            try:
                st = as_json(llm(STORY.format(conversation=conv, lyrics=lyrics_txt, problem=problem), a.llm,
                                 max_tokens=3000, temperature=0.2))
                why = story_problem(st)
                if not why:
                    break
                print(f"  {why}; retry", flush=True)
                problem = f"\nA previous draft was sent back: {why}. Fix that."
            except (ValueError, TypeError) as ex:
                print(f"  treatment unparsable ({ex}); retry", flush=True)
        open(out_story, "w").write(story_md(os.path.basename(P), st))
        open(out_look, "w").write(flat(st.get("look") or st.get("visual_treatment")) + "\n")
        uni = st.get("universe") or {}
        if isinstance(uni, dict) and uni.get("name"):   # line 1: the style word; then its brief (./poptok: STYLE_WORD/BRIEF)
            open(out_uni, "w").write(flat(uni["name"]) + "\n" + flat(uni.get("brief")) + "\n")
        md = [f"# Directors' workshop: {os.path.basename(P)}", "",
              "Simulated by `pipeline/workshop.py` with the local LLM. The directors are fictional composites standing "
              "for traditions, not real people.", "", "**Directors**", DIRECTORS, "", f"**Song:** {song_facts(P, e)}  ",
              f"**Performer:** {performer}  ", f"**Brief:** {brief}", "", "## The conversation", "", conv, "",
              "## The result", "", f"**Concept:** {flat(st.get('concept'))}", "",
              f"**Story:** {flat(st.get('logline'))} The treatment (the whole story) is `inputs/story.md`; the scenes and "
              "shots built from it are `inputs/storyboard.md`.", "",
              f"**Universe style, the directors' proposal** (`inputs/universe.workshop.txt`; chosen in the universe step): {flat(uni.get('name')) if isinstance(uni, dict) else ''}: "
              f"{flat(uni.get('brief')) if isinstance(uni, dict) else ''}", "",
              f"**Look, the directors' proposal** (`inputs/visual-look.workshop.txt`): {flat(st.get('look'))}", "",
              f"**Concrete vs abstract:** {flat(st.get('abstraction'))}", "",
              "**Rules**", *[f"- {flat(r)}" for r in st.get("rules", [])], ""]
        open(out_md, "w").write("\n".join(md) + "\n")
        print(f"workshop: wrote {out_md}, {out_story}, {out_look}" + (", " + out_uni if os.path.exists(out_uni) else ""))
    if not do_board:
        return

    # (2b) the storyboard, from the treatment: scenes (each a part of its arc) and a shot per line; (3) plausibility
    if not os.path.exists(out_story):
        sys.exit(f"workshop: no {out_story}; run the story stage first (./poptok PIECE --redo story)")
    story_txt = open(out_story).read()
    conv = section(open(out_md).read(), "## The conversation", "## The result") if os.path.exists(out_md) else "(none)"
    logline = (re.search(r"^\*\*Logline:\*\*\s*(.*)$", story_txt, re.M) or [None, ""])[1].strip()
    print("workshop: storyboard", flush=True)
    res, problem = {}, ""
    for attempt in range(3):
        try:
            res = as_json(llm(RESULT.format(story=story_txt, conversation=conv, numbered="\n".join(numbered),
                                            has_photo=has_photo, modes=", ".join(modes), problem=problem), a.llm,
                              max_tokens=6000, temperature=0.2))
            got = {str(x["n"]).strip() for x in res.get("lines", [])}
            notes = [" ".join(str(x.get("note", "")).lower().split()) for x in res.get("lines", [])]
            dup = len(notes) - len(set(notes))       # a note copied onto several lines merges them into one long scene
            scenes = {str(x.get("id", "")).strip() for x in (res.get("story") or {}).get("scenes", [])}
            runs = [str(x.get("scene", "")).strip() for x in sorted(res.get("lines", []), key=lambda x: order.get(
                str(x["n"]).strip(), 0))]
            changes = sum(1 for x, y in zip(runs, runs[1:]) if x != y)
            story_ok = 3 <= len(scenes) <= 10 and set(runs) <= scenes and changes <= 2 * len(scenes)
            why = shape_problem(res, order)
            if got >= set(ids) and dup <= max(2, len(ids) // 20) and story_ok and not why:
                break
            if got >= set(ids) and not story_ok:     # a slide show: no scenes, or lines hopping between them
                why = f"{len(scenes)} scenes and {changes} scene changes (4-8 scenes, each a run of consecutive lines)"
            if got >= set(ids) and why:
                print(f"  {why}; retry", flush=True)
                problem = f"\nA previous draft was sent back: {why}. Fix that."
                continue
            if got >= set(ids):
                print(f"  result repeats {dup} shot notes; retry", flush=True)
                continue
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
        fixes = as_json(llm(REVIEW.format(concept=logline, shots=shots), a.llm, max_tokens=1500, temperature=0.1)).get("fixes", [])
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
    story = res.get("story") if isinstance(res.get("story"), dict) else {}
    story_scenes = [x for x in story.get("scenes", []) if isinstance(x, dict)]
    places = {str(x.get("id", "")).strip(): " ".join(str(x.get("place", "")).replace("{", "(").replace("}", ")").split())
              for x in story_scenes}
    # write the annotated lyrics: the original sung words, a {mode: note} per line; headers kept as they were
    width = max(len(t) for k, t in lyr if k == "line") + 4
    cued, i, times = [], 0, {}
    for k, t, j in items:
        if k == "header":
            cued.append(("" if not cued else "\n") + f"[{t}]")
            continue
        if k == "slot":
            key, times[t[0]], t = t[0], (t[1], t[2]), f"(instrumental {mmss(t[1])})"
        else:
            i += 1
            key = str(i)
            if i - 1 < len(se):
                times[key] = se[i - 1]
        note = lines[key]["note"].replace("{", "(").replace("}", ")").strip()
        place = places.get(str(lines[key].get("scene", "")).strip(), "")
        if note and place and place.lower()[:20] not in note.lower():   # every still of a scene is drawn in its place
            note = f"{place.rstrip('.')}: {note}"
        lvl = str(lines[key].get("level", "")).strip().lower()
        if note and lvl in ("stylised", "stylized", "abstract"):        # how literally to show it (prompts_per_line reads it)
            note = f"({'stylised' if lvl.startswith('styli') else 'abstract'}) {note}"
        cued.append(f"{t:<{width}}{{{lines[key]['mode']}: {note}}}" if note else f"{t:<{width}}{{{lines[key]['mode']}}}")
    open(out_cues, "w").write("\n".join(cued) + "\n")
    # the story scenes with their times, so every later step knows which part of the story a moment belongs to
    # (prompts_per_line.py --story: each shot's place in the arc; scenes.py: a continuing camera move inside a story
    # scene, an event only between story scenes)
    sj = []
    for x in story_scenes:
        sid = str(x.get("id", "")).strip()
        keys = sorted((k for k in ids if str(lines[k].get("scene", "")).strip() == sid), key=lambda k: order.get(k, 0))
        tt = [times[k] for k in keys if k in times]
        sj.append({"id": sid, **{f: flat(x.get(f)) for f in ("movement", "from", "place", "action", "begins", "turn", "ends")},
                   "lines": keys, "start": round(min(t[0] for t in tt), 2) if tt else None,
                   "end": round(max(t[1] for t in tt), 2) if tt else None})
    json.dump({"logline": logline, "scenes": sj}, open(out_scenes, "w"), indent=1, ensure_ascii=False)
    md = [f"# Storyboard: {os.path.basename(P)}", "",
          "Built from the treatment (`inputs/story.md`) by `pipeline/workshop.py --stage storyboard`.", "",
          f"**Story:** {logline}", "", "**Scenes** (`inputs/story-scenes.json`)", "",
          *[f"- {x['id']} [{x['movement']}] ({x['place']}), {mmss(x['start']) if x['start'] is not None else '?'}-"
            f"{mmss(x['end']) if x['end'] is not None else '?'}: {x['action']} *Begins:* {x['begins']} *Turn:* {x['turn']} "
            f"*Ends:* {x['ends']}" + (f" *Stages:* \"{x['from']}\"" if x['from'] else "") for x in sj], "",
          "**Shots** (`inputs/lyrics-cued.txt`)", "", "| # | Line | Scene | Mode | Shot |", "|---|---|---|---|---|"]
    i = 0
    for k, t, j in items:
        if k == "header":
            md.append(f"| | **{t}** | | | |")
        elif k == "slot":
            md.append(f"| {t[0]} | *(instrumental {mmss(t[1])}-{mmss(t[2])})* | {lines[t[0]].get('scene', '')} | {lines[t[0]]['mode']} {lines[t[0]].get('level', '')} | {lines[t[0]]['note']} |")
        else:
            i += 1
            md.append(f"| {i} | {mmss(se[i - 1][0]) if i - 1 < len(se) else ''} {t} | {lines[str(i)].get('scene', '')} | {lines[str(i)]['mode']} {lines[str(i)].get('level', '')} | {lines[str(i)]['note']} |")
    md += ["", "## Plausibility fixes", ""] + ([f"- line {f.get('n')}: {f.get('problem', '')} → {f.get('note', '')}"
                                                for f in fixes] or ["- none needed"])
    open(out_board, "w").write("\n".join(md) + "\n")
    print(f"workshop: wrote {out_cues}, {out_scenes}, {out_board} ({n} lines, {len(slots)} instrumental slots, "
          f"{len(sj)} scenes, {len(fixes)} plausibility fixes)")


if __name__ == "__main__":
    main()
