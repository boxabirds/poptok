"""Actor cues: decide, line by line, when and how the performer appears, with the LOCAL LLM, and write them into the
lyrics as annotations (see docs/pocs/22-amber-investigation.md and 23-actor-cues.md).

    python3 pipeline/cast_cues.py LYRICS --treatment TREATMENT --out ANNOTATED [--performer "..."] [--llm URL]
    python3 pipeline/cast_cues.py --score REFERENCE CANDIDATE        (line-by-line agreement, no LLM)

Modes: none (the lyric's world without her), pov (through her eyes, unseen), part (a hand, her back, a reflection;
never the full face), scene (acting in the lyric's world, not singing), sing (lip-synced performance).
Annotation format: a section header `[Verse 1] {amber: pov}` sets the default; a line ending `{part: a note}` or
`{none}` overrides it. Existing `{...}` notes in LYRICS are ignored; other `[...]` lines (e.g. `[Singer: ...]`) are the
user's directions: kept as they are, and the LLM is told to follow them. The sung words are never changed: the LLM
returns line numbers and cues, not text, and the result is checked against the input.
"""
from __future__ import annotations

import argparse, json, re, sys, urllib.request

MODES = ("none", "pov", "part", "scene", "sing")
NOTE = re.compile(r"\{[^}]*\}")
SECTION = re.compile(r"^((final|last|pre|post)[- ]?)?(verse|chorus|bridge|outro|intro|pre-chorus|hook|refrain|break|"
                     r"middle 8|interlude)\b[\w\s]*$", re.I)

PROMPT = """You are the director of a music video. The song is sung by one performer, {performer}. Decide, line by line,
when and how she appears on screen. She should NOT be in every shot: use her where it serves the song's story.

Modes:
- none: the lyric's world without her (objects, places, other people)
- pov: seen through her eyes; she is unseen, other people may react to the camera as if to her
- part: part of her only (a hand, her back, a silhouette, a reflection); never her full face
- scene: she acts in the lyric's world (walking, sitting, reacting), not singing
- sing: she sings the line on camera (lip-synced performance)

Rules:
- Follow the song's story. Only if the lyrics deliberately hold back who the narrator is and reveal it later, keep her
  face hidden (pov, part, none) until that line and make it a clear "scene" with a note saying REVEAL. Otherwise she
  is the singer of the song and should be seen.
- Observing lines (describing other people or places) are usually pov or none. Lines where the narrator acts are
  scene. Emotional peaks and choruses are good for sing, but not every chorus line needs her.
- Prefer callbacks: if an image returns (a hand, a place), say so in the note.
- Her full face (scene or sing) should be on screen for about {face_share} of the lines; the rest none, pov or part.
- The user's own directions, the lines in [square brackets] other than section names, must be followed exactly; they
  describe the section that follows them.
- Notes are short staging directions (under 15 words) for the image and video models, in the visual treatment's world.

Visual treatment: {treatment}

Lyrics (numbered; S1, S2 ... are the songwriter's sections if given, otherwise blank lines separate sections;
other [bracketed] lines are the user's directions, not sung):
{numbered}

Reply with JSON only:
{{"sections": [{{"first_line": <number of the section's first sung line>, "name": "Verse 1", "mode": "<mode>",
   "note": "<optional staging for the whole section>"}}],   (or {{"section": "S1", ...}} when sections are given)
  "lines": [{{"n": <line number>, "mode": "<mode>", "note": "<staging>"}}]}}
{sections_rule} In "lines", list only lines whose mode differs from their section's mode or that need a note."""


def parse(path):
    """-> items: ('sung', text) | ('dir', text) | ('blank', '') ; section names in the input are dropped."""
    items = []
    for raw in open(path).read().split("\n"):
        s = NOTE.sub("", raw).rstrip()
        t = s.strip()
        if not t:
            items.append(("blank", ""))
        elif re.match(r"^\[.*\]$", t):
            inner = t[1:-1].strip()
            if SECTION.match(inner):
                items.append(("sec", inner))                      # the songwriter's section name: kept
                continue
            items.append(("dir", t))
        else:
            items.append(("sung", s.strip()))
    return items


def complete(llm, prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 4096,
                       "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request(f"{llm}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def render(items, cues):
    sung = [k for k, (kind, _) in enumerate(items) if kind == "sung"]
    first = {s["first_line"]: s for s in cues["sections"]}
    per = {l["n"]: l for l in cues.get("lines", [])}
    out = []
    for n, k in enumerate(sung, 1):
        if n in first:
            s = first[n]
            while out and out[-1] == "":
                out.pop()
            # the user's directions just above this section stay just above its header
            j = len(out)
            while j > 0 and out[j - 1].startswith("["):
                j -= 1
            note = f": {s['note']}" if s.get("note") else ""
            if j > 0 and out[j - 1] != "":
                out.insert(j, "")
            out.append(f"[{s['name']}] {{amber: {s['mode']}{note}}}")
        text = items[k][1]
        if n in per:
            l = per[n]
            cue = l["mode"] + (f": {l['note']}" if l.get("note") else "")
            text = f"{text:<58} {{{cue}}}"
        out.append(text)
        nxt = sung[n] if n < len(sung) else len(items)
        for kind, t in items[k + 1:nxt]:                          # blanks and the user's directions after this line
            out.append(t if kind == "dir" else "")
    return "\n".join(out).rstrip() + "\n"


def render_given(items, cues):
    """The songwriter's own sections (S1, S2 ... in the prompt): each header gets the LLM's section mode."""
    secs = {c["section"]: c for c in cues["sections"]}
    per = {l["n"]: l for l in cues.get("lines", [])}
    out, n, k = [], 0, 0
    for kind, t in items:
        if kind == "sec":
            k += 1
            c = secs.get(f"S{k}", {"mode": "none"})
            note = f": {c['note']}" if c.get("note") else ""
            out.append(f"[{t}] {{amber: {c['mode']}{note}}}")
        elif kind == "sung":
            n += 1
            if n in per:
                l = per[n]
                t = f"{t:<58} {{{l['mode']}{': ' + l['note'] if l.get('note') else ''}}}"
            out.append(t)
        else:
            out.append(t)
    return "\n".join(out).rstrip() + "\n"


def effective(path):
    """-> [(sung text, mode)] for a cued lyrics file."""
    mode, res = "none", []
    for raw in open(path).read().split("\n"):
        t = raw.strip()
        m = re.match(r"^\[[^\]]*\]\s*\{amber:\s*(\w+)", t)
        if m:
            mode = m.group(1)
            continue
        if not t or re.match(r"^\[.*\]$", t):
            continue
        cue = re.search(r"\{(\w+)[^}]*\}\s*$", t)
        res.append((NOTE.sub("", t).strip(), cue.group(1) if cue else mode))
    return res


def score(ref, cand):
    a, b = effective(ref), effective(cand)
    if [x for x, _ in a] != [x for x, _ in b]:
        sys.exit("the two files don't have the same sung lines")
    full = lambda m: m in ("scene", "sing")
    agree = sum(x == y for (_, x), (_, y) in zip(a, b))
    face = sum(full(x) == full(y) for (_, x), (_, y) in zip(a, b))
    print(f"{len(a)} lines: same mode {agree} ({100 * agree / len(a):.0f}%), same face-on-screen yes/no {face} "
          f"({100 * face / len(a):.0f}%); face on screen: reference {sum(full(x) for _, x in a)}, "
          f"candidate {sum(full(y) for _, y in b)}")
    for (t, x), (_, y) in zip(a, b):
        print(f"  {'  ' if x == y else '≠ '}{x:<6} {y:<6} {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyrics", nargs="?")
    ap.add_argument("--treatment")
    ap.add_argument("--out")
    ap.add_argument("--performer", default="a young woman with long wavy blond hair")
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    ap.add_argument("--face-share", default="a third", help="how much of the song shows her face, in words")
    ap.add_argument("--score", nargs=2, metavar=("REFERENCE", "CANDIDATE"))
    a = ap.parse_args()
    if a.score:
        return score(*a.score)
    items = parse(a.lyrics)
    given = any(kind == "sec" for kind, _ in items)
    numbered, n, k = [], 0, 0
    for kind, t in items:
        if kind == "sung":
            n += 1
            numbered.append(f"{n:>3}  {t}")
        elif kind == "sec":
            k += 1
            numbered.append(f"  S{k} [{t}]")
        else:
            numbered.append(f"     {t}")
    rule = ('The sections are given (S1, S2 ...): in "sections" use {"section": "S1", "mode": ..., "note": ...} for '
            'every one, and keep them as they are.' if given else "List every section.")
    treatment = " ".join(open(a.treatment).read().split())
    reply = complete(a.llm, PROMPT.format(performer=a.performer, treatment=treatment, numbered="\n".join(numbered),
                                          face_share=a.face_share, sections_rule=rule))
    cues = json.loads(reply)
    bad = [c for c in cues["sections"] + cues.get("lines", []) if c.get("mode") not in MODES]
    if bad:
        sys.exit(f"unknown modes in the reply: {bad}")
    if not given and 1 not in {s["first_line"] for s in cues["sections"]}:
        cues["sections"].insert(0, {"first_line": 1, "name": "Intro", "mode": "none"})
    text = render_given(items, cues) if given else render(items, cues)
    sung_in = [t for kind, t in items if kind == "sung"]
    sung_out = [NOTE.sub("", l).strip() for l in text.split("\n")
                if l.strip() and not re.match(r"^\[.*\]", l.strip())]
    if sung_in != sung_out:
        sys.exit("rendering changed the sung lines; not writing")
    open(a.out, "w").write(text)
    json.dump(cues, open(a.out + ".json", "w"), indent=1)
    print(f"wrote {a.out} ({len(cues['sections'])} sections, {len(cues.get('lines', []))} line cues)")


if __name__ == "__main__":
    main()
