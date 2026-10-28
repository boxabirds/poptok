"""Experiment (2026-10-26): can the directors' workshop choose an ambitious universe (stained glass, wool, cardboard,
oil) rather than photographic realism with a colour grade? The user, reviewing the treatment round: "both videos have
concrete real world context with some colouring ... nothing creative"; what they enjoyed was choosing between stained
glass, oil, wool, cardboard boxes.

Per piece, from its lyrics, brief and treatment (inputs/story.md):
    baseline   the universe the workshop chose (inputs/universe.txt)
    materials  a dedicated pass: 8 universes where the world is MADE of something (a craft or a medium)
    metaphor   a dedicated pass: 6 universes derived from the song's central image (what the material means)
    pick       a critic chooses the 3 best of all candidates (ambition, meaning, can it be drawn, can she live in it)
               and writes 2 still prompts for each from the treatment (its opening image, its turn)
    python3 pipeline/universe_lab.py OUT propose PIECE...     (LLM, port 8091: WORKSHOP_PY=... bin/workshop.sh OUT propose ...)
    python3 pipeline/universe_lab.py OUT render                (MOSS stills, in the keyframe container)
    python3 pipeline/universe_lab.py OUT sheet
"""
from __future__ import annotations

import argparse, json, os, re, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONTEXT = """Song lyrics:
---
{lyrics}
---
The artist's brief:
---
{brief}
---
The video's story (the treatment the directors agreed):
---
{treatment}
---
"""
EXAMPLES = (" For example a world of felted wool, of stained glass, of corrugated cardboard boxes and string, of oil paint "
            "thick as cake icing, of folded paper, of clay, of embroidery on linen, of a toy theatre, of ice, of candle wax, "
            "of mosaic tiles, of woodcut prints, of matchsticks, of tin toys, of light through lace.")
NO_EXAMPLES = (" Invent it from this song: no stock list of crafts; the less obvious the material, the better, as long as "
               "it can be drawn.")
RULES = """What a UNIVERSE is: the stuff the whole video is made of, true in every frame, so that the city, the sky, the
rain, the people and the performer herself are all made of it, as in a stop-motion or art film.{examples} It is NOT a colour grade, a film stock, a lighting style or a
mood ("noir", "desaturated cyan", "high-contrast monochrome", "gritty realism" are not universes). Photographic realism
is not allowed in this exercise.
The brief says WHO is in the video and WHAT happens (she is a soldier in a field uniform, she carries a cloth, and so
on): keep all of that, but render it IN the universe (a soldier's uniform of felted wool; rain of glass beads). Words
like "real fabric" in the brief are about honesty and weight, not about photography; the universe can be honest and
heavy too. Choose universes where the MATERIAL itself carries the song's meaning (cardboard that soaks and sags in
the rain of a war; wool that unravels; glass that cracks and lets light through), so that what happens in the story
can happen to the material.
Every universe must be drawable by an image model in a single still and stay the same across 40 shots, and the
performer must be able to appear in it (her face recognisable, made of the same stuff or lit within it)."""
RULES = RULES.replace("{examples}", NO_EXAMPLES if "--no-examples" in sys.argv else EXAMPLES)
MATERIALS = CONTEXT + RULES + """

Propose 8 DIFFERENT universes for this video, as different from each other as you can (no two of the same family:
not three kinds of paper). Be ambitious: the most striking music videos commit to one bold made world.
Return ONLY JSON: {{"universes": [{{"name": "2-4 words", "made_of": "40-70 words: what everything is made of, surfaces,
how light behaves, palette", "meaning": "one sentence: why this material serves this song and story", "story": "one
sentence: what happens to the material at the story's turn"}}]}}"""
METAPHOR = CONTEXT + RULES + """

Start from the song's central image and feeling (what it is really about), and ask: if the whole world of this video
were MADE of one thing, what would make the meaning physical? Propose 6 universes found this way, each a different
material, at least two of them surprising (a material nobody would first think of for this song).
Return ONLY JSON: {{"universes": [{{"name": "2-4 words", "made_of": "40-70 words: what everything is made of, surfaces,
how light behaves, palette", "meaning": "one sentence: why this material serves this song and story", "story": "one
sentence: what happens to the material at the story's turn"}}]}}"""
PICK = CONTEXT + RULES + """

Candidate universes:
{cands}

You are the producer. Choose the 3 best for this video: ambitious and striking, the material carrying the song's
meaning, drawable and consistent for a whole video, the performer at home in it. Prefer three that differ from each
other. For each, write two keyframe prompts from the treatment: its OPENING image and the moment of its TURN, each
60-90 words, starting with the universe name and a colon, describing what is in the frame and what everything is
made of (the performer described as {performer}, wearing what the brief says, made of or rendered in the universe).
No text or lettering.
Return ONLY JSON: {{"picks": [{{"name": "...", "why": "one sentence", "made_of": "...", "opening": "...",
"turn": "..."}}]}}"""
BASE = CONTEXT + """
The universe the directors chose: {name}: {made_of}
Write two keyframe prompts in this universe from the treatment: its OPENING image and the moment of its TURN, each
60-90 words, starting with "{name}:", describing what is in the frame (the performer described as {performer},
wearing what the brief says). No text or lettering.
Return ONLY JSON: {{"opening": "...", "turn": "..."}}"""


def llm(prompt, temperature=0.9, url="http://127.0.0.1:8091"):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                       "max_tokens": 4096}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def ask(prompt, temperature=0.9):
    for _ in range(3):
        t = llm(prompt, temperature)
        m = re.search(r"\{.*\}", t, re.S)
        try:
            return json.loads(m.group(0))
        except (AttributeError, json.JSONDecodeError):
            print("  unparsable answer, again", flush=True)
    raise SystemExit("no JSON from the LLM")


def env(piece, key):
    """KEY's value in piece.env as bash reads it: quotes removed, an inline # comment dropped, the last one wins (the
    regex this replaced kept a trailing comment in the value: cuckoo's PERFORMER_PHOTO, 2026-10-28)."""
    import shlex
    v = ""
    for line in open(f"{piece}/piece.env"):
        if line.startswith(f"{key}="):
            w = shlex.split(line[len(key) + 1:], comments=True); v = w[0] if w else ""
    return v


def propose(a):
    plan = json.load(open(f"{a.out}/plan.json")) if os.path.exists(f"{a.out}/plan.json") else {}
    for piece in a.pieces:
        name = os.path.basename(piece.rstrip("/"))
        if name in plan:
            continue
        lyr = env(piece, "LYRICS") or "inputs/lyrics.txt"
        ctx = dict(lyrics=open(f"{piece}/{lyr}").read().strip(), brief=open(f"{piece}/inputs/brief.txt").read().strip(),
                   treatment=open(f"{piece}/inputs/story.md").read().split("## Treatment", 1)[-1].split("**Opening")[0].strip())
        performer = env(piece, "PERFORMER") or "the singer"
        t = time.time()
        uni = open(f"{piece}/inputs/universe.txt").read().split("\n", 1)
        base = {"name": uni[0].strip(), "made_of": " ".join(uni[1].split()) if len(uni) > 1 else ""}
        base.update(ask(BASE.format(**ctx, **base, performer=performer), 0.5))
        mats = ask(MATERIALS.format(**ctx))["universes"]
        meta = ask(METAPHOR.format(**ctx))["universes"]
        cands = "\n".join(f"- {u['name']}: {u['made_of']} ({u.get('meaning', '')} {u.get('story', '')})" for u in mats + meta)
        picks = ask(PICK.format(**ctx, cands=cands, performer=performer), 0.5)["picks"][:3]
        plan[name] = {"baseline": base, "materials": mats, "metaphor": meta, "picks": picks}
        json.dump(plan, open(f"{a.out}/plan.json", "w"), indent=1, ensure_ascii=False)
        print(f"{name} ({time.time() - t:.0f} s): baseline {base['name']}; candidates "
              f"{', '.join(u['name'] for u in mats + meta)}; picks {', '.join(p['name'] for p in picks)}", flush=True)
    md = ["# Universe lab", ""]
    for name, p in plan.items():
        md += [f"## {name}", "", f"**Baseline (the workshop's choice):** {p['baseline']['name']}: {p['baseline']['made_of']}", "",
               "**Candidates:**", ""]
        md += [f"- *{u['name']}*: {u['made_of']} {u.get('meaning', '')} {u.get('story', '')}" for u in p["materials"] + p["metaphor"]]
        md += ["", "**Picks:**", ""] + [f"{i + 1}. **{u['name']}**: {u.get('why', '')}" for i, u in enumerate(p["picks"])] + [""]
    open(f"{a.out}/universes.md", "w").write("\n".join(md))


def stills(plan):
    for name, p in plan.items():
        for i, u in enumerate([p["baseline"]] + p["picks"]):
            for k in ("opening", "turn"):
                yield name, i, u["name"], k, u[k]


def render(a):
    from keyframe_model import load
    plan = json.load(open(f"{a.out}/plan.json"))
    r = load("moss", offload=False)
    for name, i, uname, k, prompt in stills(plan):
        f = f"{a.out}/{name}_{i}_{k}.png"
        if os.path.exists(f):
            continue
        t = time.time()
        im, _ = r.render(None, [(prompt, 1.0)], 1.0, seed=4100 + i, size=(1024, 576))
        im.save(f)
        print(f"{name} {uname} {k}: {time.time() - t:.0f} s", flush=True)


def sheet(a):
    from PIL import Image, ImageDraw
    plan = json.load(open(f"{a.out}/plan.json"))
    w, h = 384, 216
    for name, p in plan.items():
        us = [p["baseline"]] + p["picks"]
        im = Image.new("RGB", (2 * w, len(us) * (h + 24)), "white")
        d = ImageDraw.Draw(im)
        for i, u in enumerate(us):
            d.text((4, i * (h + 24) + 6), ("baseline: " if i == 0 else f"pick {i}: ") + u["name"], fill="black")
            for j, k in enumerate(("opening", "turn")):
                f = f"{a.out}/{name}_{i}_{k}.png"
                if os.path.exists(f):
                    im.paste(Image.open(f).convert("RGB").resize((w, h)), (j * w, i * (h + 24) + 24))
        im.save(f"{a.out}/{name}_sheet.jpg", quality=88)
        print("wrote", f"{a.out}/{name}_sheet.jpg")


ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("step", choices=["propose", "render", "sheet"])
ap.add_argument("pieces", nargs="*")
ap.add_argument("--no-examples", action="store_true", help="no list of example materials in the prompt (the first run's "
                "candidates came almost all from that list)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
{"propose": propose, "render": render, "sheet": sheet}[a.step](a)
