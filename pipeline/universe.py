"""The universe step (2026-10-26): the world the whole video is made of, chosen from a sheet of candidates. Between
the story (inputs/story.md) and the storyboard; manual, so a choice made by hand is kept.

Why a step of its own: inside the directors' conversation the universe was one topic among many and came out as
photographic realism with a colour grade ("Hand-Tinted Noir", "Monochrome Electric Blue"); the user: "no ambition at
all ... what I really enjoyed was choosing between stained glass, oil, wool, cardboard boxes". In a dedicated pass,
without a list of example materials (the universe lab, pieces/_bench/universes*: with a list, the candidates were the
list), the LLM proposes made worlds that MOSS and Qwen-Edit can draw.

    python3 pipeline/universe.py PIECE propose      LLM (port 8091): 14 candidates (8 open + 6 from the song's central
                                                    image, plus the directors' own from story), 3 picks, 2 shot specs
    python3 pipeline/universe.py PIECE render       stills: each pick x each shot, from her photo (Qwen-Image-Edit,
                                                    in morph-spark on a Spark); without a photo, MOSS
    python3 pipeline/universe.py PIECE sheet        inputs/universes/sheet.jpg
    python3 pipeline/universe.py PIECE choose N     pick N (1-3) becomes inputs/universe.txt and inputs/visual-look.txt
`propose` chooses pick 1 when no universe has been chosen yet. The two shot specs are written once, universe-neutral
(camera, framing, where she is and what she does), and every universe renders the same two: the lab's stills had a
new composition per universe and a different woman per still (no photo), so neither the worlds nor she could be
compared (user review 2026-10-26).
Writes in PIECE/inputs/: universes.json, universes.md, universes/ (stills, sheet.jpg), universe.txt, visual-look.txt.
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
The video's story (the treatment):
---
{treatment}
---
"""
RULES = """What a UNIVERSE is: the stuff the whole video is made of, true in every frame, so that the places, the sky,
the weather, the people and the performer herself are all made of it, as in a stop-motion or art film. Invent it from
this song: no stock list of crafts; the less obvious the material, the better, as long as it can be drawn. It is NOT a
colour grade, a film stock, a lighting style or a mood ("noir", "desaturated cyan", "high-contrast monochrome",
"gritty realism" are not universes), and photographic realism is not one.
The brief says WHO is in the video and WHAT happens: keep all of that, but render it IN the universe (a soldier's
uniform of felted wool; rain of glass beads). Words like "real fabric" in the brief are about honesty and weight, not
about photography. A look the brief mentions as tried before is one candidate, not a rule, unless the brief says the
look is fixed. Choose universes where the MATERIAL itself carries the song's meaning, so that what happens in the story
can happen to the material. Every universe must be drawable by an image model and stay the same across 40 shots, and
the performer must be able to appear in it, her face recognisable."""
OPEN = CONTEXT + RULES + """

Propose 8 DIFFERENT universes for this video, as different from each other as you can (no two of the same family).
Be ambitious: the most striking music videos commit to one bold made world.
Return ONLY JSON: {{"universes": [{{"name": "2-4 words", "made_of": "40-70 words: what everything is made of, surfaces,
how light behaves, palette", "meaning": "one sentence: why this material serves this song and story", "story": "one
sentence: what happens to the material at the story's turn"}}]}}"""
METAPHOR = CONTEXT + RULES + """

Start from the song's central image and feeling, and ask: if the whole world of this video were MADE of one thing,
what would make the meaning physical? Propose 6 universes found this way, each a different material, at least two of
them surprising.
Return ONLY JSON: {{"universes": [{{"name": "2-4 words", "made_of": "40-70 words: what everything is made of, surfaces,
how light behaves, palette", "meaning": "one sentence: why this material serves this song and story", "story": "one
sentence: what happens to the material at the story's turn"}}]}}"""
PICK = CONTEXT + RULES + """

Candidate universes:
{cands}

You are the producer. Choose the 3 best for this video: ambitious and striking, the material carrying the song's
meaning, drawable and consistent for a whole video, the performer at home in it; three that differ from each other.
Rank them, best first. For each, also write its LOOK: the text that styles every shot (what everything is made of,
light, palette, how the camera moves and how changes happen, taken from the treatment), 60-110 words, never
describing the performer.
Return ONLY JSON: {{"picks": [{{"name": "...", "made_of": "40-70 words", "why": "one sentence", "look": "..."}}]}}"""
SHOTS = CONTEXT + """
Write TWO shot specifications from this story, to be rendered in several different universes (materials) for
comparison, so write them WITHOUT any material, colour grade or style: only what a camera sees. Shot 1: the story's
opening image. Shot 2: the moment of its turn. Both show the performer ({performer}). Each 50-80 words: the camera
(height, angle, lens: wide / normal / long), the framing (wide, medium, close-up), where she is in the frame and how
big, her pose and what she does, what else is in the frame and where. Concrete and physically possible.
Return ONLY JSON: {{"shots": [{{"name": "opening", "spec": "..."}}, {{"name": "turn", "spec": "..."}}]}}"""


def llm(prompt, temperature=0.9, url="http://127.0.0.1:8091"):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                       "max_tokens": 4096}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def ask(prompt, key, temperature=0.9):
    for _ in range(3):
        m = re.search(r"\{.*\}", llm(prompt, temperature), re.S)
        try:
            v = json.loads(m.group(0))[key]
            if v:
                return v
        except (AttributeError, json.JSONDecodeError, KeyError, TypeError):
            pass
        print("  unusable answer, again", flush=True)
    raise SystemExit(f"universe: no {key} from the LLM")


def env(piece, key):
    m = re.search(rf'^{key}="?([^"\n]*)', open(f"{piece}/piece.env").read(), re.M)
    return m.group(1).strip() if m else ""


def paths(piece):
    i = f"{piece}/inputs"
    return dict(json=f"{i}/universes.json", md=f"{i}/universes.md", dir=f"{i}/universes", uni=f"{i}/universe.txt",
                look=f"{i}/visual-look.txt", workshop=f"{i}/universe.workshop.txt")


def propose(a):
    P = paths(a.piece)
    ctx = dict(lyrics=open(f"{a.piece}/{env(a.piece, 'LYRICS') or 'inputs/lyrics.txt'}").read().strip(),
               brief=open(f"{a.piece}/inputs/brief.txt").read().strip() if os.path.exists(f"{a.piece}/inputs/brief.txt") else "",
               treatment=open(f"{a.piece}/inputs/story.md").read().split("## Treatment", 1)[-1].strip())
    performer = env(a.piece, "PERFORMER") or "the singer"
    t = time.time()
    cands = ask(OPEN.format(**ctx), "universes") + ask(METAPHOR.format(**ctx), "universes")
    if os.path.exists(P["workshop"]):                  # the directors' own universe (workshop.py --stage story)
        w = open(P["workshop"]).read().split("\n", 1)
        cands.append({"name": w[0].strip(), "made_of": " ".join(w[1].split()) if len(w) > 1 else "",
                      "meaning": "the directors' choice in the workshop", "story": ""})
    listing = "\n".join(f"- {u['name']}: {u['made_of']} ({u.get('meaning', '')} {u.get('story', '')})" for u in cands)
    picks = ask(PICK.format(**ctx, cands=listing), "picks", 0.5)[:3]
    shots = ask(SHOTS.format(**ctx, performer=performer), "shots", 0.4)[:2]
    json.dump({"candidates": cands, "picks": picks, "shots": shots}, open(P["json"], "w"), indent=1, ensure_ascii=False)
    md = [f"# Universe: {os.path.basename(a.piece.rstrip('/'))}", "",
          "Made by `pipeline/universe.py` (the universe step). The sheet `inputs/universes/sheet.jpg` shows each pick in "
          "the same two shots. Choose with `python3 pipeline/universe.py PIECE choose N`, or edit `inputs/universe.txt` "
          "(line 1 the name, then what everything is made of) and `inputs/visual-look.txt` by hand.", "",
          "## Picks", ""]
    md += [f"{i + 1}. **{u['name']}**: {u['made_of']} *{u.get('why', '')}*" for i, u in enumerate(picks)]
    md += ["", "## The two shots", ""] + [f"- **{s['name']}**: {s['spec']}" for s in shots]
    md += ["", "## All candidates", ""] + [f"- **{u['name']}**: {u['made_of']} {u.get('meaning', '')}" for u in cands]
    open(P["md"], "w").write("\n".join(md) + "\n")
    print(f"universe: {len(cands)} candidates, picks {', '.join(u['name'] for u in picks)} ({time.time() - t:.0f} s)")
    if not os.path.exists(P["uni"]) or a.force:
        choose(a, 1)


def choose(a, n=None):
    P = paths(a.piece)
    u = json.load(open(P["json"]))["picks"][(n or a.n) - 1]
    open(P["uni"], "w").write(u["name"] + "\n" + u["made_of"] + "\n")
    open(P["look"], "w").write(f"{u['name']}: {u['made_of']} {u.get('look', '')}".strip() + "\n")
    print(f"universe: chose {u['name']} -> {P['uni']}, {P['look']}")


def prompt_for(u, shot, wardrobe):
    wear = f"wearing {wardrobe}, rendered in this universe" if wardrobe else "her clothes rendered in this universe"
    return (f"{u['name']}: {u['made_of']} The whole image is in this universe: everything in it, the place, the objects "
            f"and the woman's clothes, is made of it. The woman from the reference image, the same person: keep her face, "
            f"hair, eyes and age exactly, {wear}, lit by the scene's own light. {shot['spec']} No text or lettering.")


def render(a):
    P = paths(a.piece)
    d = json.load(open(P["json"]))
    os.makedirs(P["dir"], exist_ok=True)
    photo = env(a.piece, "PERFORMER_PHOTO")
    photo = os.path.normpath(os.path.join(a.piece, photo)) if photo else ""
    wardrobe = env(a.piece, "WARDROBE")
    if photo:
        from qwen_edit import load
        edit = load()
        draw = lambda p, seed: edit([photo], p, seed=seed, size=(1024, 576))
    else:
        from keyframe_model import load
        r = load("moss", offload=False)
        draw = lambda p, seed: r.render(None, [(p, 1.0)], 1.0, seed=seed, size=(1024, 576))[0]
    for i, u in enumerate(d["picks"]):
        for j, s in enumerate(d["shots"]):
            f = f"{P['dir']}/pick{i + 1}_{s['name']}.png"
            if os.path.exists(f):
                continue
            t = time.time()
            draw(prompt_for(u, s, wardrobe), 4200 + j).save(f)      # the same seed per shot in every universe
            print(f"{u['name']} {s['name']}: {time.time() - t:.0f} s", flush=True)


def sheet(a):
    from PIL import Image, ImageDraw
    P = paths(a.piece)
    d = json.load(open(P["json"]))
    w, h = 512, 288
    im = Image.new("RGB", (len(d["shots"]) * w, len(d["picks"]) * (h + 26)), "white")
    dr = ImageDraw.Draw(im)
    for i, u in enumerate(d["picks"]):
        dr.text((6, i * (h + 26) + 7), f"{i + 1}. {u['name']}", fill="black")
        for j, s in enumerate(d["shots"]):
            f = f"{P['dir']}/pick{i + 1}_{s['name']}.png"
            if os.path.exists(f):
                im.paste(Image.open(f).convert("RGB").resize((w, h)), (j * w, i * (h + 26) + 26))
    im.save(f"{P['dir']}/sheet.jpg", quality=88)
    print("universe: sheet", f"{P['dir']}/sheet.jpg")


ap = argparse.ArgumentParser()
ap.add_argument("piece")
ap.add_argument("step", choices=["propose", "render", "sheet", "choose"])
ap.add_argument("n", nargs="?", type=int, default=1)
ap.add_argument("--force", action="store_true", help="propose: choose pick 1 even if a universe was chosen")
a = ap.parse_args()
{"propose": propose, "render": render, "sheet": sheet, "choose": choose}[a.step](a)
