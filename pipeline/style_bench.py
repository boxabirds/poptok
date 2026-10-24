"""One-word styles (macvideo's "filter"): does a few-word style like "stained glass" or "boxes and string" reshape every
frame when the LLM first writes a style brief from it? Stills only, by MOSS.

Method per style, after macvideo pocs/19-abstraction-gallery (PASS_A_PROMPT / PASS_B_PROMPT), with one change: the
brief says only what the world is made of and how it's lit, never who or what is in the shot, so the lyric line
still decides the content (the lesson of POC 12).
    1. style brief: the LLM describes how the style shows in materials, textures, palette, line quality and light
    2. per line: the LLM writes the image prompt for that line, honouring the brief, at an abstraction level 0-100
Baseline "suffix": today's per-line prompt with the style word appended (no brief).

    python3 pipeline/style_bench.py prompts LYRICS TREATMENT OUT --lines 1,2,5,15,18,40   (needs the LLM, port 8091)
    python3 pipeline/style_bench.py render OUT                (MOSS, any keyframe box: tritus, a Spark)
    python3 pipeline/style_bench.py sheet OUT                 (grid: styles x lines)
"""
from __future__ import annotations

import argparse, json, os, re, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STYLES = ["stained glass", "boxes and string", "cyberpunk", "cyanotype", "claymation", "oil impasto"]
ABSTRACTION = {
    0: "fully representational, subjects rendered as concrete recognisable form with grounded proportions and depth",
    25: "loosely expressive: line and texture given primacy over accuracy; subjects clearly legible but simplified",
    50: "heavily stylised: figures become simplified masses, architecture reduced to structural shapes",
}
BRIEF = """You are the art director of a music video rendered ENTIRELY in the "{style}" style.

Song lyrics:
---
{lyrics}
---

Visual treatment from the songwriter (setting, mood, light): {treatment}

Write a style brief (4-6 sentences): how the {style} style manifests in EVERY shot: the materials everything is made
of, textures, palette, line quality, lighting and camera feel, within the treatment's setting and mood. Concrete visual
cues only. Do NOT describe characters, events, places or a story: the lyrics decide what is in each shot.

Return ONLY the brief as a single paragraph."""
LINE = """You are a music video director writing ONE image prompt for a keyframe.

Song lyrics:
---
{lyrics}
---

Style brief (applies to every shot):
---
{brief}
---

Target line: "{line}"
Abstraction level: {abstraction}/100: {abstraction_descriptor}

Write ONE image prompt (2-3 sentences, under 70 words). Depict what this line shows, in context, entirely within the
style brief, at this abstraction level. Describe what IS present. No text, lettering or signage. No emotional labels.

Return ONLY the image prompt as a single paragraph."""
SUFFIX = """Below is a single line from the lyrics of a song, followed by the full lyrics. We want to create a supporting
visual for this single line. Using all lyrics as context for meaning, generate a corresponding vivid visual prompt that
can be used to communicate the meaning of the single line visually. Under 60 words. Do not prefix with any text.

Line: "{line}"

{lyrics}"""


def llm(prompt, url="http://127.0.0.1:8091"):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 600}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return " ".join(json.loads(r.read())["choices"][0]["message"]["content"].split())


def sung_lines(path):
    return [l for l in (re.sub(r"\{[^}]*\}", "", x).strip() for x in open(path)) if l and not re.match(r"^\[.*\]$", l)]


def prompts(a):
    lines = sung_lines(a.lyrics)
    lyrics = "\n".join(lines)
    treatment = " ".join(open(a.treatment).read().split())
    picks = [int(n) for n in a.lines.split(",")]
    plan = {"treatment": treatment, "abstraction": a.abstraction, "lines": {n: lines[n - 1] for n in picks}, "styles": {}}
    plan["styles"]["suffix"] = {"brief": None, "prompts": {
        n: f"{llm(SUFFIX.format(line=lines[n - 1], lyrics=lyrics))} {treatment}" for n in picks}}
    for style in STYLES:
        brief = llm(BRIEF.format(style=style, lyrics=lyrics, treatment=treatment))
        plan["styles"][style] = {"brief": brief, "prompts": {
            n: f"{style}: " + llm(LINE.format(lyrics=lyrics, brief=brief, line=lines[n - 1], abstraction=a.abstraction,
                                              abstraction_descriptor=ABSTRACTION[a.abstraction])) for n in picks}}
        print(style, "|", brief[:120], flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(plan, open(f"{a.out}/plan.json", "w"), indent=1)


def render(a):
    import torch
    from keyframe_model import load
    plan = json.load(open(f"{a.out}/plan.json"))
    r = load("moss", offload=False)
    for style, s in plan["styles"].items():
        for n, p in s["prompts"].items():
            f = f"{a.out}/{style.replace(' ', '_')}_{int(n):02d}.png"
            if os.path.exists(f):
                continue
            t = time.time()
            im, _ = r.render(None, [(p, 1.0)], 1.0, seed=4000 + int(n), size=(1024, 576))
            im.save(f)
            print(f"{style} line {n}: {time.time() - t:.0f}s", flush=True)


def sheet(a):
    from PIL import Image, ImageDraw
    plan = json.load(open(f"{a.out}/plan.json"))
    styles, ns = list(plan["styles"]), list(plan["lines"])
    w, h, lab = 320, 180, 150
    im = Image.new("RGB", (lab + w * len(ns), 40 + h * len(styles)), "white")
    d = ImageDraw.Draw(im)
    for j, n in enumerate(ns):
        d.text((lab + j * w + 4, 4), f"{n}: {plan['lines'][n][:44]}", fill="black")
    for i, style in enumerate(styles):
        d.text((6, 40 + i * h + h // 2), style, fill="black")
        for j, n in enumerate(ns):
            f = f"{a.out}/{style.replace(' ', '_')}_{int(n):02d}.png"
            if os.path.exists(f):
                im.paste(Image.open(f).convert("RGB").resize((w, h)), (lab + j * w, 40 + i * h))
    im.save(f"{a.out}/sheet.jpg", quality=88)
    print("wrote", f"{a.out}/sheet.jpg")


ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["prompts", "render", "sheet"])
ap.add_argument("args", nargs="+")
ap.add_argument("--lines", default="1,2,5,15,18,40")
ap.add_argument("--abstraction", type=int, default=25, choices=sorted(ABSTRACTION))
a = ap.parse_args()
if a.step == "prompts":
    a.lyrics, a.treatment, a.out = a.args
else:
    a.out = a.args[0]
{"prompts": prompts, "render": render, "sheet": sheet}[a.step](a)
