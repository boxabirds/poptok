"""Multi-character casting test with LOCAL models: macvideo POC 34 (Alan, Barry, Christy; 16 scenes of singles, pairs
and trios, with absent cast that must not appear) on MOSS + Qwen-Image-Edit-2511 (Lightning 8 steps by default). Reference policies:
    text_only   every still by MOSS from text alone (the baseline)
    min_cover   macvideo's cast_min_cover: the fewest earlier stills that together show everyone in the scene
                (ties: fewer extra people, then the most recent); scenes with no earlier stills are MOSS text-only
    anchors     each character's first solo still is their fixed anchor; a scene gets the anchors of who is in it
                (macvideo POC 33's lesson: fixed anchors, no chaining through generated group shots)
At most 3 people per scene, so at most 3 references (Qwen-Image-Edit's limit).

    python3 pipeline/cast_bench.py FIXTURE OUT --policy min_cover
    python3 pipeline/cast_bench.py FIXTURE OUT --assemble      (contact sheet per policy, no GPU)
Runs in morph-spark on a DGX Spark (models in /models, no offload). Writes OUT/<policy>/scene_NN_<name>.png and
OUT/<policy>/meta.json; --assemble writes OUT/<policy>/contact-sheet.jpg and OUT/comparison.md.
"""
from __future__ import annotations

import argparse, gc, glob, itertools, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("fixture")
ap.add_argument("out")
ap.add_argument("--policy", choices=["text_only", "min_cover", "anchors"])
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--cfg", type=float, default=4.0)
ap.add_argument("--size", default="1024x576")
ap.add_argument("--lightning", default="/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
                help="step-distilled LoRA (default; '' for the full 40-step model)")
ap.add_argument("--assemble", action="store_true")
a = ap.parse_args()
fx = json.load(open(a.fixture))
scenes, cast, order = fx["scenes"], fx["cast"], fx["cast_order"]
label = lambda c: " + ".join(m.capitalize() for m in c) or "nobody"


def prompt(i, with_refs):
    s = scenes[i]
    absent = [m for m in order if m not in s["cast"]]
    present = "\n".join(f"- {m.capitalize()}: {cast[m]}" for m in s["cast"])
    lead = ("The people in the reference images are the cast; keep each one's face, hair and clothing exactly. "
            if with_refs else "")
    return (f"{lead}Create one 16:9 cinematic music-video keyframe image. Keep each listed cast member visually distinct "
            "and do not merge identities. No readable text, no subtitles, no labels, no signage.\n\n"
            f"Scene {i + 1:02d}/{len(scenes)}: {s['intent']}. Setting: {s['setting']}.\n"
            f"Required cast: {label(s['cast'])}." + (f" Do not show {', '.join(m.capitalize() for m in absent)}." if absent else "")
            + f"\n\nCast descriptions present in this scene:\n{present}\n\n"
            "Composition: every required cast member must be clearly visible enough to identify, with face or defining "
            "wardrobe visible. Background extras may appear only if they do not resemble or replace the required cast. "
            "Coastal realism, natural lighting, 16mm grain.")


def min_cover(i, done):
    need = set(scenes[i]["cast"])
    cands = [j for j in done if set(scenes[j]["cast"]) & need]
    for size in range(1, len(need) + 1):
        best = None
        for combo in itertools.combinations(cands, size):
            cov = set().union(*(scenes[j]["cast"] for j in combo))
            if need <= cov:
                score = (len(cov - need), -max(combo))
                if best is None or score < best[0]:
                    best = (score, list(combo))
        if best:
            return best[1]
    return []


def anchors(i, done):
    first = {}
    for j in done:
        if len(scenes[j]["cast"]) == 1:
            first.setdefault(scenes[j]["cast"][0], j)
    if not all(m in first for m in scenes[i]["cast"]):
        return []
    return [first[m] for m in scenes[i]["cast"]]


def assemble():
    from PIL import Image, ImageDraw
    rows = []
    for pdir in sorted(glob.glob(f"{a.out}/*/meta.json")):
        meta = json.load(open(pdir))
        d0 = os.path.dirname(pdir)
        w, h, cols = 384, 216, 8
        n = len(meta)
        sheet = Image.new("RGB", (w * cols, (h + 34) * ((n + cols - 1) // cols)), "white")
        d = ImageDraw.Draw(sheet)
        for k, m in enumerate(meta):
            x, y = (k % cols) * w, (k // cols) * (h + 34)
            sheet.paste(Image.open(f"{d0}/{m['file']}").convert("RGB").resize((w, h)), (x, y))
            d.text((x + 4, y + h + 2), f"#{m['index']:02d} {m['name']}: {label(m['cast'])}", fill="black")
            d.text((x + 4, y + h + 17), f"{m['model']} refs {m['reference_indices']}  {m['seconds']}s", fill="black")
        sheet.save(f"{d0}/contact-sheet.jpg", quality=88)
        secs = [m["seconds"] for m in meta]
        rows.append(f"| {os.path.basename(d0)} | {n}/{len(scenes)} | {sum(len(m['reference_indices']) for m in meta)} | "
                    f"{sum(secs) / len(secs):.0f} s | | | | |")
    open(f"{a.out}/comparison.md", "w").write(
        "# Multicast casting (macvideo POC 34 fixture, local models)\n\nOpen `<policy>/contact-sheet.jpg`.\n\n"
        "| Policy | Completed | Total refs | Avg time | Required present | Absent leaked | Identity merge | Notes |\n"
        "|---|---:|---:|---:|---|---|---|---|\n" + "\n".join(rows) + "\n")
    print("wrote", f"{a.out}/comparison.md")


if a.assemble:
    assemble()
    sys.exit()

import torch
from PIL import Image

W, H = (int(v) for v in a.size.split("x"))
pdir = f"{a.out}/{a.policy}"
os.makedirs(pdir, exist_ok=True)
path = lambda i: f"{pdir}/scene_{i:02d}_{scenes[i]['name']}.png"
meta, done, moss, edit = [], [], None, None
for i, s in enumerate(scenes):
    refs = [] if a.policy == "text_only" else (min_cover if a.policy == "min_cover" else anchors)(i, done)
    t = time.time()
    if not refs:
        if moss is None:
            from keyframe_model import load
            moss = load("moss", offload=False)
        im, _ = moss.render(None, [(prompt(i, False), 1.0)], 1.0, seed=2000 + i, size=(W, H))
        model = "moss"
    else:
        if edit is None:
            if moss is not None:                                   # one model on the GPU at a time
                del moss; moss = None; gc.collect(); torch.cuda.empty_cache()
            from qwen_edit import load as load_edit
            edit = load_edit(lightning=a.lightning or None, steps=a.steps if not a.lightning else None, cfg=a.cfg)
        im = edit([path(r) for r in refs], prompt(i, True), seed=2000 + i, size=(W, H))
        model = "qwen-image-edit-2511"
    im.save(path(i))
    done.append(i)
    meta.append({"index": i, **s, "file": os.path.basename(path(i)), "reference_indices": refs, "model": model,
                 "seconds": round(time.time() - t, 1)})
    json.dump(meta, open(f"{pdir}/meta.json", "w"), indent=1)
    print(f"=== {a.policy} {i:02d} {s['name']} cast={label(s['cast'])} refs={refs} {model} {meta[-1]['seconds']}s", flush=True)
