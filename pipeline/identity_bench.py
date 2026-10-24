"""Identity endurance test with LOCAL models: macvideo POC 32 (40 trap scenes, one recurring character) on
Qwen-Image-Edit-2511. Still 0 is by MOSS from text; every later still is a Qwen-Image-Edit render whose reference
images are chosen by the policy:
    anchors   the first N stills (default 3) as fixed identity anchors, nothing else (macvideo POC 33's passing policy)
    rolling   2 anchors + the previous still (Qwen-Image-Edit takes up to 3 references)
With `anchors`, every scene after the anchors depends only on the anchors, so ranges can run on different boxes at once.

    python3 pipeline/identity_bench.py FIXTURE OUT --range 0:3            (the anchors, in order)
    python3 pipeline/identity_bench.py FIXTURE OUT --range 3:21           (on one box)
    python3 pipeline/identity_bench.py FIXTURE OUT --range 21:40          (on another, at the same time)
    python3 pipeline/identity_bench.py FIXTURE OUT --assemble             (contact sheet + assessment, no GPU)
Runs in morph-spark on a DGX Spark (models in /models, no offload). Writes OUT/scene_NNN_<name>.png, OUT/meta/NNN.json,
and with --assemble OUT/contact-sheet.jpg and OUT/assessment.md (macvideo's POC 32 scoring table).
"""
from __future__ import annotations

import argparse, gc, glob, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("fixture")
ap.add_argument("out")
ap.add_argument("--range", default="0:40", help="scene indices START:END")
ap.add_argument("--policy", choices=["anchors", "rolling"], default="anchors")
ap.add_argument("--anchor-count", type=int, default=3)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--cfg", type=float, default=4.0)
ap.add_argument("--size", default="1024x576")
ap.add_argument("--lightning", default="", help="Qwen-Image-Edit-2511-Lightning LoRA (8 or 4 steps): pass --steps to "
                "match; true CFG is then off (1.0)")
ap.add_argument("--assemble", action="store_true")
a = ap.parse_args()
fx = json.load(open(a.fixture))
scenes = fx["scenes"]
os.makedirs(f"{a.out}/meta", exist_ok=True)
path = lambda i: f"{a.out}/scene_{i:03d}_{scenes[i]['name']}.png"


def references(i):
    anchors = list(range(min(a.anchor_count, i)))[:3]
    if a.policy == "anchors" or i <= a.anchor_count:
        return anchors
    return anchors[:2] + [i - 1]                                  # 2 anchors + the previous still


def prompt(i, with_refs):
    lead = f"The same {fx['subject']} as in the reference images. " if with_refs else ""
    return f"{lead}{fx['brief']}\n\nScene {i + 1:02d}/{len(scenes)}: {scenes[i]['prompt']}. {fx['suffix']}"


def assemble():
    from PIL import Image, ImageDraw
    metas = [json.load(open(p)) for p in sorted(glob.glob(f"{a.out}/meta/*.json"))]
    w, h, cols = 384, 216, 8
    rows = (len(metas) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, (h + 34) * rows), "white")
    d = ImageDraw.Draw(sheet)
    for k, m in enumerate(metas):
        x, y = (k % cols) * w, (k // cols) * (h + 34)
        sheet.paste(Image.open(f"{a.out}/{m['file']}").convert("RGB").resize((w, h)), (x, y))
        d.text((x + 4, y + h + 2), f"#{m['index']:02d} {m['name']}  refs {m['reference_indices']}", fill="black")
        d.text((x + 4, y + h + 17), f"trap: {m['trap']}  {m['seconds']}s", fill="black")
    sheet.save(f"{a.out}/contact-sheet.jpg", quality=88)
    table = "\n".join(f"| {m['index']} | {m['name']} | {m['trap']} | `{m['file']}` | {m['reference_indices']} |"
                      + " |" * 16 for m in metas)
    secs = [m["seconds"] for m in metas]
    open(f"{a.out}/assessment.md", "w").write(f"""# Identity endurance (macvideo POC 32 fixture, local models)

Open `contact-sheet.jpg`. Policy: {metas[-1]['policy']}; anchors: first {a.anchor_count} stills. Still 0 by MOSS, the
rest by Qwen-Image-Edit-2511 ({metas[-1]['steps']} steps, true CFG {metas[-1]['cfg']}). {len(metas)} stills,
{sum(secs) / len(secs):.0f} s average, {max(secs):.0f} s slowest.

## Identity anchor

{fx['brief']}

## Scoring

Use 1-5 where 1 is consistent/no drift and 5 is severe drift (macvideo's POC 32 table).

| # | Scene | Trap | Image | Refs | Same person | Woman swap | Bald swap | Young swap | Clean-shaven swap | Beard | Eyes | Cap/hair | Build | Wardrobe | Primary subject | Distractor takeover | Palette | Environment | Hard fail | Notes |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{table}

## Visual assessment (Claude)

- Identity verdict:
- Drift start, if any:
- Hard failures:
- Soft failures:
- Decision:

## Visual assessment (user)

- Identity verdict:
- Decision:
""")
    print("wrote", f"{a.out}/contact-sheet.jpg", f"{a.out}/assessment.md")


if a.assemble:
    assemble()
    sys.exit()

import torch
from PIL import Image

W, H = (int(v) for v in a.size.split("x"))
lo, hi = (int(v) for v in a.range.split(":"))
todo = [i for i in range(lo, hi) if not os.path.exists(path(i))]


def record(i, refs, model, seconds):
    json.dump({"index": i, **scenes[i], "file": os.path.basename(path(i)), "reference_indices": refs, "model": model,
               "policy": a.policy, "steps": a.steps, "cfg": a.cfg, "seconds": round(seconds, 1)},
              open(f"{a.out}/meta/{i:03d}.json", "w"), indent=1)
    print(f"=== {i:02d} {scenes[i]['name']} refs={refs} trap={scenes[i]['trap']} {seconds:.0f}s", flush=True)


if 0 in todo:                                                     # still 0: text only, by MOSS
    from keyframe_model import load
    r = load("moss", offload=False)
    t = time.time()
    im, _ = r.render(None, [(prompt(0, False), 1.0)], 1.0, seed=1000, size=(W, H))
    im.save(path(0)); record(0, [], "moss", time.time() - t)
    del r; gc.collect(); torch.cuda.empty_cache()
    todo.remove(0)

if todo:
    from diffusers import QwenImageEditPlusPipeline
    pipe = QwenImageEditPlusPipeline.from_pretrained("/models/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
    if a.lightning:                     # step-distilled: the scheduler settings from ModelTC/Qwen-Image-Lightning
        import math
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
            "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
            "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
            "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False})
        pipe.load_lora_weights(a.lightning)
        pipe.fuse_lora()
        a.cfg = 1.0
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for i in todo:
        refs = references(i)
        missing = [r for r in refs if not os.path.exists(path(r))]
        if missing:
            sys.exit(f"scene {i} needs references {missing} first")
        t = time.time()
        im = pipe(image=[Image.open(path(r)).convert("RGB") for r in refs], prompt=prompt(i, True), negative_prompt=" ",
                  true_cfg_scale=a.cfg, num_inference_steps=a.steps, height=H, width=W,
                  generator=torch.Generator("cpu").manual_seed(1000 + i)).images[0]
        im.save(path(i)); record(i, refs, "qwen-image-edit-2511", time.time() - t)
