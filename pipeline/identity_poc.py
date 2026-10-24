"""Identity-chain test with LOCAL models (macvideo POC 6 equivalent, locally): with ANCHOR=photo the performer's photo
is the fixed reference for every still; without it still 1 is by MOSS (text only). Stills by Qwen-Image-Edit-2511 with reference images = still 1 (fixed
anchor) + the 2 most recent stills (Qwen-Image-Edit takes up to 3 images).

    python3 pipeline/identity_poc.py OUT_DIR [--steps 40]      (runs in morph-spark on a DGX Spark; models in /models)
Writes OUT_DIR/<scene>.png, meta.json (per-still seconds, references used), sheet.jpg (anchor + all stills).
    ANCHOR=photo STYLE="..." python3 pipeline/identity_poc.py OUT  -> the singer as the fixed identity reference
"""
from __future__ import annotations

import argparse, gc, json, os, sys, time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PERFORMER = ("the young woman in the reference photo: long wavy blond hair, blue eyes, fair skin with light freckles, "
             "slim, early twenties")
ANCHOR = os.environ.get("ANCHOR", "")      # a reference photo of the performer (the singer); empty: still 1 by MOSS
SCENES = [("01_window", "Light finds me: she sits by a thin-paned window as warm early light falls across her face"),
          ("02_hanger", "Something soft slipping off its hanger: she watches a cream cardigan slide from a wooden hanger"),
          ("03_flowers", "Pressing flowers into margins: she presses dried rose petals between the pages of an old book"),
          ("04_boats", "Little paper boats sinking in a drawer: she kneels by an open drawer of folded paper boats"),
          ("05_hallway", "Names I whisper then let go of in the hallway: she stands in a dim narrow hallway, eyes closed")]
STYLE = os.environ.get("STYLE", "cinematic photograph, natural light, no text, no lettering, no signage")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--cfg", type=float, default=4.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
meta = []

made = []
if ANCHOR:                                                        # the singer's photo is the fixed identity anchor
    Image.open(ANCHOR).convert("RGB").save(f"{a.out}/00_anchor.png"); made.append(f"{a.out}/00_anchor.png")
else:                                                             # still 1: MOSS, text only
    from keyframe_model import load
    r = load("moss", offload=False)
    t = time.time()
    first, _ = r.render(None, [(f"{PERFORMER}. {SCENES[0][1]}. {STYLE}", 1.0)], 1.0, seed=1001, size=(1024, 576))
    first.save(f"{a.out}/{SCENES[0][0]}.png"); made.append(f"{a.out}/{SCENES[0][0]}.png")
    meta.append({"shot": SCENES[0][0], "model": "moss", "seconds": round(time.time() - t, 1), "references": []})
    print(meta[-1], flush=True)
    del r; gc.collect(); torch.cuda.empty_cache()

from diffusers import QwenImageEditPlusPipeline                   # stills 2-5: identity-chained edits
pipe = QwenImageEditPlusPipeline.from_pretrained("/models/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16).to("cuda")
pipe.set_progress_bar_config(disable=True)
for name, scene in (SCENES if ANCHOR else SCENES[1:]):
    refs = [made[0]] + [p for p in made[1:]][-2:]                   # anchor + 2 most recent
    images = [Image.open(p).convert("RGB") for p in refs]
    prompt = (f"The same person as in the reference images: {PERFORMER}. Keep her face, hair, eyes and age exactly the "
              f"same. New scene, 16:9 landscape: {scene}. {STYLE}")
    t = time.time()
    out = pipe(image=images, prompt=prompt, negative_prompt=" ", true_cfg_scale=a.cfg, num_inference_steps=a.steps,
               generator=torch.Generator("cpu").manual_seed(1001 + len(made)), height=576, width=1024).images[0]
    out.save(f"{a.out}/{name}.png"); made.append(f"{a.out}/{name}.png")
    meta.append({"shot": name, "model": "qwen-image-edit-2511", "seconds": round(time.time() - t, 1),
                 "references": [os.path.basename(p) for p in refs]})
    print(meta[-1], flush=True)
json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
ims = [Image.open(p).resize((512, 288)) for p in made]
sheet = Image.new("RGB", (512 * 3, 288 * 2), "black")
for i, im in enumerate(ims):
    sheet.paste(im, ((i % 3) * 512, (i // 3) * 288))
sheet.save(f"{a.out}/sheet.jpg")
print("wrote", a.out)
