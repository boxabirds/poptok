"""D38 test 1: can MOSS-Image draw charcoal? cuckoo's punchline in several wordings, 2 seeds each, run where MOSS fits
(a Spark, morph-spark container, from ~/poptok):  python3 lean/poc_medium.py OUT"""
import json, os, sys, time

sys.path.insert(0, "pipeline")
from keyframe_model import load

SCENE = ("seen from high up among the trees, looking down: a young woman sunk to her waist in mud in a vast endless "
         "forest, looking up, reaching one hand towards the sky, a forlorn face, a small black cuckoo flying away above")
MED = "a charcoal drawing on rough paper, black and white"
RICH = ("vine charcoal drawing on textured paper, smudged strokes, heavy blacks, charcoal dust, white paper showing "
        "through, hatching, hand-drawn, black and white")
WORDINGS = [
    ("A medium first", f"{MED[0].upper()}{MED[1:]}: {SCENE}", ""),
    ("B rich medium first", f"{RICH[0].upper()}{RICH[1:]}: {SCENE}", ""),
    ("C medium last", f"{SCENE[0].upper()}{SCENE[1:]}, drawn in charcoal on paper, black and white", ""),
    ("D rich + negative", f"{RICH[0].upper()}{RICH[1:]}: {SCENE}", "photograph, photo, photorealistic, 3d render, colour"),
    ("E as poptok wrote it", "Charcoal, black and white, motion blur. Torn Paper Terrain: The forest is jagged, layered "
     "paper shards. The ground is soft, crumpled pulp that absorbs light. The woman is cut from heavy cardstock with "
     "frayed edges. The sky is a stark white void bleeding through the paper. Medium shot, normal lens, low angle "
     f"looking up. {SCENE}.", ""),
]
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
r = load("moss", offload=False)
log = []
for name, prompt, neg in WORDINGS:
    r.negative = neg                          # read at each render (keyframe_model.MOSSImage.render)
    for seed in (1, 2):
        t = time.time()
        r.render(None, [(prompt, 1.0)], 1.0, seed=seed, size=(1024, 576))[0].save(f"{out}/{name[0]}{seed}.png")
        print(f"{name} seed {seed}: {time.time() - t:.0f} s", flush=True)
    log.append({"name": name, "prompt": prompt, "negative": neg})
json.dump(log, open(f"{out}/wordings.json", "w"), indent=1)
