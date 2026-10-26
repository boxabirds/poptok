"""Paint one view for paint_world.py (2026-10-26): restyle the view's partial render (painted surfaces with their paint,
the rest grey clay) with Qwen-Image-Edit, then keep the new paint only where this view paints (need.png, feathered), so
surfaces painted by earlier views keep their pixels exactly and the new paint continues them.

    python3 pipeline/world/fill_view.py VIEW_DIR STYLE.json [--seed 7]
STYLE.json: {"universe": "what the world is made of, its palette and light", "place": "what the place is, in a few
words"}. The place must not list objects: the model then adds them where the blockout has none, and that paint lands on
walls and floor (first run, 2026-10-26: a door, a crate and a chair that weren't there).
Writes VIEW_DIR/styled.png (and VIEW_DIR/qwen.png, the raw restyle). From the second view on, crops of what earlier
views painted go in as picture 2 (VIEW_DIR/swatches.png): with only a little painted surface in frame, the second view
of the first run invented a different material (cracked tiles instead of felt). Samples, not a whole earlier view: given
a whole image, Qwen-Edit copies its camera (world test round 2). Runs in morph-spark on a Spark.
"""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qwen_edit import load

ap = argparse.ArgumentParser()
ap.add_argument("view")
ap.add_argument("style")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
st = json.load(open(a.style))
partial = Image.open(f"{a.view}/partial.png").convert("RGB")
W, H = partial.size
need = np.array(Image.open(f"{a.view}/need.png").convert("L"), np.float32) / 255
clay = not glob.glob(f"{a.view}/../v*/styled.png")        # the first view: nothing painted yet
def swatches(view, n=12, side=256):     # crops of surfaces earlier views painted, as a sheet of material samples
    import random
    rnd, tiles = random.Random(0), []
    for d in sorted(glob.glob(f"{view}/../v*/styled.png")):
        d = os.path.dirname(d)
        img = Image.open(f"{d}/styled.png").convert("RGB")
        m = np.array(Image.open(f"{d}/need.png").convert("L")) > 127
        h, w = m.shape
        for _ in range(400):
            if len(tiles) >= n:
                break
            x, y = rnd.randrange(0, w - side), rnd.randrange(0, h - side)
            if m[y:y + side, x:x + side].mean() > 0.95:
                tiles.append(img.crop((x, y, x + side, y + side)))
    if not tiles:
        return None
    sheet = Image.new("RGB", (4 * side, 3 * side), (128, 128, 128))
    for i, t in enumerate(tiles[:n]):
        sheet.paste(t, ((i % 4) * side, (i // 4) * side))
    sheet.save(f"{view}/swatches.png")
    return sheet


sheet = None if clay else swatches(a.view)
PROMPT = ("Restyle this image. Keep EXACTLY its composition: the same camera position, angle and lens, the same framing, "
          "every object and wall in the same place with the same size and outline. Add nothing and remove nothing: no "
          "new furniture, doors, openings, objects or debris; only give the surfaces that are there their materials. "
          + ("" if clay else "Parts of it are already finished: keep them exactly as they are, and make the plain grey "
             "parts match them, the same materials, colours and light, so it reads as one continuous place. ")
          + ("" if sheet is None else "Picture 2 is not a scene: it is a sheet of material samples cut from this same "
             "place. Paint picture 1's surfaces in exactly these materials, textures and colours; take nothing else "
             "from picture 2 (not its layout, not its framing). ")
          + "The materials and light: " + st["universe"] + " The place is " + st["place"] + ". No people. No text.")
out = load()([partial] + ([sheet] if sheet else []), PROMPT, seed=a.seed, size=(W, H)).convert("RGB").resize((W, H))
out.save(f"{a.view}/qwen.png")
k = max(3, int(W * 0.006)) | 1                 # feather the seam by a few pixels, inside the painted area only
m = np.array(Image.fromarray((need * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(k))
             .filter(ImageFilter.GaussianBlur(k / 2)), np.float32)[..., None] / 255
Image.fromarray((np.array(partial) * (1 - m) + np.array(out) * m).astype(np.uint8)).save(f"{a.view}/styled.png")
print("styled", a.view, f"paints {need.mean():.0%} of the frame", flush=True)
