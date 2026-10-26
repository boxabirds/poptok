"""Training targets for a per-world renderer (2026-10-26): restyle each clay view from flat_layout.py on its own with
Qwen-Image-Edit (no stitching onto geometry: paint_world.py's stitched views were incoherent), one picture only, the same
prompt and seed for every view. (--sheet adds a sheet of material samples from the first views as picture 2, for
consistent materials; it made Qwen ignore the layout, so targets no longer matched their layouts.)

    python3 pipeline/world/style_views.py VIEWS_DIR STYLE.json [--seed 7]
STYLE.json: {"universe": "...", "place": "a few words, no list of objects"}. Writes VIEWS_DIR/NNN_styled.png for each
NNN_clay.png still missing one. Runs in morph-spark on a Spark.
"""
from __future__ import annotations

import argparse, glob, json, os, random, sys, time

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qwen_edit import load

ap = argparse.ArgumentParser()
ap.add_argument("views")
ap.add_argument("style")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--from-flat", action="store_true", help="restyle the flat-colour view, with the colour key in the prompt "
                "(instead of grey clay, where blank walls left Qwen free to add windows, doors and furniture)")
ap.add_argument("--only", nargs="*", help="view numbers to do (e.g. 026 033)")
ap.add_argument("--suffix", default="styled")
ap.add_argument("--person", default="", help="the figure in the view, described (views of her); default: no people")
ap.add_argument("--sheet", action="store_true", help="add the material-sample sheet as picture 2 (off: with it, Qwen "
                "ignored the layout and drew the same room corner in ~55 of 60 views, 2026-10-26)")
a = ap.parse_args()
st = json.load(open(a.style))
BASE = ("Restyle this image. Keep EXACTLY its composition: the same camera position, angle and lens, the same framing, "
        "every object and wall in the same place with the same size and outline. Add nothing and remove nothing: no new "
        "furniture, doors, openings, objects or debris; only give the surfaces that are there their materials. "
        "The materials and light: " + st["universe"] + " The place is " + st["place"] + ". "
        + (f"The figure is a real person, in exactly its pose, size and place: {a.person}. " if a.person else "No people. ")
        + "No text.")
SHEET = (" Picture 2 is not a scene: it is a sheet of material samples from this same place. Paint picture 1's surfaces "
         "in exactly these materials, textures and colours; take nothing else from picture 2 (not its layout, not its "
         "framing).")


def swatches(paths, n=12, side=192):
    rnd, tiles = random.Random(0), []
    imgs = [Image.open(p).convert("RGB") for p in paths]
    while len(tiles) < n:
        im = rnd.choice(imgs)
        x, y = rnd.randrange(0, im.width - side), rnd.randrange(0, im.height - side)
        tiles.append(im.crop((x, y, x + side, y + side)))
    sheet = Image.new("RGB", (4 * side, 3 * side))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 4) * side, (i // 4) * side))
    return sheet


KEY = (" The image is a colour-coded layout: green surfaces are walls, brown is the floor, pale lilac is the ceiling, "
       "yellow is the window frame and sill (the glass shows the night outside), pink is the door, orange is a table, "
       "blue is a chair, dark red blocks are rubble, the cream block is a crate. A plain stretch of one colour is a plain "
       "stretch of that surface: keep it plain.")
edit = load()
clays = sorted(glob.glob(f"{a.views}/*_clay.png"))
if a.only:
    clays = [c for c in clays if os.path.basename(c)[:3] in a.only]
sheet = None
for c in clays:
    out = c.replace("_clay.png", f"_{a.suffix}.png")
    done = sorted(glob.glob(f"{a.views}/*_styled.png"))
    if a.sheet and sheet is None and len(done) >= 3:
        sheet = swatches(done[:3]); sheet.save(f"{a.views}/swatches.png")
    if os.path.exists(out):
        continue
    t = time.time()
    im = Image.open(c.replace("_clay.png", "_flat.png") if a.from_flat else c).convert("RGB")
    res = edit([im] + ([sheet] if sheet else []), BASE + (KEY if a.from_flat else "") + (SHEET if sheet else ""),
               seed=a.seed, size=im.size)
    res.convert("RGB").resize(im.size).save(out)
    print(f"{os.path.basename(out)}: {time.time() - t:.0f} s", flush=True)
