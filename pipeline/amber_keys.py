"""Keyframes for the shots that show the performer: Qwen-Image-Edit-2511 (Lightning, 8 steps) with her photo as the only
reference (fixed anchor, no chaining through generated stills: POC 18/20), so she is the same person in every shot.
The other shots are MOSS's (keyframes.py --skip-cast).

    python3 pipeline/amber_keys.py OUT --photo pieces/artists/r.amber.jpeg [--style "..."]
Reads OUT/shots.json (shots with "cast": scene / sing / part), writes OUT/keys/shot_XXX.png for those still missing.
Runs in morph-spark on a DGX Spark (models in /models, no offload).
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--photo", required=True)
ap.add_argument("--style", default="")
ap.add_argument("--style-word", default="", help="a few-word style (POC 24): she is rendered in it too")
ap.add_argument("--style-brief", default="")
ap.add_argument("--size", default="1024x576")
ap.add_argument("--wardrobe", default="", help="what she wears in this piece (e.g. 'a soldier's camouflage uniform, her "
                "face covered in camouflage paint'); default: the same sheer black dress as in the reference")
ap.add_argument("--keys", default="keys")
a = ap.parse_args()
WEAR = (f"and dress her in {a.wardrobe} (not the clothes in the reference)" if a.wardrobe else
        "and the same sheer black dress as in the reference (unless the shot says otherwise)")
KEEP = (f"keep her face, hair, eyes and age exactly, {WEAR}; light her with the scene's own light, direction and colour, "
        "so she belongs in it")
LEAD = {                                # wardrobe and lighting lines: user review 2026-10-24 (a blouse turned into a
    "scene": f"The woman from the reference image, the same person: {KEEP}. Her pose suits the scene. ",  # cardigan; she
    "sing": f"The woman from the reference image, the same person: {KEEP}. She is singing, lips parted. ",  # was lit apart
    "part": "Only part of the woman from the reference image is visible (a hand, her back, her silhouette or a "
            "reflection), never her full face; keep her hair and skin the same"
            + (f", wearing {a.wardrobe}" if a.wardrobe else " and her black dress the same") + ", lit by the scene's own light. ",
}
plan = json.load(open(os.path.join(a.out, "shots.json")))
style = a.style or plan.get("style", "")
todo = [s for s in plan["shots"] if s.get("cast") in LEAD
        and not os.path.exists(os.path.join(a.out, a.keys, f"shot_{s['i']:03d}.png"))]
print(f"{len(todo)} performer shots to render", flush=True)
if todo:
    from qwen_edit import load
    edit = load()
    W, H = (int(v) for v in a.size.split("x"))
    os.makedirs(os.path.join(a.out, a.keys), exist_ok=True)
    for s in todo:
        t = time.time()
        lead = (f"The whole image is in the {a.style_word} style: {a.style_brief} The woman is rendered in this "
                f"{a.style_word} style too. " if a.style_word else "")
        prompt = lead + LEAD[s["cast"]] + s["prompt"] + (f" {style}" if style else "")
        rr = json.load(open(os.path.join(a.out, "reroll.json"))) if os.path.exists(os.path.join(a.out, "reroll.json")) else {}
        im = edit([a.photo], prompt, seed=1000 + s["i"] + 7919 * rr.get(str(s["i"]), 0), size=(W, H))
        im.save(os.path.join(a.out, a.keys, f"shot_{s['i']:03d}.png"))
        print(f"shot {s['i']:3d} {s['cast']:5s} {time.time() - t:.0f}s  {s['prompt'][:70]}", flush=True)
