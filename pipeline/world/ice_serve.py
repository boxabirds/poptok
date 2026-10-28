"""A Blender that stays up and renders ranges of an Ice Walk world (D23, 2026-10-28): it opens the built world once
(ice_world.py --build-only, D22), then renders each range a farm worker sends it on stdin, so a chunk pays neither
Blender's start nor the world's load. One per farm worker, kept open over ssh / docker -i.

    blender -b -P pipeline/world/ice_serve.py -- SCENE.blend [--gpu] [--snow-cache DIR]
stdin, one line per range: "A B OUT" -> OUT/render/A..B.png, the same "ice check" and "ice ok" lines as ice_render.py,
then "done A B" (or "failed A B: error"); "quit" or end of input stops it. "ready" once the world is open.
"""
import argparse, os, sys, traceback

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ice_range

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--snow-cache", default="")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.scene), load_ui=False)
sc = bpy.context.scene
if a.gpu:
    ice_range.use_gpu(sc)
if a.snow_cache:
    ice_range.point_snow(a.snow_cache)
mode = "layout" if sc.render.engine == "BLENDER_WORKBENCH" else "render"
print("ready", flush=True)
for line in sys.stdin:
    w = line.split()
    if not w:
        continue
    if w[0] == "quit":
        break
    R0, R1, out = int(w[0]), int(w[1]), os.path.abspath(w[2])
    try:
        os.makedirs(out, exist_ok=True)
        ice_range.check_and_render(sc, out, mode, R0, R1)
        print("done", R0, R1, flush=True)
    except Exception as e:                  # this range failed; the next may not (the worker re-queues it)
        traceback.print_exc()
        print("failed", R0, R1, ":", repr(e).replace("\n", " "), flush=True)
