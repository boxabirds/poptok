"""Render a range of an Ice Walk world built once (D22, 2026-10-28): every chunk rebuilt the same world for 4.3 min before
its first frame; the farm now builds it once (ice_world.py --build-only, cached by content) and each chunk opens it here.

    blender -b -P pipeline/world/ice_render.py -- SCENE.blend OUT --range A,B [--gpu] [--snow-cache DIR] [--only F,F]
OUT/render/%04d.png (clip frames A..B), the same checks and "ice ok" report as ice_world.py.
"""
import argparse, os, sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ice_range

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("out")
ap.add_argument("--range", required=True)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--snow-cache", default="", help="this box's DIR/blendcache_snow (the .blend holds the building box's path)")
ap.add_argument("--only", default="")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.scene), load_ui=False)
sc = bpy.context.scene
if a.gpu:
    ice_range.use_gpu(sc)
if a.snow_cache:
    ice_range.point_snow(a.snow_cache)
R0, R1 = (int(v) for v in a.range.split(","))
mode = "layout" if sc.render.engine == "BLENDER_WORKBENCH" else "render"
os.makedirs(os.path.abspath(a.out), exist_ok=True)
ice_range.check_and_render(sc, os.path.abspath(a.out), mode, R0, R1, a.only)
