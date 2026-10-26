"""Flat-colour layouts of a blockout world, and training views for a per-world renderer (2026-10-26). Matrix-Game 4.0's
recipe: the 3D proxy is deliberately plain (one flat colour per kind of thing) and a renderer fine-tuned on that one
world supplies the look. Painting the look onto the geometry instead (paint_world.py) stitched separately restyled views
into an incoherent patchwork that Layout-To-Render then reproduced exactly (user review: "a comically incoherent mess").

    docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work blender -b SCENE.blend \
        -P pipeline/world/flat_layout.py -- OUT [--views 60] [--shots] [--seed 0]
Writes, for training pairs, OUT/views/NNN_flat.png (one colour per kind of thing, under soft even light) and OUT/views/NNN_clay.png (the same view as lit grey
clay, what an image model restyles into the target), with OUT/views/views.json (camera of each); with --shots, the shot
cameras' flat layout videos OUT/flat_<cam>/%04d.png. Her figure (figure.py) is hidden in the training views (her look
comes from her references, not from this world's renderer) and has her own flat colour in the shots.
"""
import argparse, json, math, os, random, sys

import bpy
from mathutils import Vector

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--views", type=int, default=60)
ap.add_argument("--shots", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
out = os.path.abspath(a.out)
sc = bpy.context.scene
rnd = random.Random(a.seed)

# one flat colour per kind of thing, by object name (room_blockout.py's names); distinct hues, mid values, so the
# renderer can tell them apart and nothing reads as a material
CLASSES = [  # (name prefixes, colour)
    (("floor",), (0.45, 0.30, 0.18)),
    (("ceiling",), (0.55, 0.55, 0.70)),
    (("wall", "far_wall", "near_wall"), (0.30, 0.55, 0.35)),
    (("window", "far_sill", "far_lintel", "windowsill"), (0.85, 0.85, 0.30)),
    (("door",), (0.70, 0.35, 0.60)),
    (("table",), (0.85, 0.50, 0.20)),
    (("chair",), (0.20, 0.45, 0.80)),
    (("rubble",), (0.55, 0.20, 0.20)),
    (("crate",), (0.95, 0.75, 0.55)),
    (("street", "building"), (0.15, 0.15, 0.25)),
]
HER = (0.95, 0.95, 0.95)
FIG = ("torso", "head", "thigh_l", "thigh_r", "shin_l", "shin_r", "cloth")
figure = [o for o in sc.objects if o.type == "MESH" and (o.name.split(".")[0] in FIG
          or (o.parent is not None and o.parent.type == "ARMATURE"))]
room = [o for o in sc.objects if o.type == "MESH" and o not in figure]


def flat(rgb, name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    # lit, not emissive: an emission-only flat colour merged neighbouring surfaces of one kind (no wall corners)
    bs = nt.nodes.new("ShaderNodeBsdfDiffuse"); bs.inputs["Color"].default_value = (*rgb, 1)
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(bs.outputs["BSDF"], o.inputs["Surface"])
    return m


def colour_of(ob):
    n = ob.name.split(".")[0]
    for prefixes, rgb in CLASSES:
        if n.startswith(prefixes):
            return rgb
    return (0.5, 0.5, 0.5)


flat_mats = {ob.name: flat(colour_of(ob), f"flat_{ob.name}") for ob in room}
her_mat = flat(HER, "flat_her")
clay = bpy.data.materials.new("clay"); clay.use_nodes = True
clay.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1)


def use(kind):                  # every object gets its flat colour, or grey clay
    for ob in room + figure:
        ob.data.materials.clear()
        ob.data.materials.append((her_mat if ob in figure else flat_mats[ob.name]) if kind == "flat" else clay)


sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.use_denoising = True
sc.view_settings.view_transform = "Standard"
sc.world.use_nodes = True
bg = sc.world.node_tree.nodes["Background"]


def render(path, kind):
    use(kind)
    bg.inputs[0].default_value = (1, 1, 1, 1)
    sc.cycles.samples = 32
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


# training views: random cameras inside the room, at eye heights 0.9-1.9 m, looking at random points in the room
fl = next(o for o in room if o.name == "floor")
bb = [fl.matrix_world @ Vector(v) for v in fl.bound_box]
x0, x1 = min(p.x for p in bb), max(p.x for p in bb)
y0, y1 = min(p.y for p in bb), max(p.y for p in bb)
z0 = min(p.z for p in bb)
cam = bpy.data.objects.new("cam_view", bpy.data.cameras.new("cam_view")); sc.collection.objects.link(cam)
os.makedirs(f"{out}/views", exist_ok=True)
sc.render.resolution_percentage = 50          # 960x544 training views (the trainer's bucket); shots stay full size
for ob in figure:
    ob.hide_render = True
meta = []
for i in range(a.views):
    m = 0.12
    pos = Vector((rnd.uniform(x0 + m * (x1 - x0), x1 - m * (x1 - x0)), rnd.uniform(y0 + m * (y1 - y0), y1 - m * (y1 - y0)),
                  z0 + rnd.uniform(0.9, 1.9)))
    at = Vector((rnd.uniform(x0, x1), rnd.uniform(y0, y1), z0 + rnd.uniform(0.2, 1.4)))
    if (at - pos).length < 1.5:
        at = pos + (at - pos).normalized() * 1.5
    cam.location = pos
    cam.rotation_euler = (at - pos).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = rnd.choice((18, 24, 28, 35))
    sc.camera = cam
    for kind in ("flat", "clay"):
        render(f"{out}/views/{i:03d}_{kind}.png", kind)
    meta.append({"i": i, "loc": list(pos), "at": list(at), "lens": cam.data.lens})
json.dump(meta, open(f"{out}/views/views.json", "w"), indent=1)
for ob in figure:
    ob.hide_render = False

if a.shots:                     # the shot cameras' flat layouts, her included
    sc.render.resolution_percentage = 100
    use("flat"); bg.inputs[0].default_value = (1, 1, 1, 1); sc.cycles.samples = 16
    for c in [o for o in sc.objects if o.type == "CAMERA" and o.name.startswith("cam_") and o.name != "cam_view"]:
        sc.camera = c
        sc.render.filepath = f"{out}/flat_{c.name[4:]}/"
        bpy.ops.render.render(animation=True)
print("flat ok", out)
