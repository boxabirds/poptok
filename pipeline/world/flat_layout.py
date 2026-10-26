"""Flat-colour layouts of a blockout world, and training views for a per-world renderer (2026-10-26). Matrix-Game 4.0's
recipe: the 3D proxy is deliberately plain (one flat colour per kind of thing) and a renderer fine-tuned on that one
world supplies the look. Painting the look onto the geometry instead (paint_world.py) stitched separately restyled views
into an incoherent patchwork that Layout-To-Render then reproduced exactly (user review: "a comically incoherent mess").

    docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work blender -b SCENE.blend \
        -P pipeline/world/flat_layout.py -- OUT [--views 60] [--her-views N] [--shots] [--seed 0] [--reuse]
Proxy (2026-10-26, after Matrix-Game 4.0's report): flat colours, black outlines, a checkered floor; plus a depth pass
(NNN_depth.png) and a caption of what each view shows, written from the scene (views.json "caption").
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
ap.add_argument("--her-views", type=int, default=0, help="also N views aimed at her figure (OUT/her_views/), at varied "
                "frames of her animation, distances and angles")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--reuse", action="store_true", help="re-render the flat layouts (and depth, captions) of the views "
                "already in OUT (views.json): same cameras, so existing clay views and styled targets stay valid")
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


def rigged(o):                  # under an armature somewhere up its parents: her body, her hair and clothes (figure.py)
    while o.parent is not None:
        o = o.parent
        if o.type == "ARMATURE":
            return True
    return False
figure = [o for o in sc.objects if o.type == "MESH" and (o.name.split(".")[0] in FIG
          or rigged(o))]
room = [o for o in sc.objects if o.type == "MESH" and o not in figure]


def flat(rgb, name, checker=False):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    # lit, not emissive: an emission-only flat colour merged neighbouring surfaces of one kind (no wall corners)
    bs = nt.nodes.new("ShaderNodeBsdfDiffuse"); bs.inputs["Color"].default_value = (*rgb, 1)
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(bs.outputs["BSDF"], o.inputs["Surface"])
    if checker:                 # Matrix-Game 4.0's proxy: a checkered ground shows scale and the camera's motion over it
        ch = nt.nodes.new("ShaderNodeTexChecker"); ch.inputs["Scale"].default_value = 2.0      # 0.5 m squares
        ch.inputs["Color1"].default_value = (*rgb, 1)
        ch.inputs["Color2"].default_value = (*[c * 0.72 for c in rgb], 1)
        geo = nt.nodes.new("ShaderNodeNewGeometry"); nt.links.new(geo.outputs["Position"], ch.inputs["Vector"])
        nt.links.new(ch.outputs["Color"], bs.inputs["Color"])
    return m


def colour_of(ob):
    n = ob.name.split(".")[0]
    for prefixes, rgb in CLASSES:
        if n.startswith(prefixes):
            return rgb
    return (0.5, 0.5, 0.5)


flat_mats = {ob.name: flat(colour_of(ob), f"flat_{ob.name}", checker=ob.name.split(".")[0] == "floor") for ob in room}
her_mat = flat(HER, "flat_her")
her_mats = {ob.name: flat(tuple(ob["flat_rgb"]), f"flat_{ob.name}") for ob in figure if "flat_rgb" in ob}   # dressed
clay = bpy.data.materials.new("clay"); clay.use_nodes = True
clay.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1)


depth = bpy.data.materials.new("depth"); depth.use_nodes = True     # distance from the camera, 0.3 m white .. 7 m black
_n = depth.node_tree; _n.nodes.clear()
_cd = _n.nodes.new("ShaderNodeCameraData"); _mr = _n.nodes.new("ShaderNodeMapRange")
_mr.inputs["From Min"].default_value, _mr.inputs["From Max"].default_value = 0.3, 7.0
_mr.inputs["To Min"].default_value, _mr.inputs["To Max"].default_value = 1.0, 0.0
_em = _n.nodes.new("ShaderNodeEmission"); _o = _n.nodes.new("ShaderNodeOutputMaterial")
_n.links.new(_cd.outputs["View Z Depth"], _mr.inputs["Value"]); _n.links.new(_mr.outputs["Result"], _em.inputs["Color"])
_n.links.new(_em.outputs["Emission"], _o.inputs["Surface"])


def use(kind):                  # every object gets its flat colour, grey clay, or its depth
    for ob in room + figure:
        ob.data.materials.clear()
        ob.data.materials.append((her_mats.get(ob.name, her_mat) if ob in figure else flat_mats[ob.name]) if kind == "flat"
                                 else depth if kind == "depth" else clay)
    # Matrix-Game 4.0's proxy draws every primitive with a black outline: edges stay readable where two surfaces of one
    # kind meet. Outlines on the flat layouts only (not the clay an image model restyles, nor depth or ids)
    sc.render.use_freestyle = kind == "flat"


sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.use_denoising = True
sc.render.line_thickness_mode = "ABSOLUTE"; sc.render.line_thickness = 1.5
_fs = bpy.context.view_layer.freestyle_settings          # a black outline line set (scenes from figure.py have none)
_ls = _fs.linesets[0] if len(_fs.linesets) else _fs.linesets.new("outline")
if _ls.linestyle is None:
    _ls.linestyle = bpy.data.linestyles.new("outline")
_ls.linestyle.color = (0, 0, 0); _ls.linestyle.thickness = 1.5
_ls.select_by_visibility = True; _ls.select_by_edge_types = True
_ls.select_silhouette = _ls.select_border = _ls.select_crease = True
sc.view_settings.view_transform = "Standard"
sc.world.use_nodes = True
bg = sc.world.node_tree.nodes["Background"]


def render(path, kind):
    use(kind)
    bg.inputs[0].default_value = (0, 0, 0, 1) if kind == "depth" else (1, 1, 1, 1)
    sc.cycles.samples = 1 if kind == "depth" else 32
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
# what a view shows: an unlit render with one exact colour per kind of thing, counted. Views that show fewer than four
# kinds (a bare wall, a corner) are rejected: Qwen-Edit cannot restyle so little without inventing a room (2026-10-26:
# windows, doors and armchairs appeared on bare walls, from clay and from a colour key alike)
ids = {}
for k, (prefixes, _) in enumerate(CLASSES):
    ids[k] = flat(((k * 37 % 11) / 10, (k * 53 % 7) / 6, (k * 71 % 5) / 4), f"id{k}")
    ids[k].node_tree.nodes.clear()
    em = ids[k].node_tree.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = ((k + 1) / 16, 0, 0, 1)
    ids[k].node_tree.links.new(em.outputs["Emission"], ids[k].node_tree.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])


ids[len(CLASSES)] = flat((1, 1, 1), "id_her")      # her (all her parts): one more id
ids[len(CLASSES)].node_tree.nodes.clear()
_e = ids[len(CLASSES)].node_tree.nodes.new("ShaderNodeEmission")
_e.inputs["Color"].default_value = ((len(CLASSES) + 1) / 16, 0, 0, 1)
ids[len(CLASSES)].node_tree.links.new(_e.outputs["Emission"],
                                      ids[len(CLASSES)].node_tree.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
NAMES = {3: "the window", 4: "the open door", 5: "a tilted table", 6: "a chair on its side", 7: "rubble blocks",
         8: "a crate", 9: "the street outside", len(CLASSES): "the young woman"}


def caption(stats):             # what the view shows and where, from the id render (Matrix-Game 4.0's dynamic captions)
    parts = []
    for k, (frac, cx) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        if k in NAMES:
            where = "on the left" if cx < 0.36 else "on the right" if cx > 0.64 else "in the middle"
            parts.append(f"{NAMES[k]} {where}")
    return "In view: " + ", ".join(parts) + "." if parts else ""


def kinds_in_view(stats=None):
    sc.render.use_freestyle = False
    for ob in room + figure:
        n = ob.name.split(".")[0]
        k = len(CLASSES) if ob in figure else next((k for k, (pf, _) in enumerate(CLASSES) if n.startswith(pf)), None)
        ob.data.materials.clear(); ob.data.materials.append(ids[k] if k is not None else clay)
    bg.inputs[0].default_value = (0, 0, 0, 1); sc.cycles.samples = 1; sc.cycles.use_denoising = False
    pct, sc.render.resolution_percentage = sc.render.resolution_percentage, 12
    sc.render.filepath = f"{out}/views/_ids.png"; bpy.ops.render.render(write_still=True)
    sc.render.resolution_percentage = pct; sc.cycles.use_denoising = True
    im = bpy.data.images.load(f"{out}/views/_ids.png", check_existing=False)
    import numpy as np
    r = np.array(im.pixels[:]).reshape(-1, 4)[:, 0]
    # the PNG holds sRGB-encoded values and an 8-bit image's pixels come back as stored: back to linear before reading the
    # id (read as linear, ids were wrong: the first captions named her in views where she was hidden)
    r = np.where(r <= 0.04045, r / 12.92, ((r + 0.055) / 1.055) ** 2.4)
    bpy.data.images.remove(im)
    k = np.round(r * 16) - 1
    w = int(round(sc.render.resolution_x * 0.12)); h = len(k) // w
    k = k.reshape(h, w)
    if stats is not None:
        for c in range(len(CLASSES) + 1):
            m = k == c
            if m.mean() >= 0.02:
                stats[c] = (float(m.mean()), float(np.nonzero(m)[1].mean() / w))
    return sum((k == c).mean() >= 0.02 for c in range(len(CLASSES)))


def place(m):
    cam.location = Vector(m["loc"])
    cam.rotation_euler = (Vector(m["at"]) - Vector(m["loc"])).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = m["lens"]; sc.camera = cam
    bpy.context.view_layer.update()


if a.reuse:                     # same cameras, new flat layouts + depth + captions; clay and targets untouched
    for sub, with_her in (("views", False), ("her_views", True)):
        path = f"{out}/{sub}/views.json"
        if not os.path.exists(path):
            continue
        ms = json.load(open(path))
        for ob in figure:
            ob.hide_render = not with_her
        for m in ms:
            if "frame" in m:
                sc.frame_set(m["frame"])
            place(m)
            st = {}; kinds_in_view(st); m["caption"] = caption(st)
            for kind in ("flat", "depth"):
                render(f"{out}/{sub}/{m['i']:03d}_{kind}.png", kind)
        json.dump(ms, open(path, "w"), indent=1)
        print(f"reused {len(ms)} {sub}", flush=True)
    a.views, a.her_views = 0, 0

meta, tries = [], 0
while len(meta) < a.views and tries < 50 * a.views:
    tries += 1
    i = len(meta)
    # stand in the outer band of the room and look across it, at eye height, with a normal-to-wide lens: shots, not
    # close-ups of a wall
    u, v = rnd.random(), rnd.random()
    side = rnd.randrange(4)
    fx = (0.15 + 0.15 * u) if side == 0 else (0.70 + 0.15 * u) if side == 1 else 0.2 + 0.6 * u
    fy = (0.15 + 0.15 * v) if side == 2 else (0.70 + 0.15 * v) if side == 3 else 0.2 + 0.6 * v
    pos = Vector((x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), z0 + rnd.uniform(1.0, 1.8)))
    at = Vector((x0 + (1 - fx + rnd.uniform(-0.2, 0.2)) * (x1 - x0), y0 + (1 - fy + rnd.uniform(-0.2, 0.2)) * (y1 - y0),
                 z0 + rnd.uniform(0.4, 1.2)))
    cam.location = pos
    cam.rotation_euler = (at - pos).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = rnd.choice((16, 18, 22, 26))
    sc.camera = cam
    bpy.context.view_layer.update()
    st = {}
    n = kinds_in_view(st)
    if n < 4:
        continue
    for kind in ("flat", "clay", "depth"):
        render(f"{out}/views/{i:03d}_{kind}.png", kind)
    meta.append({"i": i, "loc": list(pos), "at": list(at), "lens": cam.data.lens, "kinds": int(n), "caption": caption(st)})
if a.views:
    print(f"views: kept {len(meta)} of {tries} tried", flush=True)
    json.dump(meta, open(f"{out}/views/views.json", "w"), indent=1)
for ob in figure:
    ob.hide_render = False

if a.her_views and figure:      # views of her: the renderer learns her from her own figure in this world
    os.makedirs(f"{out}/her_views", exist_ok=True)
    sc.render.resolution_percentage = 50
    hm = []
    for i in range(a.her_views):
        f = rnd.randint(sc.frame_start, sc.frame_end); sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        pts = [ob.matrix_world @ v.co for ob in figure if ob.type == "MESH" for v in list(ob.evaluated_get(dg).data.vertices)[::25]]
        c = sum(pts, Vector()) / len(pts)
        ang, dist = rnd.uniform(0, 2 * math.pi), rnd.uniform(1.3, 3.5)
        pos = Vector((c.x + dist * math.cos(ang), c.y + dist * math.sin(ang), z0 + rnd.uniform(0.6, 1.8)))
        pos.x, pos.y = min(max(pos.x, x0 + 0.2), x1 - 0.2), min(max(pos.y, y0 + 0.2), y1 - 0.2)   # stay in the room
        at = c + Vector((rnd.uniform(-0.3, 0.3), rnd.uniform(-0.3, 0.3), rnd.uniform(-0.1, 0.3)))
        cam.location = pos; cam.rotation_euler = (at - pos).to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = rnd.choice((24, 28, 35, 50)); sc.camera = cam
        bpy.context.view_layer.update()
        st = {}; kinds_in_view(st)
        for kind in ("flat", "clay", "depth"):
            render(f"{out}/her_views/{i:03d}_{kind}.png", kind)
        hm.append({"i": i, "frame": f, "loc": list(pos), "at": list(at), "lens": cam.data.lens, "caption": caption(st)})
    json.dump(hm, open(f"{out}/her_views/views.json", "w"), indent=1)
    sc.render.resolution_percentage = 100

if a.shots:                     # the shot cameras' flat layouts, her included
    sc.render.resolution_percentage = 100
    use("flat"); bg.inputs[0].default_value = (1, 1, 1, 1); sc.cycles.samples = 16   # outlines on (use("flat"))
    for c in [o for o in sc.objects if o.type == "CAMERA" and o.name.startswith("cam_") and o.name != "cam_view"]:
        sc.camera = c
        sc.render.filepath = f"{out}/flat_{c.name[4:]}/"
        bpy.ops.render.render(animation=True)
print("flat ok", out)
