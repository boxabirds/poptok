"""A title card in the film's own ice (2026-10-27, Ice Walk: "Black with big text "Ice Walk" and second line "by Julian
Harris" in an icy font similar to the spikes in the movie"): the title extruded in clear blue ice (the same glass and
blue absorption as ice_world.py's crystals), a few sharp shards breaking up out of the letters, lit from behind and
above so the light comes through blue, on black. One still.

    blender -b -P pipeline/world/title_card.py -- OUT.png "Ice Walk" "by Julian Harris" [--res 1920x1080 --samples 128]
Font: Iceberg (Cyreal, SIL Open Font License), pipeline/world/fonts/.
"""
import argparse, math, os, random, sys

import bpy
from mathutils import Vector

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("title")
ap.add_argument("byline")
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
rnd = random.Random(a.seed)
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Iceberg-Regular.ttf")
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.resolution_x, sc.render.resolution_y = (int(v) for v in a.res.split("x"))
sc.render.engine = "CYCLES"; sc.cycles.samples = a.samples; sc.cycles.use_denoising = True
if a.gpu:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"; prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type != "CPU"
    sc.cycles.device = "GPU"
sc.cycles.max_bounces = 12; sc.cycles.transmission_bounces = 12
sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Punchy"
w = bpy.data.worlds.new("black"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)


def ice(name, absorb):                  # ice_world.py's crystal ice: clear, glossy, blue deep inside
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.78, 0.9, 1.0, 1); b.inputs["Roughness"].default_value = 0.04
    b.inputs["IOR"].default_value = 1.31; b.inputs["Transmission Weight"].default_value = 0.95
    b.inputs["Coat Weight"].default_value = 0.6
    ab = m.node_tree.nodes.new("ShaderNodeVolumeAbsorption")
    ab.inputs["Color"].default_value = (0.45, 0.75, 1.0, 1); ab.inputs["Density"].default_value = absorb
    m.node_tree.links.new(ab.outputs["Volume"], m.node_tree.nodes["Material Output"].inputs["Volume"])
    return m


def text(body, size, y, depth, mat):
    cu = bpy.data.curves.new(body, "FONT"); cu.body = body; cu.font = bpy.data.fonts.load(FONT)
    cu.size = size; cu.align_x = "CENTER"; cu.align_y = "CENTER"
    cu.extrude = depth; cu.bevel_depth = depth * 0.12; cu.bevel_resolution = 0     # one sharp chamfer, cut-crystal edges
    ob = bpy.data.objects.new(body, cu); sc.collection.objects.link(ob)
    ob.location = (0, 0, y); ob.rotation_euler = (math.radians(90), 0, 0)          # standing up, facing -y
    cu.materials.append(mat)
    return ob


def shard(base, h, r, lean):            # a thin leaning hexagonal crystal, like the film's spires
    import bmesh
    me = bpy.data.meshes.new("shard"); bm = bmesh.new(); n = 6
    ring = [bm.verts.new((r * math.cos(2 * math.pi * k / n) * rnd.uniform(0.7, 1.2),
                          r * math.sin(2 * math.pi * k / n) * rnd.uniform(0.7, 1.2), 0)) for k in range(n)]
    tip = bm.verts.new((0, 0, h))
    bm.faces.new(list(reversed(ring)))
    for k in range(n):
        bm.faces.new((ring[k], ring[(k + 1) % n], tip))
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new("shard", me); sc.collection.objects.link(ob)
    ob.location = base; ob.rotation_euler = (lean[0], lean[1], rnd.uniform(0, 6.3))
    return ob


title_m, by_m = ice("title_ice", 1.2), ice("byline_ice", 0.3)
bb = by_m.node_tree.nodes["Principled BSDF"]                 # the byline frosted, so small letters stay readable
bb.inputs["Roughness"].default_value = 0.35; bb.inputs["Transmission Weight"].default_value = 0.5
bb.inputs["Base Color"].default_value = (0.85, 0.93, 1.0, 1)
t = text(a.title, 1.6, 0.35, 0.22, title_m)
b = text(a.byline, 0.42, -0.95, 0.04, by_m)
bpy.context.view_layer.update()
tw = t.dimensions.x
for k in range(9):                      # shards breaking up out of the title's top edge, leaning outwards
    x = rnd.uniform(-0.5, 0.5) * tw; s = rnd.uniform(0.35, 1.0)
    sh = shard(Vector((x, rnd.uniform(-0.05, 0.15), 0.55)), 0.9 * s, 0.07 * s,
               (rnd.uniform(-0.25, 0.25), x / tw * 0.9))
    sh.data.materials.append(title_m)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.location = (0, -11, 0.2); cam.rotation_euler = (math.radians(90), 0, 0); cam.data.lens = 50


def area(name, loc, energy, color, size):
    L_ = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA")); sc.collection.objects.link(L_)
    L_.data.energy = energy; L_.data.color = color; L_.data.size = size; L_.location = loc
    L_.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    L_.visible_camera = False
    return L_


area("back", (0, 4, 2.5), 900, (0.75, 0.88, 1.0), 6)       # behind and above: light through the ice, blue
area("key", (-4, -6, 4), 120, (1.0, 0.96, 0.92), 3)        # a cool white key for the facets' glints
area("rim", (5, 2, -1), 200, (0.6, 0.8, 1.0), 2)
sc.render.filepath = os.path.abspath(a.out)
bpy.ops.render.render(write_still=True)
print("title ok", a.out)
