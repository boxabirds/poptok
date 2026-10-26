"""A procedural ice landscape the camera travels through in time with the music (2026-10-26, Ice Walk proof of concept:
an abstract, lyric-free piece). The geometry is generated here (a wind-carved ice sheet, pressure ridges, translucent ice
spires and distant ice cliffs); the camera's motion comes from music_curves.py's spectral bands: the lows push it
forward in surges, the mids sway and turn it, the highs make the low sun glint in the ice.

    docker run --rm [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all] --user $(id -u):$(id -g) -v $PWD:/work -w /work \
        blender -b -P pipeline/world/ice_world.py -- CURVES.json OUT [--mode render|layout] [--frames N] [--seed S]
        [--res 1920x1080] [--samples 64] [--gpu]
render: Antarctica at dusk directly in Blender (Cycles: translucent white-blue ice, the sun setting on the horizon) ->
        OUT/render/%04d.png.  layout: grey clay (Workbench) for Layout-To-Render -> OUT/layout/%04d.png. Both: OUT/scene.blend.
"""
import argparse, json, math, os, random, sys

import bpy
from mathutils import Vector, noise

ap = argparse.ArgumentParser()
ap.add_argument("curves")
ap.add_argument("out")
ap.add_argument("--mode", default="render", choices=["render", "layout"])
ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--gpu", action="store_true")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
out = os.path.abspath(a.out)
C = json.load(open(a.curves))
N = a.frames or len(C["energy"])
rnd = random.Random(a.seed)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = int(C["fps"]); sc.frame_start, sc.frame_end = 1, N
W, H = (int(v) for v in a.res.split("x"))
sc.render.resolution_x, sc.render.resolution_y = W, H


def band(name, key="energy"):
    v = C["bands"][name][key]
    return [v[min(i, len(v) - 1)] for i in range(N)]


def smooth(v, k):
    out_, acc = [], v[0]
    for x in v:
        acc += (x - acc) * k; out_.append(acc)
    return out_


# --- the camera's path through the music ------------------------------------------------------------------------
low, mid, high = smooth(band("low"), 0.25), smooth(band("mid"), 0.08), band("high")
surge = [0.0] * N                                      # each low transient: a forward surge that decays over ~0.5 s
for f in C["bands"]["low"]["hits"]:
    for i in range(f, min(N, f + 14)):
        surge[i] += 1.6 * math.exp(-(i - f) / 4)
speed = [2.2 + 3.0 * low[i] + surge[i] for i in range(N)]   # metres per second
dist = [0.0]
for i in range(1, N):
    dist.append(dist[-1] + speed[i] / C["fps"])
L = dist[-1] + 40


def path(s):                    # a gently meandering line through the ice field, distance s along it
    return Vector((6 * math.sin(s / 23) + 3 * math.sin(s / 9.5), s, 0))


# --- the ice sheet: rolling, wind-carved, with pressure ridges --------------------------------------------------
def height(x, y):
    p = Vector((x * 0.035, y * 0.035, 0.3))
    h = 1.6 * noise.noise(p) + 0.6 * noise.noise(p * 3.1)
    ridge = 1 - abs(noise.noise(Vector((x * 0.012, y * 0.05, 1.7))))      # pressure ridges, across the path
    h += 2.2 * max(0, ridge - 0.75) / 0.25
    d = abs(x - path(y).x)                                                  # keep the corridor walkable
    h += 0.25 * noise.noise(Vector((x * 0.4, y * 0.12, 3.3)))            # sastrugi: wind-carved ridges
    return h * min(1, 0.5 + d / 18)


def grid(name, x0, x1, y0, y1, n, fn):
    import bmesh
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    vs = [[bm.verts.new((x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * j / n, 0)) for i in range(n + 1)] for j in range(n + 1)]
    for row in vs:
        for v in row:
            v.co.z = fn(v.co.x, v.co.y)
    for j in range(n):
        for i in range(n):
            bm.faces.new((vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]))
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    for p in me.polygons:
        p.use_smooth = True
    return ob


sheet = grid("ice_sheet", -140, 140, -60, L + 220, 260, height)


# --- ice spires and shards: tall, sharp, leaning crystals, thicker near the path but never on it ----------------
def spire(name, base, h, r, sides, lean, rot):
    import bmesh
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    ring = [bm.verts.new((r * math.cos(2 * math.pi * k / sides) * rnd.uniform(0.7, 1.2),
                          r * math.sin(2 * math.pi * k / sides) * rnd.uniform(0.7, 1.2), -1.0)) for k in range(sides)]
    mid_ = [bm.verts.new((v.co.x * 0.75, v.co.y * 0.75, h * rnd.uniform(0.45, 0.65))) for v in ring]
    tip = bm.verts.new((rnd.uniform(-0.3, 0.3) * r, rnd.uniform(-0.3, 0.3) * r, h))
    bm.faces.new(list(reversed(ring)))
    for k in range(sides):
        bm.faces.new((ring[k], ring[(k + 1) % sides], mid_[(k + 1) % sides], mid_[k]))
        bm.faces.new((mid_[k], mid_[(k + 1) % sides], tip))
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    ob.location = base; ob.rotation_euler = (lean[0], lean[1], rot)
    return ob


spires = []
y = 4.0
while y < L + 160:
    for side in (-1, 1):
        if rnd.random() < 0.8:
            px = path(y).x + side * rnd.uniform(4.5, 30)
            hgt = rnd.uniform(2, 9) * (1 + 1.5 * (abs(px - path(y).x) > 15))
            spires.append(spire(f"spire{len(spires)}", Vector((px, y, height(px, y))), hgt, rnd.uniform(0.6, 2.2),
                                rnd.choice((5, 6, 7)), (rnd.uniform(-0.25, 0.25), rnd.uniform(-0.25, 0.25)),
                                rnd.uniform(0, 6.3)))
            if rnd.random() < 0.5:                  # a cluster of smaller shards at its foot
                for _ in range(rnd.randint(2, 5)):
                    q = spires[-1].location + Vector((rnd.uniform(-3, 3), rnd.uniform(-3, 3), 0))
                    q.z = height(q.x, q.y)
                    spires.append(spire(f"shard{len(spires)}", q, rnd.uniform(0.6, 2.5), rnd.uniform(0.2, 0.7),
                                        rnd.choice((4, 5, 6)), (rnd.uniform(-0.6, 0.6), rnd.uniform(-0.6, 0.6)),
                                        rnd.uniform(0, 6.3)))
    y += rnd.uniform(3, 7)
for k in range(14):                                 # distant ice cliffs and tabular bergs on the horizon
    yy = L + rnd.uniform(60, 200); xx = rnd.uniform(-120, 120)
    spires.append(spire(f"cliff{k}", Vector((xx, yy, 0)), rnd.uniform(12, 35), rnd.uniform(10, 30), rnd.choice((6, 7, 8)),
                        (0, 0), rnd.uniform(0, 6.3)))

# --- camera -------------------------------------------------------------------------------------------------------
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 24; cam.data.clip_end = 2000
for i in range(N):
    s = dist[i]
    p = path(s)
    p.z = height(p.x, s) + 1.6 + 1.2 * mid[i]                       # rises a little with the mids
    look = path(s + 12); look.z = height(look.x, s + 12) + 1.0
    sway = 0.18 * (mid[i] - 0.5) * math.sin(i / 11)                  # the mids sway the view
    look.x += 6 * sway
    cam.location = p
    cam.rotation_euler = (look - p).to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.y += 0.04 * math.sin(i / 17) * (0.5 + mid[i])  # a slight roll
    cam.data.lens = 24 - 3 * min(1, surge[i])                         # a surge widens the view a touch
    cam.keyframe_insert("location", frame=i + 1); cam.keyframe_insert("rotation_euler", frame=i + 1)
    cam.data.keyframe_insert("lens", frame=i + 1)

# --- look ---------------------------------------------------------------------------------------------------------
if a.mode == "layout":                  # grey clay for Layout-To-Render: shape and motion only
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"; sc.display.shading.color_type = "SINGLE"
    sc.display.shading.single_color = (0.75, 0.75, 0.75)
    sc.display.shading.show_shadows = True; sc.display.shading.show_cavity = True
    sc.world = bpy.data.worlds.new("w"); sc.world.color = (0.05, 0.05, 0.05)
else:                                   # Antarctica at dusk, rendered
    sc.render.engine = "CYCLES"; sc.cycles.samples = a.samples; sc.cycles.use_denoising = True
    if a.gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for t_ in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = t_; prefs.get_devices()
                if any(d.type == t_ for d in prefs.devices):
                    break
            except TypeError:
                continue
        for d in prefs.devices:
            d.use = d.type != "CPU"
        sc.cycles.device = "GPU"
    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"
    w = bpy.data.worlds.new("dusk"); sc.world = w; w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky"); sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(3.0); sky.sun_rotation = math.radians(-12)  # the sun on the horizon, ahead
    sky.altitude = 50; sky.air_density = 1.4; sky.dust_density = 2.0; sky.sun_size = math.radians(1.2)
    bgn = nt.nodes.new("ShaderNodeBackground"); bgn.inputs["Strength"].default_value = 0.35
    o = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs["Color"], bgn.inputs["Color"]); nt.links.new(bgn.outputs["Background"], o.inputs["Surface"])
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.color = (1.0, 0.55, 0.3); sun.data.angle = math.radians(1.2)
    sun.rotation_euler = (math.radians(88.5), 0, math.radians(180))      # low, from far ahead (+y), grazing the ice
    for i in range(N):                                                  # the highs make the low sun glint
        sun.data.energy = 3.0 + 4.0 * high[i] ** 2
        sun.data.keyframe_insert("energy", frame=i + 1)

    def ice(name, tint, rough, trans):
        m = bpy.data.materials.new(name); m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*tint, 1)
        b.inputs["Roughness"].default_value = rough
        b.inputs["IOR"].default_value = 1.31
        b.inputs["Transmission Weight"].default_value = trans
        b.inputs["Subsurface Weight"].default_value = 0.4
        b.inputs["Subsurface Radius"].default_value = (0.4, 0.8, 1.6)
        b.inputs["Subsurface Scale"].default_value = 0.3
        b.inputs["Coat Weight"].default_value = 0.6
        return m
    sheet_m = ice("ice_sheet", (0.82, 0.9, 1.0), 0.5, 0.1)
    b = sheet_m.node_tree.nodes["Principled BSDF"]; nt_ = sheet_m.node_tree
    b.inputs["Coat Weight"].default_value = 0.15
    tc = nt_.nodes.new("ShaderNodeTexCoord")
    nz = nt_.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 2.5; nz.inputs["Detail"].default_value = 10
    vo = nt_.nodes.new("ShaderNodeTexVoronoi"); vo.feature = "DISTANCE_TO_EDGE"; vo.inputs["Scale"].default_value = 0.35
    nt_.links.new(tc.outputs["Object"], nz.inputs["Vector"]); nt_.links.new(tc.outputs["Object"], vo.inputs["Vector"])
    cr = nt_.nodes.new("ShaderNodeMapRange")                             # cracks: thin lines at the cell edges
    cr.inputs["From Min"].default_value, cr.inputs["From Max"].default_value = 0.0, 0.012
    wob = nt_.nodes.new("ShaderNodeTexNoise"); wob.inputs["Scale"].default_value = 0.6     # irregular, not a honeycomb
    add = nt_.nodes.new("ShaderNodeVectorMath"); add.operation = "ADD"
    sca = nt_.nodes.new("ShaderNodeVectorMath"); sca.operation = "SCALE"; sca.inputs["Scale"].default_value = 3.0
    nt_.links.new(tc.outputs["Object"], wob.inputs["Vector"]); nt_.links.new(wob.outputs["Color"], sca.inputs[0])
    nt_.links.new(tc.outputs["Object"], add.inputs[0]); nt_.links.new(sca.outputs["Vector"], add.inputs[1])
    nt_.links.new(add.outputs["Vector"], vo.inputs["Vector"])
    nt_.links.new(vo.outputs["Distance"], cr.inputs["Value"])
    mx = nt_.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"
    mx.inputs["A"].default_value = (0.55, 0.74, 0.92, 1); mx.inputs["B"].default_value = (0.84, 0.91, 1.0, 1)
    nt_.links.new(cr.outputs["Result"], mx.inputs["Factor"]); nt_.links.new(mx.outputs["Result"], b.inputs["Base Color"])
    bump = nt_.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.35
    nt_.links.new(nz.outputs["Fac"], bump.inputs["Height"]); nt_.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    spire_m = ice("ice_spire", (0.78, 0.9, 1.0), 0.04, 0.95)
    ab = spire_m.node_tree.nodes.new("ShaderNodeVolumeAbsorption")      # blue deep inside: translucent white-blue ice
    ab.inputs["Color"].default_value = (0.45, 0.75, 1.0, 1); ab.inputs["Density"].default_value = 0.35
    spire_m.node_tree.links.new(ab.outputs["Volume"], spire_m.node_tree.nodes["Material Output"].inputs["Volume"])
    cliff_m = ice("ice_cliff", (0.85, 0.92, 1.0), 0.15, 0.6)               # big pieces: lighter, or absorption turns
    ab2 = cliff_m.node_tree.nodes.new("ShaderNodeVolumeAbsorption")         # them cobalt (it grows with thickness)
    ab2.inputs["Color"].default_value = (0.6, 0.82, 1.0, 1); ab2.inputs["Density"].default_value = 0.03
    cliff_m.node_tree.links.new(ab2.outputs["Volume"], cliff_m.node_tree.nodes["Material Output"].inputs["Volume"])
    sheet.data.materials.append(sheet_m)
    for s_ in spires:
        big = max(s_.dimensions) > 12
        s_.data.materials.append(cliff_m if big else spire_m)
        for p in s_.data.polygons:
            p.use_smooth = False
    sc.cycles.max_bounces = 8; sc.cycles.transmission_bounces = 8; sc.cycles.glossy_bounces = 4

os.makedirs(out, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=f"{out}/scene_{a.mode}.blend")
sc.render.filepath = f"{out}/{a.mode}/"
bpy.ops.render.render(animation=True)
print("ice ok", out, N, "frames")
