"""A procedural ice landscape the camera travels through in time with the music (2026-10-26, Ice Walk proof of concept:
an abstract, lyric-free piece). The geometry is generated here (a wind-carved ice sheet, pressure ridges, translucent ice
spires and distant ice cliffs); the camera's motion comes from music_curves.py's spectral bands: the lows push it
forward in surges, the mids sway and turn it, the highs make the low sun glint in the ice.

    docker run --rm [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all] --user $(id -u):$(id -g) -v $PWD:/work -w /work \
        blender -b -P pipeline/world/ice_world.py -- CURVES.json OUT [--mode render|layout] [--frames N] [--seed S]
        [--res 1920x1080] [--samples 64] [--gpu]
Motion (POC 2, 5): forward only, a slow base speed surging with the low band like breathing; a steady camera.
--theme ice|chrome|icechrome, --snow N flakes.
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
ap.add_argument("--breath", default="0.8,6.0", help="BASE,GAIN: speed (m/s) = BASE + GAIN x the low band's breath, "
                "scaled 0..1 within the clip (POC 5: 0.8,6.0, about 1 to 6.5 m/s on each swell)")
ap.add_argument("--breath-norm", default="window", choices=["window", "song"], help="window: the breath envelope is "
                "stretched to its own 5-95th percentile in the clip and curved (^1.5), so troughs are slow and swells surge "
                "(user on POC 4, 2026-10-27: 'not seeing much variation in speed'); song: as POC 1-4")
ap.add_argument("--snow", type=int, default=0, help="snow flurries: this many flakes drifting through the corridor, "
                "gusting sideways with the highs")
ap.add_argument("--theme", default="ice", choices=["ice", "chrome", "icechrome"], help="chrome: the same world in polished chrome, "
                "the ground rippled liquid metal (user, 2026-10-27, after the Liquid Mercury universe). icechrome: the ice "
                "with mercury frozen into it: chrome cores inside the crystals, chrome veins in the sheet's cracks, some "
                "formations solid chrome (user, 2026-10-27: 'a combination of ice and chrome somehow')")
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
# Forward only, breathing (user review of POC 1, 2026-10-26: "utterly nausea-inducing ... accelerated a bit by
# low-frequency sounds (which sound a bit like breathing) ... NO movement backwards"). The low band's loudness, through
# an envelope that rises quickly and releases slowly, eases a slow base speed up; nothing else moves the camera.
high = band("high")
low = band("low")
env, e = [], (low[0] if a.breath_norm == "window" else 0.0)
for x in low:
    e += (x - e) * (0.30 if x > e else 0.04)        # breath: quick in, slow out
    env.append(e)
if a.breath_norm == "window":                       # the swells, not the song's overall loudness, set the range
    q = sorted(env); lo_, hi_ = q[int(0.05 * (N - 1))], q[int(0.95 * (N - 1))]
    env = [min(1, max(0, (v - lo_) / (hi_ - lo_ + 1e-9))) ** 1.5 for v in env]
B0, BG = (float(v) for v in a.breath.split(","))
speed = smooth([B0 + BG * v for v in env], 0.15)    # metres per second, never below B0: never backwards
dist = [0.0]
for i in range(1, N):
    dist.append(dist[-1] + speed[i] / C["fps"])
L = dist[-1] + 40


def path(s):                    # a long, slow curve through the ice field, distance s along it
    return Vector((5 * math.sin(s / 45), s, 0))


# --- the ice sheet: rolling, wind-carved, with pressure ridges --------------------------------------------------
def height(x, y):
    p = Vector((x * 0.035, y * 0.035, 0.3))
    h = 1.6 * noise.noise(p) + 0.6 * noise.noise(p * 3.1)
    ridge = 1 - abs(noise.noise(Vector((x * 0.012, y * 0.05, 1.7))))      # pressure ridges, across the path:
    h += 3.0 * max(0, ridge - 0.7) ** 2 / 0.09                              # sharp-crested, not smooth swells
    h += 0.5 * max(0, noise.noise(Vector((x * 0.25, y * 0.25, 5.1))))      # broken, lumpy ice between them
    d = abs(x - path(y).x)                                                  # keep the corridor open
    h += 0.25 * noise.noise(Vector((x * 0.4, y * 0.12, 3.3)))            # sastrugi: wind-carved ridges
    return h * min(1, 0.35 + d / 14)


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


def box(name, base, size, rot):  # a slab or a serac: a box, its top edges knocked off unevenly
    import bmesh
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= size[0]; v.co.y *= size[1]; v.co.z = (v.co.z + 0.5) * size[2] - 0.6
        if v.co.z > 0:
            v.co.z *= rnd.uniform(0.7, 1.15)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    ob.location = base; ob.rotation_euler = rot
    return ob


def mound(name, base, r, h):    # an ice hummock the formations grow out of, in the sheet's own ice
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1, location=base)
    ob = bpy.context.object; ob.name = name
    ob.scale = (r * rnd.uniform(0.8, 1.2), r * rnd.uniform(0.8, 1.2), h)
    ob.location.z -= h * 0.45
    for v in ob.data.vertices:              # lumpy, not a dome
        v.co *= 1 + 0.18 * noise.noise(v.co * 2.3 + Vector((base.x, base.y, 0)) * 0.1)
    return ob


# Denser and more integrated than POC 1 ("not nearly enough geometry ... very sharp spikes and some very smooth areas.
# Maybe they can be a bit more integrated?"): clusters that grow out of hummocks, broken slabs tilted out of the ice,
# blocky seracs, and only a few needles; every few metres on both sides, the corridor kept open.
spires, mounds = [], []
y = 3.0
while y < L + 160:
    for side in (-1, 1):
        for _ in range(rnd.randint(1, 3)):
            off = rnd.uniform(3.5, 34)
            px = path(y).x + side * off; py = y + rnd.uniform(-1.5, 1.5)
            base = Vector((px, py, height(px, py)))
            big = 1 + 1.2 * (off > 14)
            kind = rnd.random()
            r0 = rnd.uniform(1.0, 2.6) * big
            mounds.append(mound(f"mound{len(mounds)}", base, r0 * 1.6, rnd.uniform(0.5, 1.4) * big))
            if kind < 0.35:                     # a cluster of leaning crystals out of the hummock
                for c_ in range(rnd.randint(3, 7)):
                    q = base + Vector((rnd.uniform(-r0, r0), rnd.uniform(-r0, r0), 0)); q.z = height(q.x, q.y)
                    spires.append(spire(f"crystal{len(spires)}", q, rnd.uniform(1.2, 4.5) * big, rnd.uniform(0.3, 0.9) * big,
                                        rnd.choice((5, 6, 7)), (rnd.uniform(-0.45, 0.45), rnd.uniform(-0.45, 0.45)),
                                        rnd.uniform(0, 6.3)))
            elif kind < 0.65:                   # broken slabs tilted up out of the ice, in a row
                ang = rnd.uniform(0, 6.3)
                for c_ in range(rnd.randint(2, 5)):
                    q = base + Vector((math.cos(ang) * c_ * 1.6, math.sin(ang) * c_ * 1.6, 0)); q.z = height(q.x, q.y)
                    spires.append(box(f"slab{len(spires)}", q, (rnd.uniform(1.2, 3.0) * big, rnd.uniform(0.25, 0.6),
                                      rnd.uniform(1.0, 3.2) * big),
                                      (rnd.uniform(0.3, 1.1), rnd.uniform(-0.3, 0.3), ang + rnd.uniform(-0.4, 0.4))))
            elif kind < 0.9:                    # blocky seracs
                for c_ in range(rnd.randint(1, 3)):
                    q = base + Vector((rnd.uniform(-r0, r0), rnd.uniform(-r0, r0), 0)); q.z = height(q.x, q.y)
                    w = rnd.uniform(1.0, 2.8) * big
                    spires.append(box(f"serac{len(spires)}", q, (w, w * rnd.uniform(0.6, 1.2), rnd.uniform(1.5, 5) * big),
                                      (rnd.uniform(-0.15, 0.15), rnd.uniform(-0.15, 0.15), rnd.uniform(0, 6.3))))
            else:                               # now and then a tall needle
                spires.append(spire(f"needle{len(spires)}", base, rnd.uniform(5, 11) * big, rnd.uniform(0.6, 1.4),
                                    rnd.choice((5, 6)), (rnd.uniform(-0.2, 0.2), rnd.uniform(-0.2, 0.2)), rnd.uniform(0, 6.3)))
    y += rnd.uniform(2.0, 4.0)
for k in range(14):                                 # distant ice cliffs and tabular bergs on the horizon
    yy = L + rnd.uniform(60, 200); xx = rnd.uniform(-120, 120)
    spires.append(spire(f"cliff{k}", Vector((xx, yy, 0)), rnd.uniform(12, 35), rnd.uniform(10, 30), rnd.choice((6, 7, 8)),
                        (0, 0), rnd.uniform(0, 6.3)))

# --- camera: steady -----------------------------------------------------------------------------------------------
# No roll, no sway, no lens change, no bobbing over every bump (POC 1's motion was nauseating): eye height above a
# heavily smoothed ground, looking far ahead along the slow curve.
def ground(s_, span=10.0, n=11):
    return sum(height(path(s_ + span * (k / (n - 1) - 0.5)).x, s_ + span * (k / (n - 1) - 0.5)) for k in range(n)) / n


cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 26; cam.data.clip_end = 2000
zs = smooth([ground(d) + 1.8 for d in dist], 0.08)
for i in range(N):
    s_ = dist[i]
    p = path(s_); p.z = zs[i]
    look = path(s_ + 30); look.z = zs[i] - 0.6
    cam.location = p
    cam.rotation_euler = (look - p).to_track_quat("-Z", "Y").to_euler()
    cam.keyframe_insert("location", frame=i + 1); cam.keyframe_insert("rotation_euler", frame=i + 1)

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
    for m_ in mounds:
        m_.data.materials.append(sheet_m)
    for s_ in spires:
        big = max(s_.dimensions) > 12
        s_.data.materials.append(cliff_m if big else spire_m)
        for p in s_.data.polygons:
            p.use_smooth = False
    if a.theme == "chrome":            # every surface mirror-polished metal, the ground a rippled liquid
        def chrome(name, rough, tint=(0.92, 0.93, 0.95)):
            m = bpy.data.materials.new(name); m.use_nodes = True
            b = m.node_tree.nodes["Principled BSDF"]
            b.inputs["Base Color"].default_value = (*tint, 1); b.inputs["Metallic"].default_value = 1.0
            b.inputs["Roughness"].default_value = rough
            return m
        ground = chrome("liquid_chrome", 0.03)
        nt_ = ground.node_tree; b = nt_.nodes["Principled BSDF"]
        tc = nt_.nodes.new("ShaderNodeTexCoord")
        wv = nt_.nodes.new("ShaderNodeTexWave"); wv.wave_type = "RINGS"; wv.inputs["Scale"].default_value = 0.15
        wv.inputs["Distortion"].default_value = 6.0; wv.inputs["Detail"].default_value = 4.0
        nt_.links.new(tc.outputs["Object"], wv.inputs["Vector"])
        bump = nt_.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.25
        nt_.links.new(wv.outputs["Fac"], bump.inputs["Height"]); nt_.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        solid = chrome("chrome", 0.06); brushed = chrome("chrome_brushed", 0.22, (0.85, 0.87, 0.92))
        for ob in [sheet] + mounds + spires:
            ob.data.materials.clear()
            ob.data.materials.append(ground if ob is sheet or ob in mounds else (brushed if rnd.random() < 0.3 else solid))
    if a.theme == "icechrome":         # mercury frozen into the ice
        def chrome(name, rough):
            m = bpy.data.materials.new(name); m.use_nodes = True
            b = m.node_tree.nodes["Principled BSDF"]
            b.inputs["Base Color"].default_value = (0.92, 0.93, 0.95, 1); b.inputs["Metallic"].default_value = 1.0
            b.inputs["Roughness"].default_value = rough
            return m
        solid = chrome("chrome", 0.05)
        nt_ = sheet_m.node_tree; b = nt_.nodes["Principled BSDF"]     # the sheet's cracks run with chrome
        inv = nt_.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
        nt_.links.new(nt_.nodes["Map Range"].outputs["Result"], inv.inputs[1])
        nt_.links.new(inv.outputs["Value"], b.inputs["Metallic"])
        for s_ in list(spires):
            if max(s_.dimensions) > 12:
                continue
            if rnd.random() < 0.3:                                      # some formations solid chrome
                s_.data.materials.clear(); s_.data.materials.append(solid)
            else:                                                       # a chrome core inside the ice
                core = s_.copy(); core.data = s_.data.copy(); sc.collection.objects.link(core)
                core.scale = (0.45, 0.45, 0.7)
                core.data.materials.clear(); core.data.materials.append(solid)
    if a.snow:                         # snow flurries: flakes drift through the corridor, gusting sideways on the highs
        # Organic, not CGI (user, 2026-10-27: "like it's an alien planet rather than CGI"): ragged flakes of very different
        # sizes, the nearest out of focus, and spindrift: fine snow streaming low over the ice.
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1)
        flake = bpy.context.object; flake.name = "flake"; flake.location = (0, -500, -50)
        for v in flake.data.vertices:          # a ragged, flattened clump, not a ball
            v.co.z *= 0.35; v.co *= rnd.uniform(0.6, 1.3)
        fm = bpy.data.materials.new("snow"); fm.use_nodes = True
        fb = fm.node_tree.nodes["Principled BSDF"]
        fb.inputs["Base Color"].default_value = (1, 1, 1, 1); fb.inputs["Roughness"].default_value = 0.6
        fb.inputs["Subsurface Weight"].default_value = 0.5
        fb.inputs["Emission Color"].default_value = (1.0, 0.82, 0.7, 1); fb.inputs["Emission Strength"].default_value = 0.6
        flake.data.materials.append(fm)

        def flurry(name, z0, z1, width, count, size, rand, mass):
            bpy.ops.mesh.primitive_cube_add(size=1)
            em = bpy.context.object; em.name = name
            em.location = (0, (L - 30) / 2, (z0 + z1) / 2); em.scale = (width, L + 30, z1 - z0)
            ps = em.modifiers.new(name, "PARTICLE_SYSTEM").particle_system.settings
            em.show_instancer_for_render = False      # hide_render on the emitter would hide its flakes too
            ps.count = count; ps.frame_start = ps.frame_end = 0; ps.lifetime = N + 10
            ps.emit_from = "VOLUME"; ps.distribution = "RAND"; ps.normal_factor = 0
            ps.physics_type = "NEWTON"; ps.mass = mass; ps.drag_factor = 0.5; ps.brownian_factor = 0.6
            ps.effector_weights.gravity = 0.04
            ps.render_type = "OBJECT"; ps.instance_object = flake; ps.use_rotations = True
            ps.rotation_mode = "NONE"; ps.rotation_factor_random = 1.0; ps.angular_velocity_mode = "RAND"
            ps.angular_velocity_factor = 3.0
            ps.particle_size = size; ps.size_random = rand
            em.particle_systems[0].point_cache.frame_start = 0; em.particle_systems[0].point_cache.frame_end = N
        flurry("flurry", 0.3, 7.0, 16, a.snow, 0.035, 0.9, 0.02)          # falling flakes, 1-6 cm
        flurry("spindrift", 0.0, 0.5, 30, a.snow // 2, 0.008, 0.8, 0.005)  # fine snow streaming over the ground
        cam.data.dof.use_dof = True; cam.data.dof.focus_distance = 14; cam.data.dof.aperture_fstop = 2.0
        wind = bpy.data.objects.new("wind", None); sc.collection.objects.link(wind)
        wind.empty_display_type = "SINGLE_ARROW"; bpy.context.view_layer.objects.active = wind
        bpy.ops.object.forcefield_toggle(); wind.field.type = "WIND"
        wind.rotation_euler = (0, math.radians(90), math.radians(-15))  # blowing across the path, a little towards us
        for i in range(N):
            wind.field.strength = 0.08 + 0.5 * high[i] ** 2
            wind.field.keyframe_insert("strength", frame=i + 1)
        sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
    sc.cycles.max_bounces = 8; sc.cycles.transmission_bounces = 8
    sc.cycles.glossy_bounces = 8 if a.theme in ("chrome", "icechrome") else 4

os.makedirs(out, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=f"{out}/scene_{a.mode}.blend")
sc.render.filepath = f"{out}/{a.mode}/"
bpy.ops.render.render(animation=True)
print("ice ok", out, N, "frames")
