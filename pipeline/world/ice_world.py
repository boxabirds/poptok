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
from mathutils import Matrix, Quaternion, Vector, noise

ap = argparse.ArgumentParser()
ap.add_argument("curves")
ap.add_argument("out")
ap.add_argument("--mode", default="render", choices=["render", "layout"])
ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--sun-el", type=float, default=1.5, help="sun elevation, degrees (higher: bluer sky, more blue light through the ice)")
ap.add_argument("--sky", default="2.2,0.7,0.22", help="AIR,DUST,STRENGTH of the Nishita sky: 2.2,0.7,0.22 is an orange dusk; "
                "1.4,1.0,0.35 a pale blue sky whose light, outweighing the low sun, turns the ice blue (POCs 1-4)")
ap.add_argument("--sun-energy", default="3,4", help="BASE,GAIN: sun lamp energy = BASE + GAIN x high band^2")
ap.add_argument("--mist-color", default="0.85,0.9,1.0")
ap.add_argument("--swifts", action="store_true", help="the tweets send swifts tearing across the sky: each draws a long, fast "
                "curve leaving a short fluffy contrail that spreads and fades")
ap.add_argument("--sun-color", default="1.0,0.55,0.3")
ap.add_argument("--exposure", type=float, default=0.0)
ap.add_argument("--look", default="AgX - Medium High Contrast")
ap.add_argument("--ice-absorb", type=float, default=0.35, help="blue absorption density inside the ice "
                "formations: higher, the sunlight through them comes out a deeper blue")
ap.add_argument("--layers", default="", help="music_layers.py JSON: the song's other layers, each with its own visual")
ap.add_argument("--layer-fx", default="chime,drone,pad,subbass,clicks", help="which of the layers to show")
ap.add_argument("--chime-glow", type=float, default=6.0, help="emission of a crystal lit by a chime note")
ap.add_argument("--halo", type=float, default=0.8, help="peak brightness of the drone's ice halo round the sun")
ap.add_argument("--cloud-sweep", type=float, default=45.0, help="metres the cloud shadows slide at each chord change")
ap.add_argument("--drift", type=float, default=0.25, help="peak density of the sub-bass's low ground drift")
ap.add_argument("--crack-glow", type=float, default=25.0, help="emission of a click's glinting crack")
ap.add_argument("--only", default="", help="render just these frames, e.g. 22,110 (quick tests)")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--breath", default="0.8,6.0", help="BASE,GAIN: speed (m/s) = BASE + GAIN x the low band's breath, "
                "scaled 0..1 within the clip (POC 5: 0.8,6.0, about 1 to 6.5 m/s on each swell)")
ap.add_argument("--breath-norm", default="window", choices=["window", "song"], help="window: the breath envelope is "
                "stretched to its own 5-95th percentile in the clip and curved (^1.5), so troughs are slow and swells surge "
                "(user on POC 4, 2026-10-27: 'not seeing much variation in speed'); song: as POC 1-4")
ap.add_argument("--sparkles", action="store_true", help="the tweets (music_curves 'tweets') flash as sparkles in the sky")
ap.add_argument("--mist", type=float, default=0, help="max density of a low mist that comes and goes with the mids "
                "(music_curves 'mist'); 0 = none, ~0.05 = thick")
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
# The clip is frames FO+1 .. FO+N of the scene; frames 1 .. FO are a run-up for the snow (a particle cache is clipped to
# the scene range, which cannot start before 0, so the run-up has to be inside it). Output files are numbered 1 .. N.
FO = 72
sc.render.fps = int(C["fps"]); sc.frame_start, sc.frame_end = 1, FO + N
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


# The low sun, in frame and unblocked (user, 2026-10-27: "we seem to have lost all detail in the sky too -- no sunlight
# coming through the ice"; POC 1's sun glowed on the horizon behind the crystals, but from POC 2 the denser ice and the
# distant cliffs hid it). Its azimuth (degrees, + = towards +x, as the Nishita sky's sun_rotation) is set from the clip's
# mean heading, a little left of centre; the sky's sun disc and the sun lamp share it; nothing tall stands in its way.
HEAD = sum(math.degrees(math.atan(5 / 45 * math.cos(d / 45))) for d in dist) / N
SUN_AZ = HEAD - 9
SUN_FROM = Vector((path(dist[-1]).x, dist[-1], 0))


def in_sun(x, y, wide=7):       # beyond the camera's last position and within `wide` degrees of the sun's bearing
    return y > dist[-1] + 8 and abs(math.degrees(math.atan2(x - SUN_FROM.x, y - SUN_FROM.y)) - SUN_AZ) < wide


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
            if off > 10 and in_sun(px, py):
                continue
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
    while in_sun(xx, yy, 12):
        xx = rnd.uniform(-120, 120)
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
    cam.keyframe_insert("location", frame=i + 1 + FO); cam.keyframe_insert("rotation_euler", frame=i + 1 + FO)

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
    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = a.look
    sc.view_settings.exposure = a.exposure
    w = bpy.data.worlds.new("dusk"); sc.world = w; w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky"); sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(a.sun_el); sky.sun_rotation = math.radians(SUN_AZ)  # the sun on the horizon, ahead
    AIR, DUST, SKY_K = (float(v) for v in a.sky.split(","))
    sky.altitude = 50; sky.air_density = AIR; sky.dust_density = DUST; sky.sun_size = math.radians(1.2)
    bgn = nt.nodes.new("ShaderNodeBackground"); bgn.inputs["Strength"].default_value = SKY_K
    o = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs["Color"], bgn.inputs["Color"]); nt.links.new(bgn.outputs["Background"], o.inputs["Surface"])
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.color = tuple(float(v) for v in a.sun_color.split(",")); sun.data.angle = math.radians(1.2)
    sun.rotation_euler = (math.radians(90 - a.sun_el), 0, math.radians(180 - SUN_AZ))      # low, from far ahead (+y), grazing the ice
    SUN_E = [float(v) for v in a.sun_energy.split(",")]
    for i in range(N):                                                  # the highs make the low sun glint
        sun.data.energy = SUN_E[0] + SUN_E[1] * high[i] ** 2
        sun.data.keyframe_insert("energy", frame=i + 1 + FO)

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
    ab.inputs["Color"].default_value = (0.45, 0.75, 1.0, 1); ab.inputs["Density"].default_value = a.ice_absorb
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
    if a.theme == "icechrome":         # a mix: separate ice and chrome formations side by side (user, 2026-10-27: "I
        # didn't want chrome in the ice, a mix. with chrome inside you don't get the light going through any more")
        def chrome(name, rough):
            m = bpy.data.materials.new(name); m.use_nodes = True
            b = m.node_tree.nodes["Principled BSDF"]
            b.inputs["Base Color"].default_value = (0.92, 0.93, 0.95, 1); b.inputs["Metallic"].default_value = 1.0
            b.inputs["Roughness"].default_value = rough
            return m
        solid = chrome("chrome", 0.05)
        for s_ in spires:
            if max(s_.dimensions) <= 12 and rnd.random() < 0.35:
                s_.data.materials.clear(); s_.data.materials.append(solid)
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
        fb.inputs["Emission Color"].default_value = (0.85, 0.9, 1.0, 1); fb.inputs["Emission Strength"].default_value = 0.8   # far flakes: snow, not grey specks against the sky
        flake.data.materials.append(fm)

        # Emitted continuously from upwind, from 3 s before the clip, each flake living ~4 s, so the snow never runs out
        # (POC 5 bug: all flakes born at frame 0 were blown out of the corridor within 2 s, user: "why does the snow stop?")
        PRE, LIFE = FO, 96

        def flurry(name, z0, z1, width, count, size, rand, mass):
            bpy.ops.mesh.primitive_cube_add(size=1)
            em = bpy.context.object; em.name = name
            em.location = (-6, (L - 30) / 2, (z0 + z1) / 2); em.scale = (width + 12, L + 30, z1 - z0)   # wider upwind (-x)
            ps = em.modifiers.new(name, "PARTICLE_SYSTEM").particle_system.settings
            em.show_instancer_for_render = False      # hide_render on the emitter would hide its flakes too
            ps.count = int(count * (N + PRE) / LIFE); ps.frame_start, ps.frame_end = 1 + FO - PRE, FO + N   # ~count alive at any time
            ps.lifetime = LIFE; ps.lifetime_random = 0.3
            ps.emit_from = "VOLUME"; ps.distribution = "RAND"; ps.normal_factor = 0
            ps.physics_type = "NEWTON"; ps.mass = mass; ps.drag_factor = 1.0; ps.brownian_factor = 0.6
            ps.effector_weights.gravity = 0.04
            ps.render_type = "OBJECT"; ps.instance_object = flake; ps.use_rotations = True
            ps.rotation_mode = "NONE"; ps.rotation_factor_random = 1.0; ps.angular_velocity_mode = "RAND"
            ps.angular_velocity_factor = 3.0
            ps.particle_size = size; ps.size_random = rand
            em.particle_systems[0].point_cache.frame_start = 1 + FO - PRE; em.particle_systems[0].point_cache.frame_end = FO + N
        flurry("flurry", 0.3, 7.0, 16, a.snow, 0.07, 0.9, 0.02)           # falling flakes, 2-13 cm
        flurry("spindrift", 0.0, 0.5, 30, a.snow // 2, 0.016, 0.8, 0.005)  # fine snow streaming over the ground
        cam.data.dof.use_dof = True; cam.data.dof.focus_distance = 14; cam.data.dof.aperture_fstop = 2.0
        wind = bpy.data.objects.new("wind", None); sc.collection.objects.link(wind)
        wind.empty_display_type = "SINGLE_ARROW"; bpy.context.view_layer.objects.active = wind
        bpy.ops.object.forcefield_toggle(); wind.field.type = "WIND"
        wind.rotation_euler = (0, math.radians(90), math.radians(-15))  # blowing across the path, a little towards us
        for i in range(N):
            wind.field.strength = 0.02 + 0.12 * high[i] ** 2
            wind.field.keyframe_insert("strength", frame=i + 1 + FO)
        sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
    sc.cycles.max_bounces = 8; sc.cycles.transmission_bounces = 8
    if a.swifts:                       # tweets -> swifts (user, 2026-10-27: "like swift birds making short fluffy contrails
        # in the sky ... they move very fast, so the curves would be long and fast-drawn"). Each tweet: an unseen swift
        # tears along a long swooping curve in ~1/3 s; behind it a short contrail of soft volume, which spreads and fades
        # as its tail catches up with the head.
        T = C["tweets"]; DRAW, LINGER, TAIL = 8, 30, 0.5
        BIRD = 1.65                                                   # swift size (x its distance / 80 m)
        # the swift at the head (user, 2026-10-27: "they're birds so should have a head of the contrail with a bird-like
        # shape"): a dark silhouette from below, nose +y, span 2.1 units: sickle wings swept back, a forked tail
        half = [(0.0, 0.55), (0.06, 0.45), (0.08, 0.15), (0.35, 0.12), (0.65, 0.0), (0.9, -0.2), (1.05, -0.45),
                (0.8, -0.3), (0.5, -0.15), (0.2, -0.08), (0.07, -0.2), (0.12, -0.62), (0.0, -0.42)]
        ring = half + [(-x, y) for x, y in reversed(half[1:-1])]
        bm_ = bpy.data.meshes.new("swift"); bm_.from_pydata([(x, y, 0) for x, y in ring], [], [list(range(len(ring)))])
        bm_.update()
        bird_m = bpy.data.materials.new("swift"); bird_m.use_nodes = True
        bb = bird_m.node_tree.nodes["Principled BSDF"]
        bb.inputs["Base Color"].default_value = (0.03, 0.03, 0.035, 1); bb.inputs["Roughness"].default_value = 0.7
        bm_.materials.append(bird_m)
        tp_cu = bpy.data.curves.new("contrail_taper", "CURVE")       # radius along the visible contrail: 0 at the old
        tsp = tp_cu.splines.new("POLY"); prof = [(0, 0.0), (0.2, 0.55), (0.55, 1.0), (0.85, 0.75), (1.0, 0.12)]
        tsp.points.add(len(prof) - 1)                                 # tail, thin where it leaves the bird
        for j, (x, y) in enumerate(prof):
            tsp.points[j].co = (x, y, 0, 1)
        taper = bpy.data.objects.new("contrail_taper", tp_cu); sc.collection.objects.link(taper); taper.hide_render = True
        for k, (h, st) in enumerate(zip(T["hits"], T["hit_strength"])):
            if h >= N:
                continue
            c0 = path(dist[h]); c0.z = zs[h]
            fw = path(dist[h] + 30) - c0; fw.z = 0; fw.normalize()
            rt = Vector((fw.y, -fw.x, 0)); up = Vector((0, 0, 1))
            D = rnd.uniform(55, 110)                                  # how far ahead it flies
            ctr = c0 + fw * D + rt * rnd.uniform(-0.4, 0.4) * D + up * rnd.uniform(0.12, 0.3) * D
            th, k1, k2 = rnd.uniform(0, 2 * math.pi), rnd.uniform(-2.5, 2.5), rnd.uniform(-2.5, 2.5)
            Lp, M = rnd.uniform(0.7, 1.2) * D, 48                    # a swoop: the heading bends one way, then may whip back
            uv, u, v = [], 0.0, 0.0
            for j in range(M):
                t = j / (M - 1); hd = th + k1 * t + k2 * t * t
                u += math.cos(hd) * Lp / M; v += math.sin(hd) * Lp / M; uv.append((u, v))
            mu, mv = sum(p[0] for p in uv) / M, sum(p[1] for p in uv) / M
            P = [ctr + rt * (p[0] - mu) + up * 0.55 * (p[1] - mv) + fw * 0.5 * (p[1] - mv) for p in uv]
            lift = max(0.0, c0.z + 5 - min(p.z for p in P))           # never below the horizon line
            cu = bpy.data.curves.new(f"swift{k}", "CURVE"); cu.dimensions = "3D"
            sp = cu.splines.new("POLY"); sp.points.add(M - 1)
            P = [p + up * lift for p in P]
            for j, p in enumerate(P):
                sp.points[j].co = (p.x, p.y, p.z, 1)
            cu.bevel_resolution = 3; cu.use_fill_caps = True
            cu.taper_object = taper; cu.use_map_taper = True
            cum = [0.0]
            for j in range(1, M):
                cum.append(cum[-1] + (P[j] - P[j - 1]).length)

            def along(e_):                                            # point and heading at arc-length fraction e_
                s = e_ * cum[-1]
                j = max(1, min(M - 1, next((q for q in range(1, M) if cum[q] >= s), M - 1)))
                t_ = (s - cum[j - 1]) / max(1e-6, cum[j] - cum[j - 1])
                return P[j - 1].lerp(P[j], t_), (P[j] - P[j - 1]).normalized()
            bird = bpy.data.objects.new(f"swift_bird{k}", bm_); sc.collection.objects.link(bird)
            bird.scale = (BIRD * D / 80,) * 3; bird.visible_shadow = False
            bird.cycles.use_motion_blur = False   # ~11 m per frame: blurred, it smeared into a faint streak
            flap0 = rnd.uniform(0, 6.3)
            cu.bevel_factor_mapping_start = cu.bevel_factor_mapping_end = "SPLINE"
            ob = bpy.data.objects.new(f"swift{k}", cu); sc.collection.objects.link(ob)
            ob.visible_shadow = False
            m = bpy.data.materials.new(f"contrail{k}"); m.use_nodes = True; nt_ = m.node_tree; nt_.nodes.clear()
            pv = nt_.nodes.new("ShaderNodeVolumePrincipled")
            pv.inputs["Color"].default_value = (0.92, 0.95, 1.0, 1); pv.inputs["Anisotropy"].default_value = 0.5
            tc = nt_.nodes.new("ShaderNodeTexCoord")
            nz = nt_.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 3.0; nz.inputs["Detail"].default_value = 8
            nz.inputs["Roughness"].default_value = 0.65
            mr = nt_.nodes.new("ShaderNodeMapRange")                  # puffs, not a smooth tube
            mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.35, 0.75
            dens = nt_.nodes.new("ShaderNodeMath"); dens.operation = "MULTIPLY"
            out_ = nt_.nodes.new("ShaderNodeOutputMaterial")
            nt_.links.new(tc.outputs["Object"], nz.inputs["Vector"]); nt_.links.new(nz.outputs["Fac"], mr.inputs["Value"])
            nt_.links.new(mr.outputs["Result"], dens.inputs[0]); nt_.links.new(dens.outputs["Value"], pv.inputs["Density"])
            glow = nt_.nodes.new("ShaderNodeMath"); glow.operation = "MULTIPLY"; glow.inputs[1].default_value = 1.5
            pv.inputs["Emission Color"].default_value = (0.95, 0.97, 1.0, 1)  # self-lit a little: in its own shadow it
            nt_.links.new(dens.outputs["Value"], glow.inputs[0])            # read as a dark line against the sky
            nt_.links.new(glow.outputs["Value"], pv.inputs["Emission Strength"])
            nt_.links.new(pv.outputs["Volume"], out_.inputs["Volume"])
            cu.materials.append(m)
            base = 1.2 * (0.5 + 0.5 * st)
            for i in range(max(0, h - 1), min(N, h + DRAW + LINGER + 2)):
                age = i - h
                if age < 0:
                    e_, s_, fade, grow = 0.0, 0.0, 1.0, 0.0
                elif age <= DRAW:
                    e_ = age / DRAW; s_ = max(0.0, e_ - TAIL); fade, grow = 1.0, age / (DRAW + LINGER)
                else:
                    q = min(1.0, (age - DRAW) / LINGER)
                    e_, s_ = 1.0, min(1.0, 1 - TAIL + TAIL * q); fade, grow = 1 - q, age / (DRAW + LINGER)
                cu.bevel_factor_end, cu.bevel_factor_start = e_, s_
                if 0 <= age <= DRAW + 6:
                    pos, hd_ = along(min(1.0, e_))
                    pos = pos + hd_ * max(0, age - DRAW) * cum[-1] / DRAW   # past the end: on at the same speed
                    bird.location = pos
                    # wings turned to the camera (laid flat they were edge-on from the ground: invisible), rolling a
                    # little through the turns
                    cp_ = path(dist[i]); cp_.z = zs[i]
                    nz_ = (cp_ - pos); nz_ = (nz_ - hd_ * nz_.dot(hd_)).normalized()
                    q_ = Matrix((hd_.cross(nz_), hd_, nz_)).transposed().to_quaternion()
                    q_ = q_ @ Quaternion((0, 1, 0), 0.45 * math.sin(age * 0.7 + flap0))
                    bird.rotation_euler = q_.to_euler()
                    beat = 1 - 0.35 * max(0.0, math.sin(age * 3.1 + flap0)) * (age % 6 < 3)  # bursts of quick beats
                    bird.scale = (BIRD * D / 80 * beat, BIRD * D / 80, BIRD * D / 80)
                bird.hide_render = not (0 <= age <= DRAW + 6)
                for prop in ("location", "rotation_euler", "scale", "hide_render"):
                    bird.keyframe_insert(prop, frame=i + 1 + FO)
                cu.bevel_depth = (0.35 + 1.0 * min(1.0, grow)) * D / 80      # the contrail spreads as it ages
                dens.inputs[1].default_value = base * fade
                for prop in ("bevel_factor_end", "bevel_factor_start", "bevel_depth"):
                    cu.keyframe_insert(prop, frame=i + 1 + FO)
                dens.inputs[1].keyframe_insert("default_value", frame=i + 1 + FO)
    # --- the song's other layers (user, 2026-10-27: "identify if you can extract other unique sounds from the audio and
    # add those as visuals too"; music_layers.py separates them with Demucs). Each layer has one visual of its own:
    #   chime (4-note bell motif) -> each note lights one crystal from inside, a phrase walks across four of them
    #   drone (a pure 937 Hz tone that swells and fades) -> a 22-degree ice halo round the sun, as bright as the drone
    #   pad (chord changes) -> cloud shadows sweep across the ice at each change
    #   subbass (the sustained low swell) -> ground drift: low blowing snow that rises with it
    #   clicks (broadband ticks) -> hairline cracks glinting on the ice surface for a moment
    if a.layers:
        LY = json.load(open(a.layers)); FX = set(a.layer_fx.split(","))
        bpy.context.view_layer.update()

        def at(v, i):
            return v[min(i, len(v) - 1)]

        def cam_at(i):                                            # camera position, forward and right at frame i
            c0 = path(dist[i]); c0.z = zs[i]
            fw = path(dist[i] + 30) - c0; fw.z = 0; fw.normalize()
            return c0, fw, Vector((fw.y, -fw.x, 0))

        def glow_material(name, color, k):                        # emission = k x the object's colour (keyed per object)
            m = bpy.data.materials.new(name); m.use_nodes = True; nt_ = m.node_tree; nt_.nodes.clear()
            oi = nt_.nodes.new("ShaderNodeObjectInfo"); sp_ = nt_.nodes.new("ShaderNodeSeparateColor")
            mu = nt_.nodes.new("ShaderNodeMath"); mu.operation = "MULTIPLY"; mu.inputs[1].default_value = k
            em = nt_.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (*color, 1)
            o_ = nt_.nodes.new("ShaderNodeOutputMaterial")
            nt_.links.new(oi.outputs["Color"], sp_.inputs[0]); nt_.links.new(sp_.outputs[0], mu.inputs[0])
            nt_.links.new(mu.outputs[0], em.inputs["Strength"]); nt_.links.new(em.outputs[0], o_.inputs["Surface"])
            return m

        def key_flash(ob, h, st, tau, span):                     # object colour: 0, a flash of st at h, decaying
            for i in (h - 1, h + span + 1):
                if 0 <= i < N:
                    ob.color = (0, 0, 0, 1); ob.keyframe_insert("color", frame=i + 1 + FO)
            for i in range(h, min(N, h + span + 1)):
                g = st * math.exp(-(i - h) / tau)
                ob.color = (g, g, g, 1); ob.keyframe_insert("color", frame=i + 1 + FO)
            return [ob.color]

        if "chime" in FX:
            # the crystals' own ice glows from inside: emission = the object's colour (black unless a note lights it)
            nt_ = spire_m.node_tree; b_ = nt_.nodes["Principled BSDF"]
            oi = nt_.nodes.new("ShaderNodeObjectInfo"); sp_ = nt_.nodes.new("ShaderNodeSeparateColor")
            mu = nt_.nodes.new("ShaderNodeMath"); mu.operation = "MULTIPLY"; mu.inputs[1].default_value = a.chime_glow
            nt_.links.new(oi.outputs["Color"], sp_.inputs[0]); nt_.links.new(sp_.outputs[0], mu.inputs[0])
            nt_.links.new(mu.outputs[0], b_.inputs["Emission Strength"])
            b_.inputs["Emission Color"].default_value = (0.55, 0.82, 1.0, 1)
            lit = [s_ for s_ in spires if s_.active_material is spire_m]
            for s_ in lit:
                s_.color = (0, 0, 0, 1)
            centre = {s_.name: s_.matrix_world @ (sum((v.co for v in s_.data.vertices), Vector()) / len(s_.data.vertices))
                      for s_ in lit}
            used, notes = set(), LY["chime"]
            for n_, nt in enumerate(notes):
                h = nt["frame"]
                if h >= N:
                    continue
                first = n_ == 0 or h - notes[n_ - 1]["frame"] > 20              # a new phrase starts on the right
                k_ = 0 if first else k_ + 1
                c0, fw, rt = cam_at(h)
                want_side, want_d = (1 if k_ % 2 == 0 else -1), 14 + 5 * k_     # alternate sides, walking away
                best, bs = None, 1e9
                for s_ in lit:
                    if s_.name in used:
                        continue
                    r_ = centre[s_.name] - c0; df, dl = r_.dot(fw), r_.dot(rt)
                    if df < 8 or df > 45 or abs(math.degrees(math.atan2(dl, df))) > 26 or dl * want_side < 1.5:
                        continue
                    sc_ = abs(df - want_d) + 0.3 * abs(abs(math.degrees(math.atan2(dl, df))) - 14)
                    if sc_ < bs:
                        best, bs = s_, sc_
                if best is None:
                    continue
                used.add(best.name)
                key_flash(best, h, 0.35 + 0.65 * nt["strength"], 9, 50)
                pl = bpy.data.objects.new(f"chime{n_}", bpy.data.lights.new(f"chime{n_}", "POINT"))
                sc.collection.objects.link(pl); pl.location = centre[best.name]
                pl.data.color = (0.55, 0.82, 1.0); pl.data.shadow_soft_size = 0.4
                for i in (h - 1, h + 51):
                    if 0 <= i < N:
                        pl.data.energy = 0; pl.data.keyframe_insert("energy", frame=i + 1 + FO)
                for i in range(h, min(N, h + 51)):                     # it lights the snow and ice around it too
                    pl.data.energy = a.chime_glow * 60 * (0.35 + 0.65 * nt["strength"]) * math.exp(-(i - h) / 9)
                    pl.data.keyframe_insert("energy", frame=i + 1 + FO)
        if "drone" in FX:
            # a 22-degree halo: a ring on a camera-facing quad far off in the sun's direction, travelling with the camera
            # (at infinity, like the sun); faintly red inside, white-blue outside; seen only by the camera
            R = 1500; sd_ = (sun.rotation_euler.to_matrix() @ Vector((0, 0, 1))).normalized()
            rr = R * math.tan(math.radians(22))
            me = bpy.data.meshes.new("halo"); s2 = 1.45
            me.from_pydata([(-s2, -s2, 0), (s2, -s2, 0), (s2, s2, 0), (-s2, s2, 0)], [], [(0, 1, 2, 3)])
            halo = bpy.data.objects.new("halo", me); sc.collection.objects.link(halo)
            halo.scale = (rr, rr, rr)
            halo.rotation_euler = (-sd_).to_track_quat("-Z", "Y").to_euler()      # its +z faces the camera
            for vis in ("visible_diffuse", "visible_shadow", "visible_volume_scatter"):
                setattr(halo, vis, False)
            halo.cycles.use_motion_blur = False
            m = bpy.data.materials.new("halo"); m.use_nodes = True; nt_ = m.node_tree; nt_.nodes.clear()
            tc = nt_.nodes.new("ShaderNodeTexCoord"); ln = nt_.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"
            nt_.links.new(tc.outputs["Object"], ln.inputs[0])
            rise = nt_.nodes.new("ShaderNodeMapRange")                # sharp inner edge, soft outer fall-off
            rise.inputs["From Min"].default_value, rise.inputs["From Max"].default_value = 0.955, 1.0
            fall = nt_.nodes.new("ShaderNodeMapRange")
            fall.inputs["From Min"].default_value, fall.inputs["From Max"].default_value = 1.0, 1.22
            fall.inputs["To Min"].default_value, fall.inputs["To Max"].default_value = 1.0, 0.0
            fall.interpolation_type = "SMOOTHSTEP"
            for n2 in (rise, fall):
                nt_.links.new(ln.outputs["Value"], n2.inputs["Value"])
            prof = nt_.nodes.new("ShaderNodeMath"); prof.operation = "MULTIPLY"
            nt_.links.new(rise.outputs["Result"], prof.inputs[0]); nt_.links.new(fall.outputs["Result"], prof.inputs[1])
            dens_ = nt_.nodes.new("ShaderNodeMath"); dens_.operation = "MULTIPLY"
            nt_.links.new(prof.outputs[0], dens_.inputs[0])
            hue = nt_.nodes.new("ShaderNodeMapRange")
            hue.inputs["From Min"].default_value, hue.inputs["From Max"].default_value = 0.97, 1.08
            nt_.links.new(ln.outputs["Value"], hue.inputs["Value"])
            mx = nt_.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"
            mx.inputs["A"].default_value = (1.0, 0.72, 0.55, 1); mx.inputs["B"].default_value = (0.82, 0.9, 1.0, 1)
            nt_.links.new(hue.outputs["Result"], mx.inputs["Factor"])
            em = nt_.nodes.new("ShaderNodeEmission"); tr = nt_.nodes.new("ShaderNodeBsdfTransparent")
            ad = nt_.nodes.new("ShaderNodeAddShader"); o_ = nt_.nodes.new("ShaderNodeOutputMaterial")
            nt_.links.new(mx.outputs["Result"], em.inputs["Color"]); nt_.links.new(dens_.outputs[0], em.inputs["Strength"])
            nt_.links.new(tr.outputs[0], ad.inputs[0]); nt_.links.new(em.outputs[0], ad.inputs[1])
            nt_.links.new(ad.outputs[0], o_.inputs["Surface"])
            me.materials.append(m)
            dr = LY["drone"]["energy"]
            for i in range(N):
                c0, _, _ = cam_at(i)
                halo.location = c0 + sd_ * R; halo.keyframe_insert("location", frame=i + 1 + FO)
                dens_.inputs[1].default_value = a.halo * at(dr, i) ** 1.5
                dens_.inputs[1].keyframe_insert("default_value", frame=i + 1 + FO)
        if "pad" in FX:
            # a broken cloud layer 40 m up, seen only through its shadows; it drifts slowly, and at each chord change it
            # slides one cloud-width across in ~1.2 s, so the shadows sweep over the ice and the light changes
            bpy.ops.mesh.primitive_plane_add(size=1, location=(0, L / 2, 40))
            cl = bpy.context.object; cl.name = "clouds"; cl.scale = (500, L + 700, 1)
            for vis in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission",
                        "visible_volume_scatter"):
                setattr(cl, vis, False)
            m = bpy.data.materials.new("clouds"); m.use_nodes = True; nt_ = m.node_tree; nt_.nodes.clear()
            tc = nt_.nodes.new("ShaderNodeTexCoord"); mp = nt_.nodes.new("ShaderNodeMapping")
            nz = nt_.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.03; nz.inputs["Detail"].default_value = 4
            rg = nt_.nodes.new("ShaderNodeMapRange")
            rg.inputs["From Min"].default_value, rg.inputs["From Max"].default_value = 0.5, 0.56
            tr = nt_.nodes.new("ShaderNodeBsdfTransparent"); df_ = nt_.nodes.new("ShaderNodeBsdfDiffuse")
            mxs = nt_.nodes.new("ShaderNodeMixShader"); o_ = nt_.nodes.new("ShaderNodeOutputMaterial")
            tcw = bpy.data.objects.new("world_ref", None); sc.collection.objects.link(tcw); tc.object = tcw
            nt_.links.new(tc.outputs["Object"], mp.inputs["Vector"]); nt_.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
            nt_.links.new(nz.outputs["Fac"], rg.inputs["Value"]); nt_.links.new(rg.outputs["Result"], mxs.inputs["Fac"])
            nt_.links.new(tr.outputs[0], mxs.inputs[1]); nt_.links.new(df_.outputs[0], mxs.inputs[2])
            nt_.links.new(mxs.outputs[0], o_.inputs["Surface"])
            cl.data.materials.append(m)
            ch = LY["pad"]["changes"]; SW = 28
            for i in range(N):
                x_ = 0.6 * i / C["fps"]
                for c_ in ch:
                    u_ = min(1.0, max(0.0, (i - c_) / SW)); x_ += a.cloud_sweep * (3 * u_ * u_ - 2 * u_ ** 3)
                mp.inputs["Location"].default_value = (x_, 0, 0)
                mp.inputs["Location"].keyframe_insert("default_value", frame=i + 1 + FO)
        if "subbass" in FX:
            # low drift: a ground-hugging volume (~1.5 m deep) of streaky blown snow travelling with the camera; its
            # density rides the sub-bass, streaks blowing across the path in world space
            bpy.ops.mesh.primitive_cube_add(size=1)
            gd = bpy.context.object; gd.name = "drift"; gd.scale = (70, 90, 3.0)
            m = bpy.data.materials.new("drift"); m.use_nodes = True; nt_ = m.node_tree; nt_.nodes.clear()
            pv = nt_.nodes.new("ShaderNodeVolumePrincipled")
            pv.inputs["Color"].default_value = (0.9, 0.95, 1.0, 1); pv.inputs["Anisotropy"].default_value = 0.55
            tcw_ = bpy.data.objects.get("world_ref") or bpy.data.objects.new("world_ref", None)
            if tcw_.name not in sc.collection.objects:
                sc.collection.objects.link(tcw_)
            tc = nt_.nodes.new("ShaderNodeTexCoord"); tc.object = tcw_; mp = nt_.nodes.new("ShaderNodeMapping")
            mp.inputs["Scale"].default_value = (0.15, 0.6, 1.4)                # streaks drawn out along the wind (x)
            nz = nt_.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 1.0; nz.inputs["Detail"].default_value = 5
            rg = nt_.nodes.new("ShaderNodeMapRange")
            rg.inputs["From Min"].default_value, rg.inputs["From Max"].default_value = 0.45, 0.75
            tco = nt_.nodes.new("ShaderNodeTexCoord"); sx = nt_.nodes.new("ShaderNodeSeparateXYZ")
            hf = nt_.nodes.new("ShaderNodeMapRange")                     # densest at the ground, gone 2 m up
            hf.inputs["From Min"].default_value, hf.inputs["From Max"].default_value = 0.5, -0.17
            hf.interpolation_type = "SMOOTHSTEP"
            mu = nt_.nodes.new("ShaderNodeMath"); mu.operation = "MULTIPLY"
            dens_ = nt_.nodes.new("ShaderNodeMath"); dens_.operation = "MULTIPLY"
            o_ = nt_.nodes.new("ShaderNodeOutputMaterial")
            nt_.links.new(tc.outputs["Object"], mp.inputs["Vector"]); nt_.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
            nt_.links.new(nz.outputs["Fac"], rg.inputs["Value"])
            nt_.links.new(tco.outputs["Object"], sx.inputs[0]); nt_.links.new(sx.outputs["Z"], hf.inputs["Value"])
            nt_.links.new(rg.outputs["Result"], mu.inputs[0]); nt_.links.new(hf.outputs["Result"], mu.inputs[1])
            nt_.links.new(mu.outputs[0], dens_.inputs[0]); nt_.links.new(dens_.outputs[0], pv.inputs["Density"])
            nt_.links.new(pv.outputs["Volume"], o_.inputs["Volume"])
            gd.data.materials.append(m)
            sb = smooth(LY["subbass"], 0.25)
            for i in range(N):
                c0, fw, _ = cam_at(i)
                gd.location = c0 + fw * 40 + Vector((0, 0, -1.8 + 0.5))   # ground - 1 m .. ground + 2 m; gd.keyframe_insert("location", frame=i + 1 + FO)
                dens_.inputs[1].default_value = a.drift * at(sb, i) ** 1.5
                dens_.inputs[1].keyframe_insert("default_value", frame=i + 1 + FO)
                mp.inputs["Location"].default_value = (-0.05 * i, 0.01 * i, 0)     # blowing across at ~8 m/s
                mp.inputs["Location"].keyframe_insert("default_value", frame=i + 1 + FO)
            sc.cycles.volume_step_rate = 4; sc.cycles.volume_max_steps = 256
        if "clicks" in FX:
            # each tick: one or two hairline cracks on the ice 5-22 m ahead flash white-blue and fade within ~1/4 s
            crack_m = glow_material("crack", (0.7, 0.88, 1.0), a.crack_glow)
            CK = LY["clicks"]
            for k, (h, st) in enumerate(zip(CK["hits"], CK["strength"])):
                if h >= N:
                    continue
                c0, fw, rt = cam_at(h)
                for j in range(1 + (st > 0.5)):
                    p0 = c0 + fw * rnd.uniform(5, 22) + rt * rnd.uniform(-7, 7)
                    ang = rnd.uniform(0, 2 * math.pi); pts = [p0]
                    for _ in range(rnd.randint(6, 10)):                     # a zigzag fracture
                        ang += rnd.uniform(-0.8, 0.8); q = pts[-1] + Vector((math.cos(ang), math.sin(ang), 0)) * rnd.uniform(0.12, 0.3)
                        pts.append(q)
                    cu = bpy.data.curves.new(f"crack{k}_{j}", "CURVE"); cu.dimensions = "3D"
                    br = [pts]
                    if len(pts) > 4:                                        # a branch off the middle
                        b0 = pts[len(pts) // 2]; ang2 = ang + rnd.choice((-1, 1)) * rnd.uniform(0.7, 1.3)
                        br.append([b0 + Vector((math.cos(ang2), math.sin(ang2), 0)) * 0.18 * t_ for t_ in range(4)])
                    for line in br:
                        sp = cu.splines.new("POLY"); sp.points.add(len(line) - 1)
                        for jj, q in enumerate(line):
                            sp.points[jj].co = (q.x, q.y, height(q.x, q.y) + 0.03, 1)
                    cu.bevel_depth = 0.012
                    ob = bpy.data.objects.new(f"crack{k}_{j}", cu); sc.collection.objects.link(ob)
                    cu.materials.append(crack_m); ob.cycles.use_motion_blur = False
                    key_flash(ob, h, 0.3 + 0.7 * st, 2.5, 10)
    if a.sparkles:                     # tweets -> sparkles in the dusk sky (user, 2026-10-27: tight band-pass)
        # ~600 points of light on a far dome ahead, in 16 patches; each tweet lights one patch, which fades in ~0.2 s
        G = 16; T = C["tweets"]; mats = []
        for g in range(G):
            m = bpy.data.materials.new(f"sparkle{g}"); m.use_nodes = True; nt_ = m.node_tree; nt_.nodes.clear()
            em_ = nt_.nodes.new("ShaderNodeEmission"); em_.inputs["Color"].default_value = (0.85, 0.92, 1.0, 1)
            tr = nt_.nodes.new("ShaderNodeBsdfTransparent"); ad = nt_.nodes.new("ShaderNodeAddShader")  # unlit: invisible
            o_ = nt_.nodes.new("ShaderNodeOutputMaterial")
            nt_.links.new(tr.outputs[0], ad.inputs[0]); nt_.links.new(em_.outputs[0], ad.inputs[1])
            nt_.links.new(ad.outputs[0], o_.inputs["Surface"])
            mats.append((m, em_))
        cy = (L - 40) / 2
        for k in range(600):
            g = rnd.randrange(G)
            az = math.radians(rnd.gauss(0, 30)); el = math.radians(rnd.uniform(4, 30)); R = 800
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=rnd.uniform(2.4, 5.2),
                                                  location=(R * math.sin(az) * math.cos(el), cy + R * math.cos(az) * math.cos(el),
                                                            R * math.sin(el)))
            o = bpy.context.object; o.data.materials.append(mats[g][0])
            o.visible_shadow = False; o.visible_diffuse = False; o.visible_glossy = True; o.pass_index = 7
        group_of = [rnd.randrange(G) for _ in T["hits"]]
        for g, (m, em_) in enumerate(mats):
            for i in range(N):
                v = sum(st * math.exp(-(i - h) / 3) for h, st, gg in zip(T["hits"], T["hit_strength"], group_of)
                        if gg == g and h <= i)
                v = v if v > 0.02 else 0                              # dark between tweets
                em_.inputs["Strength"].default_value = 150 * v
                em_.inputs["Strength"].keyframe_insert("default_value", frame=i + 1 + FO)
        # star-shaped glints on the sparkles only: glare on the emission pass (the sparkles are see-through, so an
        # object-index mask never records them; the sky and sun are not in this pass, so the sun gets no cross)
        bpy.context.view_layer.use_pass_emit = True
        sc.use_nodes = True; ct = sc.node_tree; ct.nodes.clear()
        rl = ct.nodes.new("CompositorNodeRLayers")
        gl = ct.nodes.new("CompositorNodeGlare")
        gl.glare_type = "STREAKS"; gl.streaks = 4; gl.angle_offset = 0; gl.threshold = 1.0
        gl.mix = 1.0; gl.quality = "HIGH"; gl.fade = 0.82                # mix 1: the streaks alone
        add = ct.nodes.new("CompositorNodeMixRGB"); add.blend_type = "ADD"; add.inputs["Fac"].default_value = 1
        co = ct.nodes.new("CompositorNodeComposite")
        dim = ct.nodes.new("CompositorNodeMixRGB"); dim.blend_type = "MULTIPLY"; dim.inputs["Fac"].default_value = 1
        dim.inputs[2].default_value = (0.2, 0.2, 0.2, 1)            # short, faint spikes (like a star's), not crosses
        ct.links.new(rl.outputs["Emit"], dim.inputs[1]); ct.links.new(dim.outputs["Image"], gl.inputs["Image"])
        ct.links.new(rl.outputs["Image"], add.inputs[1]); ct.links.new(gl.outputs["Image"], add.inputs[2])
        ct.links.new(add.outputs["Image"], co.inputs["Image"])
    if a.mist:                         # mids -> a low mist that comes and goes, lit gold by the low sun
        mi = C["mist"]; mi = [mi[min(i, len(mi) - 1)] for i in range(N)]
        q = sorted(mi); lo_, hi_ = q[int(0.05 * (N - 1))], q[int(0.95 * (N - 1))]
        mi = [min(1, max(0, (v - lo_) / (hi_ - lo_ + 1e-9))) ** 1.5 for v in mi]
        bpy.ops.mesh.primitive_cube_add(size=1)
        mc = bpy.context.object; mc.name = "mist"
        mc.location = (0, (L - 30) / 2, 2.5); mc.scale = (120, L + 120, 5)
        m = bpy.data.materials.new("mist"); m.use_nodes = True; nt_ = m.node_tree; nt_.nodes.clear()
        pv = nt_.nodes.new("ShaderNodeVolumePrincipled")
        pv.inputs["Color"].default_value = (*(float(v) for v in a.mist_color.split(",")), 1); pv.inputs["Anisotropy"].default_value = 0.6
        tc = nt_.nodes.new("ShaderNodeTexCoord"); mp = nt_.nodes.new("ShaderNodeMapping")
        nz = nt_.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.08; nz.inputs["Detail"].default_value = 3
        ramp = nt_.nodes.new("ShaderNodeMapRange")                   # patchy banks, not a uniform fog
        ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = 0.42, 0.7
        hgt = nt_.nodes.new("ShaderNodeSeparateXYZ")                 # thinning with height
        hf = nt_.nodes.new("ShaderNodeMapRange")
        hf.inputs["From Min"].default_value, hf.inputs["From Max"].default_value = 4.5, 0.0
        mul = nt_.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
        dens = nt_.nodes.new("ShaderNodeMath"); dens.operation = "MULTIPLY"
        out_ = nt_.nodes.new("ShaderNodeOutputMaterial")
        nt_.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt_.links.new(tc.outputs["Object"], hgt.inputs[0])
        tcw = nt_.nodes.new("ShaderNodeTexCoord")
        nt_.links.new(tcw.outputs["Object"], mp.inputs["Vector"])
        nt_.links.new(mp.outputs["Vector"], nz.inputs["Vector"]); nt_.links.new(nz.outputs["Fac"], ramp.inputs["Value"])
        nt_.links.new(hgt.outputs["Z"], hf.inputs["Value"])
        nt_.links.new(ramp.outputs["Result"], mul.inputs[0]); nt_.links.new(hf.outputs["Result"], mul.inputs[1])
        nt_.links.new(mul.outputs["Value"], dens.inputs[0]); nt_.links.new(dens.outputs["Value"], pv.inputs["Density"])
        nt_.links.new(pv.outputs["Volume"], out_.inputs["Volume"])
        mc.data.materials.append(m)
        for i in range(N):
            dens.inputs[1].default_value = a.mist * mi[i]
            dens.inputs[1].keyframe_insert("default_value", frame=i + 1 + FO)
            mp.inputs["Location"].default_value = (0.004 * i, 0.0015 * i, 0)      # banks drifting across
            mp.inputs["Location"].keyframe_insert("default_value", frame=i + 1 + FO)
        sc.cycles.volume_step_rate = 4; sc.cycles.volume_max_steps = 256
    sc.cycles.max_bounces = 8; sc.cycles.transmission_bounces = 8
    sc.cycles.glossy_bounces = 8 if a.theme in ("chrome", "icechrome") else 4

if a.snow and os.environ.get("SNOW_CHECK"):     # flakes within 20 m in front of the camera, per second, no render
    for f in range(1, N + 1, int(C["fps"])):
        sc.frame_set(f + FO); dg = bpy.context.evaluated_depsgraph_get()
        cp = cam.matrix_world.translation; fw = cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        n_ = {}
        for ob in bpy.data.objects:
            if ob.particle_systems:
                pts = ob.evaluated_get(dg).particle_systems[0].particles
                n_[ob.name] = sum(1 for q in pts if q.alive_state == "ALIVE" and (q.location - cp).length < 20
                                  and (q.location - cp).dot(fw) > 0)
        print("snow check frame", f, n_, flush=True)
    sys.exit(0)
os.makedirs(out, exist_ok=True)
# bake the snow over the whole scene range, run-up included
if a.snow:
    bpy.ops.ptcache.bake_all(bake=True)
bpy.ops.wm.save_as_mainfile(filepath=f"{out}/scene_{a.mode}.blend")
sc.render.filepath = f"{out}/{a.mode}/"
if a.only:
    for f in (int(v) for v in a.only.split(",")):
        sc.frame_set(f + FO); sc.render.filepath = f"{out}/{a.mode}/{f:04d}.png"
        bpy.ops.render.render(write_still=True)
else:
    sc.frame_start = FO + 1
    bpy.ops.render.render(animation=True)
    for k in range(1, N + 1):                  # scene frame FO+k -> file k
        os.replace(f"{out}/{a.mode}/{FO + k:04d}.png", f"{out}/{a.mode}/{k:04d}.png")
print("ice ok", out, N, "frames")
