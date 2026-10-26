"""Paint a blockout world view by view until every surface has been seen head-on (2026-10-26). Round 4 projected two
styled views onto the room: every camera then saw the same room, but surfaces those two views never saw (or saw edge-on)
were smeared streaks, and Layout-To-Render reproduces that render frame by frame, flaws included. This is the
TEXTure / Text2Tex scheme for a Blender scene:

  next     pick the candidate camera that sees the most still-unpainted surface well, and render from it
           partial.png (painted surfaces with their paint, the rest as lit grey clay) and need.png (white where this
           view will paint); an image model then restyles partial.png, keeping the painted parts (fill_view.py),
           and writes styled.png
  render   texture every surface from the view that sees it best, save it (painted.blend), and render the shot cameras
           (tex_<cam>/%04d.png, her stand-in grey) and a ring of views around the room (texviews/NN.png)

A surface counts as painted by a view when it is in frame, not hidden behind anything (ray cast) and faces the camera
(cosine >= GOOD). Nothing here is specific to one room: candidates are the shot cameras (every 20th frame) and a ring
of cameras inside the room's bounding box.

    docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work blender -b OUT/room.blend \
        -P pipeline/world/paint_world.py -- OUT next|render [DEST]
(render with another .blend of the same room, e.g. one with her figure.py figure in it, and DEST for its renders)
State: OUT/paint/views.json ([{"name", "loc", "rot", "lens", "img"}]), OUT/paint/vNN/.
"""
import json, math, os, sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

GOOD = 0.35          # a view paints a surface it sees at up to ~70 degrees off head-on
FIG = ("torso", "head", "thigh_l", "thigh_r", "shin_l", "shin_r", "cloth")   # her stand-in (room_blockout.py)
args = sys.argv[sys.argv.index("--") + 1:]
out, mode = os.path.abspath(args[0]), args[1]
dest = os.path.abspath(args[2]) if len(args) > 2 else out      # render: where tex_* and texviews go
P = os.path.join(out, "paint")
os.makedirs(P, exist_ok=True)
views = json.load(open(f"{P}/views.json")) if os.path.exists(f"{P}/views.json") else []
sc = bpy.context.scene
W, H = sc.render.resolution_x, sc.render.resolution_y
figure = [o for o in sc.objects if o.type == "MESH" and (o.name.split(".")[0] in FIG     # the stand-in, or her
          or (o.parent is not None and o.parent.type == "ARMATURE"))]                       # rigged body (figure.py)
room = [o for o in sc.objects if o.type == "MESH" and o not in figure]
shot_cams = sorted(o.name for o in sc.objects if o.type == "CAMERA")


def camera(name, loc, rot, lens):
    c = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    sc.collection.objects.link(c)
    c.location, c.rotation_euler, c.data.lens = Vector(loc), rot, lens
    bpy.context.view_layer.update()          # matrix_world is stale until the scene updates
    return c


def candidates():               # (name, loc, rot, lens): the shot cameras every 20 frames, then a ring inside the room
    out_ = []
    for n in shot_cams:
        c = bpy.data.objects[n]
        for f in range(sc.frame_start, sc.frame_end + 1, 20):
            sc.frame_set(f)
            out_.append((f"{n}_{f:04d}", tuple(c.matrix_world.translation), tuple(c.matrix_world.to_euler()), c.data.lens))
    inside = [o for o in room if o.name == "floor"] or room     # the ring stays inside the room (not the street)
    pts = [ob.matrix_world @ Vector(v) for ob in inside for v in ob.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
    mid = (lo + hi) / 2
    rx, ry = 0.38 * (hi.x - lo.x), 0.38 * (hi.y - lo.y)
    for i in range(16):
        a = 2 * math.pi * i / 16
        for z, look in ((1.0, 0.7), (1.6, 0.7), (2.3, 0.0)):    # eye heights, and a high one looking down at the floor
            pos = Vector((mid.x + rx * math.cos(a), mid.y + ry * math.sin(a), lo.z + z))
            at = Vector((mid.x - 0.5 * rx * math.cos(a), mid.y - 0.5 * ry * math.sin(a), lo.z + look))
            out_.append((f"ring{i:02d}_{z:.1f}", tuple(pos), tuple((at - pos).to_track_quat("-Z", "Y").to_euler()), 30))
    return out_


for ob in figure if mode == "next" else []:
    bpy.data.objects.remove(ob, do_unlink=True)
figure = [] if mode == "next" else figure
for ob in room:                 # small faces, so each piece is judged (and painted) on its own
    bpy.context.view_layer.objects.active = ob
    sub = ob.modifiers.new("sub", "SUBSURF"); sub.subdivision_type = "SIMPLE"; sub.levels = 3
    bpy.ops.object.modifier_apply(modifier="sub")
for ob in figure:               # she never hides the room from a view (ray casts see visible objects only)
    ob.hide_viewport = True
dg = bpy.context.evaluated_depsgraph_get()
faces = [(ob, p.index, ob.matrix_world @ p.center, (ob.matrix_world.to_3x3() @ p.normal).normalized(),
          p.area * ob.matrix_world.to_3x3().determinant() ** (2 / 3)) for ob in room for p in ob.data.polygons]


def quality(cam):               # per face: how head-on this camera sees it (0 if out of frame or hidden)
    sc.camera = cam
    o = cam.matrix_world.translation
    q = []
    for ob, i, c, n, _ in faces:
        d = c - o
        dist = d.length
        f = -n.dot(d / dist)
        v = world_to_camera_view(sc, cam, c)
        if f <= 0 or not (0 <= v.x <= 1 and 0 <= v.y <= 1 and v.z > 0):
            q.append(0.0); continue
        hit, loc, _, _, hob, _ = sc.ray_cast(dg, o, d / dist, distance=dist + 0.01)
        q.append(f if hit and (loc - o).length > dist - 0.02 else 0.0)
    return q


vcams = [camera(f"view_{v['name']}", v["loc"], v["rot"], v["lens"]) for v in views]
vq = [quality(c) for c in vcams]
best = [max((q[j] for q in vq), default=0.0) for j in range(len(faces))]
owner = [max(range(len(vq)), key=lambda k: vq[k][j]) if vq and best[j] > 0 else -1 for j in range(len(faces))]
fl = [o for o in room if o.name == "floor"]       # only the room counts (the street seen through the window is painted
if fl:                                             # by whichever view sees it, but never chases a camera outside)
    bb = [fl[0].matrix_world @ Vector(v) for v in fl[0].bound_box]
    x0, x1 = min(p.x for p in bb) - 0.3, max(p.x for p in bb) + 0.3
    y0, y1 = min(p.y for p in bb) - 0.3, max(p.y for p in bb) + 0.3
    faces = [f[:4] + ((f[4] if x0 <= f[2].x <= x1 and y0 <= f[2].y <= y1 else 0.0),) for f in faces]
area = sum(f[4] for f in faces)
painted = sum(f[4] for f, b in zip(faces, best) if b >= GOOD)
print(f"painted {painted / area:.1%} of the room with {len(views)} views", flush=True)


def material(name, k=None, rgb=None):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); o = nt.nodes.new("ShaderNodeOutputMaterial")
    if k is not None:           # view k's image, read through its own UV map, as emission (the paint is the light)
        tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = bpy.data.images.load(views[k]["img"]); tex.extension = "EXTEND"
        uv = nt.nodes.new("ShaderNodeUVMap"); uv.uv_map = "proj"; nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
        em = nt.nodes.new("ShaderNodeEmission"); nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], o.inputs["Surface"])
    elif rgb is not None:       # flat emission (masks)
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (*rgb, 1)
        nt.links.new(em.outputs["Emission"], o.inputs["Surface"])
    else:                       # lit grey clay: the shape, for the image model to restyle
        bs = nt.nodes.new("ShaderNodeBsdfDiffuse"); bs.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
        nt.links.new(bs.outputs["BSDF"], o.inputs["Surface"])
    return m


def texture(slot):              # slot(j) -> material index per face; materials: views..., clay, white, black
    """One UV map for every view: each face's UVs are its owning view's projection of its corners (a UV map per view, as
    first written, broke past 8 views: Blender meshes hold at most 8 UV maps, so views 9+ painted flat grey)."""
    mats = [material(f"view{k}", k=k) for k in range(len(views))] + [material("clay"), material("white", rgb=(1, 1, 1)),
                                                                     material("black", rgb=(0, 0, 0))]
    j = 0
    for ob in room:
        ob.data.materials.clear()
        for m in mats:
            ob.data.materials.append(m)
        uvl = ob.data.uv_layers.get("proj") or ob.data.uv_layers.new(name="proj")
        mw, vs, loops = ob.matrix_world, ob.data.vertices, ob.data.loops
        for p in ob.data.polygons:
            k = slot(j); j += 1
            p.material_index = k
            if k < len(views):
                for li in p.loop_indices:
                    v = world_to_camera_view(sc, vcams[k], mw @ vs[loops[li].vertex_index].co)
                    uvl.data[li].uv = (v.x, v.y)
    return len(views)


def render(path, samples=4):
    sc.cycles.samples = samples; sc.render.filepath = path; bpy.ops.render.render(write_still=True)


sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.use_denoising = False
sc.view_settings.view_transform = "Standard"
sc.world.use_nodes = True
bg = sc.world.node_tree.nodes["Background"]; bg.inputs[0].default_value = (1, 1, 1, 1)   # clay's light (soft, even)

if mode == "next":
    cands = [c for c in candidates() if c[0] not in {v["name"] for v in views}]
    gain = []
    for name, loc, rot, lens in cands:
        cam = camera(f"cand_{name}", loc, rot, lens)
        q = quality(cam)
        gain.append((sum(f[4] for f, b, x in zip(faces, best, q) if b < GOOD <= x), name, loc, rot, lens, q))
    g, name, loc, rot, lens, q = max(gain, key=lambda t: t[0])
    seen = [max([b] + [t[5][j] for t in gain]) >= GOOD for j, b in enumerate(best)]   # paintable from some camera
    reach = sum(f[4] for f, ok in zip(faces, seen) if ok)
    print(f"paintable: {reach / area:.0%} of the room's surface (the rest faces away from every camera); "
          f"painted {painted / reach:.0%} of that", flush=True)
    if g < 0.005 * reach:
        print("done: no candidate paints more than 0.5% of the paintable surface"); sys.exit(0)
    k = len(views)
    D = f"{P}/v{k:02d}"
    os.makedirs(D, exist_ok=True)
    cam = bpy.data.objects[f"cand_{name}"]; sc.camera = cam
    CLAY, WHITE, BLACK = k, k + 1, k + 2
    texture(lambda j: owner[j] if best[j] >= GOOD else CLAY)
    sc.cycles.use_denoising = True
    render(f"{D}/partial.png", samples=32)
    sc.cycles.use_denoising = False
    j = 0                       # the mask: what this view will paint (unpainted faces it sees well)
    for ob in room:
        for p in ob.data.polygons:
            p.material_index = WHITE if best[j] < GOOD <= q[j] else BLACK; j += 1
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    render(f"{D}/need.png", samples=1)
    json.dump({"name": name, "loc": list(loc), "rot": list(rot), "lens": lens, "img": f"{D}/styled.png",
               "gain": g / area}, open(f"{D}/view.json", "w"), indent=1)
    print(f"next: {name}, paints {g / reach:.1%} more of the paintable surface -> {D}")
else:
    bg.inputs[0].default_value = (0, 0, 0, 1)
    for ob in figure:
        ob.hide_viewport = False
    texture(lambda j: owner[j] if owner[j] >= 0 else len(views) + 2)        # never seen at all: black
    os.makedirs(dest, exist_ok=True)    # the painted world as a .blend to open and orbit (image paths relative to it)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(dest, "painted.blend"), relative_remap=True, copy=True)
    grey = bpy.data.materials.new("figure"); grey.diffuse_color = (0.55, 0.55, 0.55, 1)
    for ob in figure:
        ob.data.materials.clear(); ob.data.materials.append(grey)
    sc.cycles.samples = 4
    for n in shot_cams:
        sc.camera = bpy.data.objects[n]
        sc.render.filepath = os.path.join(dest, f"tex_{n.replace('cam_', '')}", "")
        bpy.ops.render.render(animation=True)
    for ob in figure:
        ob.hide_render = True
    for i, (name, loc, rot, lens) in enumerate(c for c in candidates() if c[0].startswith("ring")):
        sc.camera = camera(f"tv_{name}", loc, rot, lens)
        render(os.path.join(dest, "texviews", f"{i:02d}.png"))
print("ok", out)
