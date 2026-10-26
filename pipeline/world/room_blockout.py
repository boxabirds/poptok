"""World-building test (2026-10-26): one Noble Hunt place, a ruined apartment room, as a low-poly Blender blockout, and
two camera passes through it, rendered as grey clay videos for LTX-2.5's Layout-To-Render IC-LoRA. Pass A enters by
the door and pushes toward the window where she sits; pass B comes back later from the window corner and looks back
at the room and the door. Both are cameras in the same world, so the room is the same room.

    docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work blender -b -P pipeline/world/room_blockout.py -- OUT
Writes OUT/clay_A/%04d.png, OUT/clay_B/%04d.png (1920x1088, 81 frames, 24 fps) and OUT/room.blend; and for the
look frames, OUT/clean_{A,B}/{0001,0041,0081}.png (the same frames without her stand-in, which an image model styles as
an object: a felt dummy beside her, round 2) with OUT/figure.json (where the stand-in is in each of those frames).
Round 3 (2026-10-26): B's pan is gentler (~45 degrees, not ~90: with one look frame LTX did not follow it).
"""
import math, os, sys

import bpy

out = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
W, H, N = 1920, 1088, 81

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def box(name, loc, size, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=[math.radians(r) for r in rot])
    o = bpy.context.object
    o.name, o.scale = name, (size[0] / 2, size[1] / 2, size[2] / 2)
    return o


def cyl(name, loc, r, d, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=r, depth=d, location=loc,
                                        rotation=[math.radians(x) for x in rot])
    bpy.context.object.name = name
    return bpy.context.object


def ball(name, loc, r):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=r, location=loc)
    bpy.context.object.name = name
    return bpy.context.object


# the room: 6 m wide (x), 5 m deep (y), 3 m high; window in the far wall (y=5), door in the near wall (y=0)
box("floor", (0, 2.5, -0.05), (6.4, 5.4, 0.1))
box("ceiling_left", (-1.6, 2.5, 3.05), (3.2, 5.4, 0.1))
box("ceiling_slab", (2.0, 3.2, 2.6), (2.0, 1.8, 0.12), rot=(10, -14, 0))      # a collapsed slab hanging down
box("wall_left", (-3.05, 2.5, 1.5), (0.1, 5.4, 3))
box("wall_right", (3.05, 2.5, 1.5), (0.1, 5.4, 3))
for x0, x1 in ((-3.1, -0.5), (1.1, 3.1)):                                      # far wall around the window
    box("far_wall", ((x0 + x1) / 2, 5.05, 1.5), (x1 - x0, 0.1, 3))
box("far_sill", (0.3, 5.05, 0.45), (1.6, 0.1, 0.9))
box("far_lintel", (0.3, 5.05, 2.6), (1.6, 0.1, 0.8))
box("windowsill", (0.3, 4.9, 0.92), (1.7, 0.35, 0.06))
box("window_mullion", (0.3, 5.05, 1.6), (0.06, 0.06, 1.3))
for x0, x1 in ((-3.1, -1.9), (-1.0, 3.1)):                                     # near wall around the door
    box("near_wall", ((x0 + x1) / 2, -0.05, 1.5), (x1 - x0, 0.1, 3))
box("door_lintel", (-1.45, -0.05, 2.6), (0.9, 0.1, 0.8))
box("door_leaf", (-2.1, 0.35, 1.0), (0.06, 0.85, 2.0), rot=(0, 0, 25))         # the door hanging open
# furniture and rubble
box("table_top", (-1.2, 3.0, 0.55), (1.4, 0.8, 0.06), rot=(0, 14, 8))           # a tilted table
for dx, dy in ((-0.6, -0.3), (0.6, -0.3), (-0.6, 0.3)):
    cyl("table_leg", (-1.2 + dx, 3.0 + dy, 0.3), 0.03, 0.6)
box("chair_seat", (1.8, 1.4, 0.25), (0.45, 0.45, 0.05), rot=(70, 0, 30))        # a chair on its side
box("chair_back", (1.95, 1.2, 0.35), (0.45, 0.05, 0.5), rot=(70, 0, 30))
for i, (x, y, s) in enumerate(((2.4, 3.8, 0.5), (2.0, 4.3, 0.35), (2.6, 4.5, 0.3), (-2.4, 1.0, 0.45),
                               (-2.6, 1.5, 0.3), (1.2, 2.6, 0.25), (-0.4, 1.6, 0.2))):
    box(f"rubble{i}", (x, y, s / 2), (s * 1.3, s, s), rot=(0, 0, 37 * i))
box("crate", (-2.3, 4.3, 0.35), (0.7, 0.7, 0.7), rot=(0, 0, 12))
# the stand-in for her: seated on the floor under the window, back to the wall, knees up, a cloth in her hands
cyl("torso", (0.9, 4.55, 0.5), 0.17, 0.6, rot=(-12, 0, 0))
ball("head", (0.9, 4.6, 0.95), 0.12)
cyl("thigh_l", (0.8, 4.3, 0.3), 0.07, 0.45, rot=(60, 0, 0))
cyl("thigh_r", (1.0, 4.3, 0.3), 0.07, 0.45, rot=(60, 0, 0))
cyl("shin_l", (0.8, 4.05, 0.2), 0.06, 0.42, rot=(-30, 0, 0))
cyl("shin_r", (1.0, 4.05, 0.2), 0.06, 0.42, rot=(-30, 0, 0))
box("cloth", (0.9, 4.25, 0.55), (0.35, 0.03, 0.3), rot=(-20, 0, 0))
# the street outside the window: ground and a few building blocks, so the window has something behind it
box("street", (0, 9, -0.6), (20, 8, 0.1))
for i, (x, y, h) in enumerate(((-4, 11, 6), (0, 12, 9), (4, 11, 5), (7, 10, 7))):
    box(f"building{i}", (x, y, h / 2 - 0.6), (3, 2.5, h))

# grey clay look: Workbench, studio light, cavity and shadows so the forms read
sc.render.engine = "BLENDER_WORKBENCH"
sh = sc.display.shading
sh.light, sh.color_type, sh.single_color = "STUDIO", "SINGLE", (0.6, 0.6, 0.6)
sh.show_cavity, sh.show_shadows = True, True
sc.display.shadow_focus = 0.6
sc.render.resolution_x, sc.render.resolution_y, sc.render.fps = W, H, 24
sc.frame_start, sc.frame_end = 1, N
world = bpy.data.worlds.new("w"); sc.world = world


def camera(name, keys, lens=24):
    """keys: [(frame, location, look-at point)]"""
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    sc.collection.objects.link(cam)
    cam.data.lens = lens
    tgt = bpy.data.objects.new(name + "_target", None)
    sc.collection.objects.link(tgt)
    c = cam.constraints.new("TRACK_TO"); c.target, c.track_axis, c.up_axis = tgt, "TRACK_NEGATIVE_Z", "UP_Y"
    for f, loc, at in keys:
        cam.location, tgt.location = loc, at
        cam.keyframe_insert("location", frame=f); tgt.keyframe_insert("location", frame=f)
    return cam


passes = {
    # A: through the door, a slow push toward the window and her
    "A": camera("cam_A", [(1, (-1.45, -0.6, 1.55), (0.6, 4.6, 0.9)), (N, (-0.6, 1.8, 1.35), (0.8, 4.6, 0.8))]),
    # B: later, from the window corner beside her, a slow pan from her across the room to the open door
    "B": camera("cam_B", [(1, (2.7, 3.6, 1.4), (0.9, 4.4, 0.7)), (N, (2.6, 3.4, 1.45), (-0.6, 2.2, 0.9))]),
}
import json
from bpy_extras.object_utils import world_to_camera_view
figure = [o for o in sc.objects if o.name.split(".")[0] in ("torso", "head", "thigh_l", "thigh_r", "shin_l", "shin_r", "cloth")]
KEYS = (1, 41, 81)
where = {}
for name, cam in passes.items():
    sc.camera = cam
    sc.render.filepath = os.path.join(out, f"clay_{name}", "")
    bpy.ops.render.render(animation=True)
    for f in KEYS:                       # where her stand-in is: frame fraction of its head and torso, and if visible
        sc.frame_set(f)
        pts = [world_to_camera_view(sc, cam, o.matrix_world.translation) for o in figure if o.name.startswith(("head", "torso"))]
        vis = [p for p in pts if 0 <= p.x <= 1 and 0 <= p.y <= 1 and p.z > 0]
        where[f"{name}_{f:04d}"] = {"visible": len(vis) == len(pts),
                                    "x": round(sum(p.x for p in pts) / len(pts), 2), "y": round(1 - sum(p.y for p in pts) / len(pts), 2)}
    for o in figure:
        o.hide_render = True
    for f in KEYS:
        sc.frame_set(f)
        sc.render.filepath = os.path.join(out, f"clean_{name}", f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    for o in figure:
        o.hide_render = False
json.dump(where, open(os.path.join(out, "figure.json"), "w"), indent=1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "room.blend"))
print("done", out)
