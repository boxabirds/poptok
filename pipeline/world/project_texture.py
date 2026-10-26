"""World-building test, round 4 (2026-10-26): give the blockout ONE appearance by projecting a styled view onto its
geometry, then render that textured world from any camera. Round 3 styled each look frame on its own and got three
different felt rooms in one shot (LTX cut between them); a LoRA trained on independently styled views would learn "a
felt room", not this room. Projected, every view shows the same room by construction; surfaces the styled view never
saw get stretched pixels, to be cleaned up afterwards.

    docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work blender -b OUT/room.blend \
        -P pipeline/world/project_texture.py -- OUT STYLED.png
Projects STYLED.png (a styled render of OUT/clean_A/0001.png, the room without her) from camera A at frame 1, then
writes OUT/tex_A/%04d.png and OUT/tex_B/%04d.png (both passes, textured, her stand-in kept in grey so the layout still
shows where she is), OUT/texclean_{A,B}/{0001,0041,0081}.png (the same frames without her) and OUT/texviews/NN.png
(24 views from around the room, for a training set).
"""
import math, os, sys

import bpy
from mathutils import Vector

# arguments: OUT, then one or more CAMERA:FRAME:STYLED.png projectors (a bare STYLED.png means cam_A:1). With several,
# each face takes the projector that faces it best (Blender's UV Project), so each surface is painted by one view:
# round 4's second view (cam_B:81) paints the reverse wall that cam_A never saw.
args = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(args[0])
specs = [a.split(":") if ":" in a else ["cam_A", "1", a] for a in args[1:]]
sc = bpy.context.scene
projs = []
for camname, frame, path in specs:
    sc.frame_set(int(frame))
    c = bpy.data.objects[camname]
    pr = bpy.data.objects.new(f"projector_{camname}_{frame}", c.data.copy())   # a fixed copy of the camera then
    sc.collection.objects.link(pr)
    pr.matrix_world = c.matrix_world.copy()
    projs.append((pr, os.path.abspath(path)))

FIG = ("torso", "head", "thigh_l", "thigh_r", "shin_l", "shin_r", "cloth")
def projected(path, k):          # projector k's image, read through its own UV map (proj<k>)
    m = bpy.data.materials.new(f"projected{k}"); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = bpy.data.images.load(path); tex.extension = "EXTEND"
    uv = nt.nodes.new("ShaderNodeUVMap"); uv.uv_map = f"proj{k}"; nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    em = nt.nodes.new("ShaderNodeEmission"); o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"]); nt.links.new(em.outputs["Emission"], o.inputs["Surface"])
    return m


mats = [projected(p, k) for k, (_, p) in enumerate(projs)]
from bpy_extras.object_utils import world_to_camera_view


def in_frame(pr, point):          # is this point inside projector pr's picture (and in front of it)?
    sc.render.resolution_x, sc.render.resolution_y = W, H
    v = world_to_camera_view(sc, pr, point)
    return 0.0 <= v.x <= 1.0 and 0.0 <= v.y <= 1.0 and v.z > 0
grey = bpy.data.materials.new("figure"); grey.diffuse_color = (0.55, 0.55, 0.55, 1)
room = [ob for ob in sc.objects if ob.type == "MESH" and not ob.name.split(".")[0] in FIG]
figure = [ob for ob in sc.objects if ob.type == "MESH" and ob.name.split(".")[0] in FIG]
W, H = sc.render.resolution_x, sc.render.resolution_y
for ob in room:
    bpy.context.view_layer.objects.active = ob
    # cut big faces (a floor, a wall) into small ones first, so each piece takes the view that sees it
    sub = ob.modifiers.new("sub", "SUBSURF"); sub.subdivision_type = "SIMPLE"; sub.levels = 4
    bpy.ops.object.modifier_apply(modifier="sub")
    for k in range(len(projs)):
        ob.data.uv_layers.new(name=f"proj{k}")
    # each face gets the material of the projector that faces it best (most opposite its normal), as UV Project picks
    ob.data.materials.clear()
    for m_ in mats:
        ob.data.materials.append(m_)
    mw = ob.matrix_world
    for poly in ob.data.polygons:
        n = (mw.to_3x3() @ poly.normal).normalized()
        c = mw @ poly.center
        facing = {k: -n.dot((c - pr.matrix_world.translation).normalized()) for k, (pr, _) in enumerate(projs)}
        seen = [k for k, (pr, _) in enumerate(projs) if facing[k] > 0.05 and in_frame(pr, c)]
        poly.material_index = max(seen or facing, key=lambda k: facing[k])   # the best-facing view that has it in frame
    for k, (pr, _) in enumerate(projs):
        m = ob.modifiers.new(f"project{k}", "UV_PROJECT")
        m.uv_layer = f"proj{k}"
        m.projector_count = 1; m.projectors[0].object = pr
        m.aspect_x, m.aspect_y = W / H, 1.0
for ob in figure:
    ob.data.materials.clear(); ob.data.materials.append(grey)

sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 4; sc.cycles.use_denoising = False   # emission only: a few samples do
sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
sc.view_settings.view_transform = "Standard"        # the texture's own colours, not Filmic/AgX


def render_to(path, frame):
    sc.frame_set(frame); sc.render.filepath = path; bpy.ops.render.render(write_still=True)


for name in ("A", "B"):
    sc.camera = bpy.data.objects[f"cam_{name}"]
    sc.render.filepath = os.path.join(out, f"tex_{name}", "")
    bpy.ops.render.render(animation=True)
    for ob in figure:
        ob.hide_render = True
    for f in (1, 41, 81):
        render_to(os.path.join(out, f"texclean_{name}", f"{f:04d}.png"), f)
    for ob in figure:
        ob.hide_render = False
# 24 views around the room for a training set: eye heights 1.1-1.7 m, looking across the room from points on an ellipse
cam = bpy.data.objects.new("cam_views", bpy.data.cameras.new("cam_views")); sc.collection.objects.link(cam)
cam.data.lens = 24; sc.camera = cam
for ob in figure:
    ob.hide_render = True
for i in range(24):
    a = 2 * math.pi * i / 24
    pos = Vector((2.2 * math.cos(a), 2.5 + 1.8 * math.sin(a), 1.1 + 0.6 * (i % 3) / 2))
    at = Vector((-0.6 * math.cos(a), 2.5 - 0.8 * math.sin(a), 0.9))
    cam.location = pos
    cam.rotation_euler = (at - pos).to_track_quat("-Z", "Y").to_euler()
    render_to(os.path.join(out, "texviews", f"{i:02d}.png"), 1)
print("done", out)
