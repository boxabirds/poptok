"""Her animated figure for the world layout (2026-10-26). Round 4b showed Layout-To-Render reproduces the 3D render frame
by frame and keeps a grey stand-in a stand-in, so her body, pose and movement have to be in the layout too. This builds
a MakeHuman body (MPFB 2, in the blender image) with the CMU-mocap-compatible rig and drives it with a CMU motion
capture clip (Motionbuilder-friendly BVH release; the bone names match the rig's exactly).

    docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work blender -b [SCENE.blend] \
        -P pipeline/world/figure.py -- OUT CLIP.bvh [--start S] [--frames N] [--at X,Y,Z] [--face DEG] [--test]
Adds the figure to the scene (a new one if none is given): the clip from its frame S (in 24 fps frames), N frames,
moved so its first hip position is at X,Y (feet on the floor at Z), turned to face DEG (0 = +Y); saves OUT/figure.blend.
--test also renders OUT/figure_test/%04d.png, a clay side view, to check the motion.
"""
import argparse, json, math, os, sys

import bpy
from mathutils import Vector

from bl_ext.user_default.mpfb.services.humanservice import HumanService
from bl_ext.user_default.mpfb.services.locationservice import LocationService
from bl_ext.user_default.mpfb.services.rigservice import RigService
from bl_ext.user_default.mpfb.services.targetservice import TargetService

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("clip")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--frames", type=int, default=81)
ap.add_argument("--at", default="0,0,0")
ap.add_argument("--face", type=float, default=0.0)
ap.add_argument("--test", action="store_true")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
out = os.path.abspath(a.out)
os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
sc.render.fps = 24
if not bpy.data.filepath:       # no scene given: start empty (not Blender's default cube, camera and light)
    for o in list(sc.objects):
        bpy.data.objects.remove(o, do_unlink=True)

# the body: a young woman (MakeHuman macro sliders: gender 0 = female, age 0.5 = 25 years)
macro = TargetService.get_default_macro_info_dict()
macro.update(gender=0.0, age=0.5, muscle=0.5, weight=0.45, height=0.55, proportions=0.6)
macro["race"] = {"asian": 0.0, "caucasian": 1.0, "african": 0.0}
body = HumanService.create_human(macro_detail_dict=macro)
rig = HumanService.add_builtin_rig(body, "cmu_mb")
# the CMU clips are relative to a T pose; MakeHuman's rest pose has the arms down, so local rotations copied across
# raised her arms in front of her (first test). Make the rig's rest pose MPFB's own T pose for this skeleton.
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
RigService.set_pose_from_dict(rig, json.load(open(os.path.join(LocationService.get_mpfb_data("poses"), "cmu_mb_fk", "t-pose.json"))))
bpy.ops.object.mode_set(mode="OBJECT")
RigService.apply_pose_as_rest_pose(rig)
for m in [m for m in body.modifiers if m.type == "MASK"]:
    m.show_render = True

# the clip: imported at 24 fps (the BVH is 120 fps), scaled so its hip height matches the body's
bpy.ops.import_anim.bvh(filepath=os.path.abspath(a.clip), axis_forward="-Z", axis_up="Y", rotate_mode="XYZ",   # the CMU release is Y-up
                        use_fps_scale=True, update_scene_fps=False, update_scene_duration=False)
src = bpy.context.object
sc.frame_set(1)
src_hip = (src.matrix_world @ src.pose.bones["Hips"].head).z      # frame 1 is the CMU release's T pose, standing
dst_hip = rig.data.bones["Hips"].head_local.z
src.scale = (dst_hip / src_hip,) * 3

# retarget in world space: each bone turns by its mocap bone's turn away from the T pose (both rigs now rest in a T
# pose). Copying local rotations instead (MPFB's own BVH pose import) assumes identical bone axes; it raised her arms.
def wrot(ob, m):
    return ob.matrix_world.to_quaternion() @ m.to_quaternion()


bpy.context.view_layer.update()
names = sorted((b.name for b in rig.pose.bones if b.name in src.pose.bones), key=lambda nm: len(rig.pose.bones[nm].parent_recursive))
sc.frame_set(1)                 # the release's T pose is frame 1 (its rest skeleton is not a T pose)
src_rest = {nm: wrot(src, src.pose.bones[nm].matrix) for nm in names}
dst_rest = {nm: wrot(rig, rig.data.bones[nm].matrix_local) for nm in names}
s0, n = a.start + 1, a.frames
clip = []
for i in range(n):
    sc.frame_set(s0 + i)
    clip.append(({nm: wrot(src, src.pose.bones[nm].matrix) for nm in names},
                 src.matrix_world @ src.pose.bones["Hips"].head))
bpy.data.objects.remove(src, do_unlink=True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
to_arm = rig.matrix_world.inverted()
for pb in rig.pose.bones:
    pb.rotation_mode = "QUATERNION"
for i, (R, hip) in enumerate(clip):
    for nm in names:                    # parents first, so each child's head sits where its parent put it
        pb = rig.pose.bones[nm]
        m = (to_arm.to_quaternion() @ R[nm] @ src_rest[nm].inverted() @ dst_rest[nm]).to_matrix().to_4x4()
        m.translation = to_arm @ hip if nm == "Hips" else pb.matrix.translation
        pb.matrix = m
        bpy.context.view_layer.update()
    for nm in names:
        rig.pose.bones[nm].keyframe_insert("rotation_quaternion", frame=i + 1)
    rig.pose.bones["Hips"].keyframe_insert("location", frame=i + 1)
bpy.ops.object.mode_set(mode="OBJECT")

sc.frame_start, sc.frame_end = 1, n
sc.frame_set(1)
h0 = rig.matrix_world @ rig.pose.bones["Hips"].head
x, y, z = (float(v) for v in a.at.split(","))
rig.rotation_euler.z = math.radians(a.face)
bpy.context.view_layer.update()
h0 = rig.matrix_world @ rig.pose.bones["Hips"].head
rig.location += Vector((x - h0.x, y - h0.y, 0))
bpy.context.view_layer.update()
lowest = min((body.matrix_world @ v.co).z for v in body.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices)
rig.location.z += z - lowest
rig.name = "her_rig"; body.name = "her"
bpy.ops.wm.save_as_mainfile(filepath=f"{out}/figure.blend")

if a.test:                      # a clay side view of the motion
    cam = bpy.data.objects.new("cam_test", bpy.data.cameras.new("cam_test")); sc.collection.objects.link(cam)
    cam.location = (x + 4.0, y, z + 1.0)          # follows her hips from the side, 4 m off
    c = cam.constraints.new("COPY_LOCATION"); c.target, c.subtarget = rig, "Hips"; c.use_z = False
    c.use_offset = True; c.target_space = c.owner_space = "WORLD"
    c = cam.constraints.new("TRACK_TO"); c.target, c.subtarget = rig, "Hips"
    c.track_axis, c.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    cam.data.lens = 35; sc.camera = cam
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sc.render.engine = "BLENDER_WORKBENCH"; sc.render.resolution_x, sc.render.resolution_y = 768, 432
    sc.render.filepath = f"{out}/figure_test/"
    bpy.ops.render.render(animation=True)
print("figure ok", out)
