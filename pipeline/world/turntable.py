"""A turntable of a GLB (her_mesh.py's her.glb): 8 views round her, lit evenly (OUT_DIR/view1..8.png; the launcher labels and strips them), so the mesh and
its texture can be judged from every side before she is skinned to the rig.
    blender -b -P pipeline/world/turntable.py -- MESH.glb OUT_DIR [--views 8]
"""
import bpy, math, sys
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb, out = argv[0], argv[1]
views = int(argv[argv.index("--views") + 1]) if "--views" in argv else 8
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=glb)
obs = [o for o in sc.objects if o.type == "MESH"]
lo = Vector([min(min((o.matrix_world @ Vector(c))[k] for c in o.bound_box) for o in obs) for k in range(3)])
hi = Vector([max(max((o.matrix_world @ Vector(c))[k] for c in o.bound_box) for o in obs) for k in range(3)])
ctr, size = (lo + hi) / 2, max(hi - lo)
print("mesh bounds", tuple(round(v, 3) for v in lo), tuple(round(v, 3) for v in hi),
      "faces", sum(len(o.data.polygons) for o in obs), flush=True)
sc.world = bpy.data.worlds.new("w"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.6, 0.62, 1)
sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 50
sc.render.engine = "CYCLES"; sc.cycles.samples = 64; sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = 512, 768
for k in range(views):
    a = 2 * math.pi * k / views
    d = size * 2.4
    # glTF is +Y up; Blender's importer turns it to +Z up, front facing -Y
    cam.location = ctr + Vector((d * math.sin(a), -d * math.cos(a), 0))
    cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{out}/view{k + 1}.png"            # view k+1 at 360*k/views degrees
    bpy.ops.render.render(write_still=True)
print("turntable ok", out, flush=True)
