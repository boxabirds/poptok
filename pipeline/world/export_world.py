"""Export a painted world (paint_world.py render's painted.blend) for viewing outside Blender (2026-10-26): its image
paths made relative (the .blend first saved container paths, /work/..., so on the user's Mac it opened untextured and
"won't render"), and a .glb (textures as JPEG, her animation included) that a browser viewer or any 3D app opens.

    docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work blender -b DIR/painted.blend \
        -P pipeline/world/export_world.py -- DIR
Writes DIR/painted.blend (paths relative, textures packed in) and DIR/painted.glb.
"""
import os, sys

import bpy

out = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
for o in [o for o in bpy.data.objects if o.type == "CAMERA" and o.name.startswith(("view_", "tv_", "cand_"))]:
    bpy.data.objects.remove(o, do_unlink=True)          # painting cameras: clutter in a viewer
for m in bpy.data.materials:                             # emission-only paint -> unlit-looking glTF: the texture as
    if not m.use_nodes:                                  # emissive and base colour (viewers light it either way)
        continue
    nt = m.node_tree
    em = next((n for n in nt.nodes if n.type == "EMISSION"), None)
    tex = next((n for n in nt.nodes if n.type == "TEX_IMAGE"), None)
    outn = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if em and outn:
        bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
        if tex:
            nt.links.new(tex.outputs["Color"], bs.inputs["Base Color"])
            nt.links.new(tex.outputs["Color"], bs.inputs["Emission Color"])
        else:
            bs.inputs["Base Color"].default_value = em.inputs["Color"].default_value
            bs.inputs["Emission Color"].default_value = em.inputs["Color"].default_value
        bs.inputs["Emission Strength"].default_value = 1.0
        bs.inputs["Roughness"].default_value = 1.0
        nt.links.new(bs.outputs["BSDF"], outn.inputs["Surface"])
bpy.ops.file.pack_all()                                  # the .blend carries its textures: opens anywhere
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "painted.blend"))
bpy.ops.export_scene.gltf(filepath=os.path.join(out, "painted.glb"), export_format="GLB", export_image_format="JPEG",
                          export_image_quality=80, export_cameras=True, export_lights=False, export_animations=True)
print("exported", out)
