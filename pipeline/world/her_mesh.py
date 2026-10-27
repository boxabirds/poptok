"""Her as a textured 3D mesh, from one full-length A-pose image (her_full_body.py), with TRELLIS.2 (MIT; weights
microsoft/TRELLIS.2-4B, MIT; image features from DINOv3, Meta's licence, gated: accepted by the user 2026-10-27).
The mesh is skinned to the MakeHuman rig in Blender afterwards (figure.py --mesh), so the CMU mocap drives her.

    python3 pipeline/world/her_mesh.py IMAGE_RGBA.png OUT_DIR [--seed 1]
Runs in the trellis2 image on a Spark (HF_HOME=/models/hf, offline), no offloading (low_vram off: on the Spark's
unified memory moving models to the CPU would gain nothing anyway). Writes OUT_DIR/her.glb (a turntable is rendered in Blender, not here).
"""
from __future__ import annotations

import argparse, os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import torch
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipe.low_vram = False
pipe.cuda()
img = Image.open(a.image)
assert img.mode == "RGBA", "give an RGBA image with the background removed (no background-removal model is loaded)"
mesh = pipe.run(img, seed=a.seed)[0]
mesh.simplify(16777216)
glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                                 attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                 decimation_target=200000, texture_size=2048, remesh=True, remesh_band=1, remesh_project=0,
                                 verbose=True)
glb.export(f"{a.out}/her.glb", extension_webp=False)
print("her mesh ok", a.out)
