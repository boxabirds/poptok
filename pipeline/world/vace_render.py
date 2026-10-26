"""Render a layout video with Wan 2.1 VACE 14B (Apache-2.0), to compare with LTX-2.5 Layout-To-Render on the same input
(2026-10-26). VACE takes a control video (here the textured 3D render, her figure in it) frame by frame, plus reference
images (her face and body, from her photo), and generates video that follows it.

    python3 pipeline/world/vace_render.py LAYOUT.mp4 OUT.mp4 --prompt "..." [--ref IMG ...] [--size 1280x720]
        [--frames 81] [--steps 30] [--scale 1.0] [--seed 42]
Runs in morph-spark on a Spark (models in /models; ~60 GB in bf16, no offload). Output at 16 fps (Wan's rate): the
layout is resampled to --frames frames over its own duration.
"""
from __future__ import annotations

import argparse, time

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("layout")
ap.add_argument("out")
ap.add_argument("--prompt", required=True)
ap.add_argument("--ref", action="append", default=[])
ap.add_argument("--size", default="1280x720")
ap.add_argument("--frames", type=int, default=81)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--scale", type=float, default=1.0, help="how strongly the layout steers (VACE conditioning_scale)")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
W, H = (int(v) for v in a.size.split("x"))

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline
from diffusers.utils import export_to_video, load_video

M = "/models/Wan2.1-VACE-14B-diffusers"
vae = AutoencoderKLWan.from_pretrained(M, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(M, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0 if H >= 720 else 3.0)
pipe.to("cuda")

src = load_video(a.layout)
idx = np.linspace(0, len(src) - 1, a.frames).round().astype(int)
video = [src[i].convert("RGB").resize((W, H), Image.LANCZOS) for i in idx]
refs = [Image.open(r).convert("RGB") for r in a.ref] or None
t = time.time()
frames = pipe(video=video, reference_images=refs, prompt=a.prompt,
              negative_prompt="blurry, distorted, deformed face, extra limbs, text, watermark, low quality",
              height=H, width=W, num_frames=a.frames, num_inference_steps=a.steps, guidance_scale=5.0,
              conditioning_scale=a.scale, generator=torch.Generator("cpu").manual_seed(a.seed)).frames[0]
export_to_video(frames, a.out, fps=16)
print(f"vace: {a.out} in {time.time() - t:.0f} s", flush=True)
