"""Transition test: one LTX-2.5 + Transition LoRA clip from still A to still B, with different prompt wordings, to see
which give a real scene-specific morph rather than a cut or a cross-fade.

    python pipeline/transition_test.py TESTS.json OUT_DIR
TESTS.json: [{"name", "a": still A, "b": still B, "seconds", "variants": {label: prompt}, optional "guide": end-still
strength (default 1.0), "seed" (default 3000)}]. LORA_STRENGTH env (default 1.0; 0 = no Transition LoRA). Writes OUT_DIR/<name>_<label>.mp4
and OUT_DIR/<name>_sheet.jpg (8 frames per variant). Runs in the ltx2 image (models in /models).
"""
from __future__ import annotations

import json, math, os, subprocess, sys

import numpy as np
import torch
from PIL import Image, ImageDraw

import ltx_pipelines.distilled as D
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode

tests, out = json.load(open(sys.argv[1])), sys.argv[2]
os.makedirs(out, exist_ok=True)
W, H, FPS = 1024, 576, 24
M = f"{os.environ.get('LTX_MODELS', '/models')}/ltx-2.5"
LORA = float(os.environ.get("LORA_STRENGTH", "1.0"))
pipe = D.DistilledPipeline(
    model_paths=ModelPaths.from_split(
        transformer_path=f"{M}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        text_encoder_path=f"{M}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        video_vae_path=f"{M}/vae/ltx-2.5-video-vae-bf16.safetensors",
        audio_vae_path=f"{M}/vae/ltx-2.5-audio-vae-bf16.safetensors"),
    spatial_upsampler_path=f"{M}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    loras=(LoraPathStrengthAndSDOps(f"{os.path.dirname(M)}/ltx-2.3/ltx2.3-transition.safetensors", LORA,
                                    LTXV_LORA_COMFY_RENAMING_MAP),) if LORA > 0 else (),
    quantization=None, offload_mode=OffloadMode.NONE)
for t in tests:
    n = 8 * math.ceil((int(t["seconds"] * FPS) - 1) / 8) + 1
    fa, fb = os.path.join(out, f"{t['name']}_A.png"), os.path.join(out, f"{t['name']}_B.png")
    Image.open(t["a"]).convert("RGB").resize((W, H)).save(fa)
    Image.open(t["b"]).convert("RGB").resize((W, H)).save(fb)
    rows = []
    for label, prompt in t["variants"].items():
        with torch.inference_mode():
            res = pipe(prompt=prompt, seed=t.get("seed", 3000), height=H, width=W, frame_rate=FPS, num_frames=n,
                       images=[ImageConditioningInput.at(path=fa, frame_idx=0, strength=1.0),
                               ImageConditioningInput.guide(path=fb, frame_idx=n - 1, strength=t.get("guide", 1.0))])
        fr = []
        for seg in res.video:
            seg = seg.float().cpu()
            seg = seg * 255 if seg.max() <= 1.5 else seg
            fr += list(seg.clamp(0, 255).to(torch.uint8).numpy())
        d = os.path.join(out, f"{t['name']}_{label}")
        os.makedirs(d, exist_ok=True)
        for k, f in enumerate(fr):
            Image.fromarray(f).save(os.path.join(d, f"{k:04d}.png"))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(d, "%04d.png"), "-c:v",
                        "libx264", "-pix_fmt", "yuv420p", "-crf", "18", f"{d}.mp4"], check=True)
        rows.append((label, [fr[i] for i in np.linspace(0, len(fr) - 1, 8).astype(int)]))
        print(f"{t['name']} {label}: {len(fr)} frames", flush=True)
    w, h = 256, 144
    sheet = Image.new("RGB", (w * 8, (h + 16) * len(rows)), "white")
    dr = ImageDraw.Draw(sheet)
    for r, (label, frames) in enumerate(rows):
        for c, f in enumerate(frames):
            sheet.paste(Image.fromarray(f).resize((w, h)), (c * w, r * (h + 16) + 16))
        dr.text((4, r * (h + 16) + 2), label, fill="black")
    sheet.save(os.path.join(out, f"{t['name']}_sheet.jpg"), quality=85)
