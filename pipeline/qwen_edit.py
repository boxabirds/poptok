"""Qwen-Image-Edit-2511 with reference images, by default step-distilled with the Lightning LoRA (8 steps, no true CFG:
about 60 s a still with 3 references on a DGX Spark, against about 580 s for 40 steps with true CFG; same identity
hold, POC 20). Needs ~60 GB of GPU memory: a Spark, no offload.

    from qwen_edit import load
    edit = load()                                   # or load(lightning=None) for the full 40-step model
    image = edit(["ref1.png", "ref2.png"], "prompt", seed=1, size=(1024, 576))
"""
from __future__ import annotations

import math

import torch
from PIL import Image

MODEL = "/models/Qwen-Image-Edit-2511"
LIGHTNING = "/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"


def load(lightning=LIGHTNING, steps=None, cfg=4.0, device="cuda"):
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    if lightning:                       # the scheduler settings from ModelTC/Qwen-Image-Lightning
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
            "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
            "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
            "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False})
        pipe.load_lora_weights(lightning)
        pipe.fuse_lora()
        steps, cfg = steps or (4 if "4steps" in lightning else 8), 1.0
    steps = steps or 40
    pipe = pipe.to(device)                  # "mps" on quintus (pipeline/mac_bench.py)
    pipe.set_progress_bar_config(disable=True)

    def edit(refs, prompt, seed=0, size=(1024, 576)):
        images = [r if isinstance(r, Image.Image) else Image.open(r).convert("RGB") for r in refs]
        return pipe(image=images, prompt=prompt, negative_prompt=" ", true_cfg_scale=cfg, num_inference_steps=steps,
                    height=size[1], width=size[0], generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    edit.steps, edit.cfg = steps, cfg
    return edit
