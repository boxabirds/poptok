"""Stack C, chained: LTX-2.5 (+ the 2.3 Transition LoRA) as ONE continuous chain of clips, each continuing from the
previous clip's actual last frame (no reset to the keyframe), each changing scene ON a downbeat.

    python pipeline/ltx_chain.py OUT [--lead 0.9] [--lora-strength 1.0] [--size 1024x576]

Why: starting every bar from its keyframe (the earlier approach), but LTX finishes the change ~0.9 s into a clip and then holds
the new scene, so the next clip restarted that same scene from its key -> the scene played twice with a jump back.
Here clip j (scene j-1 -> scene j) starts LEAD seconds before downbeat j from the last frame written, with key j as a
guide at its last frame, and ends where clip j+1 starts. Clip 0 is key 0 animated (no guide).
Frames -> OUT/ltx_chain/%05d.png; OUT/ltx_chain/clips.json = [{j, a, e, downbeat}] (frames) for retime_to_beats.py.
"""
from __future__ import annotations

import argparse, json, math, os

import numpy as np
import torch
from PIL import Image

import ltx_pipelines.distilled as D
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--lead", type=float, default=0.9)
ap.add_argument("--lora-strength", type=float, default=1.0)
ap.add_argument("--size", default="1024x576")
ap.add_argument("--fps", type=float, default=24)
ap.add_argument("--merge", action="store_true", help="one scene per run of bars sharing a prompt: change scene only where "
                "the prompt changes; other bars continue the scene with no keyframe guide")
ap.add_argument("--dst", default="ltx_chain")
ap.add_argument("--max-clips", type=int, default=0, help="render only the first N clips (short permutation tests)")
ap.add_argument("--seed-base", type=int, default=3000)
ap.add_argument("--plan", default="shots.json", help="e.g. shots_merged.json from merge_similar.py")
ap.add_argument("--trigger", default="zhuanchang", help="the Transition LoRA's trigger word, ended onto every scene-change "
                "prompt as on its model card (valiantcat/LTX-2.3-Transition-LORA); '' to leave it out. Not used before "
                "2026-10-24, which is why slow songs got cuts and cross-fades instead of morphs")
ap.add_argument("--even-light", action="store_true", help="ask for constant lighting through the change: the Transition "
                "LoRA often uses a white light burst (15/50 changes in C3); the distilled model has no negative prompt")
a = ap.parse_args()
W, H = (int(x) for x in a.size.split("x"))
plan = json.load(open(os.path.join(a.out, a.plan)))
shots, style = plan["shots"], plan["style"]
tr = plan.get("treatment", {})                     # from treatment.py: how things move and how scenes transform
motion = tr.get("motion", "One continuous shot, slow cinematic camera movement.")
transition = tr.get("transition", "Smooth transformation, no cut.")
STILL = " The camera is locked off and the written word stays sharp, unchanged and perfectly legible."
od = os.path.join(a.out, a.dst); os.makedirs(od, exist_ok=True)
SC = {}                                            # scenes.json (scenes.py): per-scene transition_in and motion
if os.path.exists(os.path.join(a.out, "scenes.json")):
    for sc in json.load(open(os.path.join(a.out, "scenes.json")))["scenes"]:
        for b in range(sc["bars"][0], sc["bars"][1] + 1):
            SC[b] = sc
TRIG = f", {a.trigger}" if a.trigger else ""
keys = os.path.join(a.out, "keys")

n_total = int(round(shots[-1]["end"] * a.fps))
starts = [0] + [int(round((s["start"] - a.lead) * a.fps)) for s in shots[1:]] + [n_total]
clips = [{"j": j, "a": starts[j], "e": starts[j + 1], "downbeat": int(round(shots[j]["start"] * a.fps)),
          "change": j > 0 and (not a.merge or shots[j]["prompt"] != shots[j - 1]["prompt"])} for j in range(len(shots))]
json.dump(clips, open(os.path.join(od, "clips.json"), "w"), indent=1)

MODELS = os.environ.get("LTX_MODELS", "/models")             # containers mount ~/models at /models; quintus runs natively
M = f"{MODELS}/ltx-2.5"
paths = ModelPaths.from_split(transformer_path=f"{M}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                              text_encoder_path=f"{M}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                              video_vae_path=f"{M}/vae/ltx-2.5-video-vae-bf16.safetensors",
                              audio_vae_path=f"{M}/vae/ltx-2.5-audio-vae-bf16.safetensors")
loras = (LoraPathStrengthAndSDOps(f"{MODELS}/ltx-2.3/ltx2.3-transition.safetensors", a.lora_strength,
                                  LTXV_LORA_COMFY_RENAMING_MAP),) if a.lora_strength > 0 else ()
pipe = D.DistilledPipeline(model_paths=paths,
                           spatial_upsampler_path=f"{M}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                           loras=loras, quantization=None, offload_mode=OffloadMode.NONE)

tmp = os.path.join(a.out, "ltx_tmp"); os.makedirs(tmp, exist_ok=True)
fa, fb = os.path.join(tmp, "chainA.png"), os.path.join(tmp, "chainB.png")
for c in clips[:a.max_clips or None]:
    j, f0, f1 = c["j"], c["a"], c["e"]
    need = f1 - f0
    if os.path.exists(os.path.join(od, f"{f1 - 1:05d}.png")):
        continue
    n = 8 * math.ceil((need - 1) / 8) + 1
    first = os.path.join(keys, "shot_000.png") if j == 0 else os.path.join(od, f"{f0 - 1:05d}.png")
    Image.open(first).convert("RGB").resize((W, H), Image.LANCZOS).save(fa)
    images = [ImageConditioningInput.at(path=fa, frame_idx=0, strength=1.0)]
    if not c["change"]:
        prompt = f"{shots[j]['prompt']} {style}. {SC.get(j, {}).get('motion') or motion}" + (STILL if shots[j].get("word") else "")
    else:
        Image.open(os.path.join(keys, f"shot_{j:03d}.png")).convert("RGB").resize((W, H), Image.LANCZOS).save(fb)
        images.append(ImageConditioningInput.guide(path=fb, frame_idx=n - 1, strength=1.0))
        light = (" The lighting and exposure stay constant throughout (no flash, no burst of light); the shapes themselves reshape into the "
                 "new scene.") if a.even_light else ""
        t_in = SC.get(j, {}).get("transition_in", "")
        if t_in:                                   # a written A->B morph (scenes.py transitions), the LoRA's own style
            prompt = (f"One continuous shot, no cut: {t_in} It settles on {shots[j]['prompt']} {style}.{light}{TRIG}"
                      + (STILL if shots[j].get("word") else ""))
        else:
            prompt = (f"One continuous shot: the scene transforms into the next scene, {shots[j]['prompt']} "
                      f"{style}. {transition} No cut.{light}{TRIG}" + (STILL if shots[j].get("word") else ""))
    with torch.inference_mode():
        out = pipe(prompt=prompt, seed=a.seed_base + j, height=H, width=W, frame_rate=a.fps, images=images, num_frames=n)
        fr = []
        for seg in out.video:
            seg = seg.float().cpu()
            seg = seg * 255 if seg.max() <= 1.5 else seg
            fr += list(seg.clamp(0, 255).to(torch.uint8).numpy())
    # frame 0 repeats the conditioning frame (= the previous clip's last frame): drop it, then resample to `need`
    src = fr[1:] if j > 0 else fr
    idx = np.round(np.linspace(0, len(src) - 1, need)).astype(int)
    for k, i in enumerate(idx):
        Image.fromarray(src[i]).save(os.path.join(od, f"{f0 + k:05d}.png"))
    print(f"clip {j:3d}: frames {f0}-{f1 - 1} ({need}, rendered {n}), downbeat {c['downbeat']}", flush=True)
print("done")
