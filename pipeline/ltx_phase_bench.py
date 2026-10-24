"""Where LTX video time goes, per phase, on this box: the chain's real workload (LTX-2.5 distilled + the 2.3 Transition
LoRA, 1024x576, one scene-change clip: first frame + end guide), timed per pipeline component with a device sync, so
boxes can be compared phase by phase (docs/hardware-performance.md).

    python pipeline/ltx_phase_bench.py A.png B.png OUT.json [--frames 65] [--runs 3] [--prompt "..."]
Runs in the ltx2 image on a Spark (models in /models) or natively on quintus (LTX_MODELS=~/models). Run 1 is warm-up;
the summary is the median of the rest. Phases: text encode (Gemma), image conditioning (VAE encode of A/B), denoising
(stage 1 + stage 2 transformer), upsample, decode (VAE decode, incl. any decode done while the result is iterated),
frame save (PNG), plus total wall time per clip.
"""
from __future__ import annotations

import argparse, json, os, platform, statistics, tempfile, time

import torch
from PIL import Image

import ltx_pipelines.distilled as D
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("out")
ap.add_argument("--frames", type=int, default=65)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--prompt", default="One continuous shot, no cut: the window glass shatters outward into a blizzard of "
                "shards that become crushed plastic bottles swirling over wet railway tracks at night, zhuanchang")
a = ap.parse_args()
W, H, FPS = 1024, 576, 24
M = f"{os.environ.get('LTX_MODELS', '/models')}/ltx-2.5"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


T: dict[str, float] = {}


def timed(cls, name):
    f = cls.__call__
    def wrap(self, *x, **k):
        sync(); t = time.time()
        r = f(self, *x, **k)
        sync(); T[name] = T.get(name, 0.0) + time.time() - t
        return r
    cls.__call__ = wrap


t0 = time.time()
pipe = D.DistilledPipeline(
    model_paths=ModelPaths.from_split(
        transformer_path=f"{M}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        text_encoder_path=f"{M}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        video_vae_path=f"{M}/vae/ltx-2.5-video-vae-bf16.safetensors",
        audio_vae_path=f"{M}/vae/ltx-2.5-audio-vae-bf16.safetensors"),
    spatial_upsampler_path=f"{M}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    loras=(LoraPathStrengthAndSDOps(f"{os.path.dirname(M)}/ltx-2.3/ltx2.3-transition.safetensors", 1.0,
                                    LTXV_LORA_COMFY_RENAMING_MAP),),
    quantization=None, offload_mode=OffloadMode.NONE)
build_s = time.time() - t0
for obj, name in ((pipe.prompt_encoder, "text_encode"), (pipe.image_conditioner, "image_condition"),
                  (pipe.stage, "denoise"), (pipe.upsampler, "upsample"), (pipe.video_decoder, "decode"),
                  (pipe.audio_decoder, "audio_decode")):
    timed(type(obj), name)

def timed_fn(mod, fname, name):                           # module-level chunk helpers the pipeline calls directly
    f = getattr(mod, fname, None)
    if f is None:
        return
    def wrap(*x, **k):
        sync(); t = time.time()
        r = f(*x, **k)
        if hasattr(r, "__next__"):                         # a generator: do its work inside the timer
            r = iter(list(r))
        sync(); T[name] = T.get(name, 0.0) + time.time() - t
        return r
    setattr(mod, fname, wrap)


for fname, name in (("decode_chunks", "decode_chunks"), ("denoise_chunks", "denoise_chunks"),
                    ("spatially_upsample_chunks", "upsample_chunks"), ("split_decoded_chunks", "split_chunks")):
    timed_fn(D, fname, name)

tmp = tempfile.mkdtemp()
fa, fb = os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png")
Image.open(a.a).convert("RGB").resize((W, H)).save(fa)
Image.open(a.b).convert("RGB").resize((W, H)).save(fb)
n = a.frames
runs = []
for r in range(a.runs):
    T.clear()
    sync(); t = time.time()
    with torch.inference_mode():
        res = pipe(prompt=a.prompt, seed=3000, height=H, width=W, frame_rate=FPS, num_frames=n,
                   images=[ImageConditioningInput.at(path=fa, frame_idx=0, strength=1.0),
                           ImageConditioningInput.guide(path=fb, frame_idx=n - 1, strength=1.0)])
        sync(); t_call = time.time() - t
        fr = []
        t1 = time.time()
        for seg in res.video:                              # decoding may happen lazily, while iterating
            seg = seg.float().cpu()
            seg = seg * 255 if seg.max() <= 1.5 else seg
            fr += list(seg.clamp(0, 255).to(torch.uint8).numpy())
        sync(); t_iter = time.time() - t1
    t2 = time.time()
    for k, f in enumerate(fr):
        Image.fromarray(f).save(os.path.join(tmp, f"{k:04d}.png"))
    t_save = time.time() - t2
    total = time.time() - t
    row = dict(T, pipe_call=t_call, iterate_result=t_iter, frame_save=t_save, total=total, frames=len(fr))
    runs.append(row)
    print(f"run {r + 1}: " + ", ".join(f"{k} {v:.1f}" for k, v in row.items() if k != "frames") + f", {len(fr)} frames",
          flush=True)

keep = runs[1:] or runs
summary = {k: round(statistics.median(x[k] for x in keep), 2) for k in keep[0]}
summary["s_per_frame"] = round(summary["total"] / max(summary["frames"], 1), 3)
doc = dict(host=platform.node(), device=("cuda:" + torch.cuda.get_device_name(0)) if torch.cuda.is_available()
           else ("mps" if torch.backends.mps.is_available() else "cpu"), torch=torch.__version__, build_s=round(build_s, 1),
           frames_requested=n, size=f"{W}x{H}", runs=runs, median_after_warmup=summary)
json.dump(doc, open(a.out, "w"), indent=1)
print("median after warm-up:", summary)
