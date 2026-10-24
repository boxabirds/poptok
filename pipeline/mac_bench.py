"""Native Apple-GPU (MPS) runs of the still models on quintus, to time them against the Sparks. Our own calling code
only: the models' normal diffusers paths with device="mps" (no shims). Logs load time, seconds per still, peak MPS
memory and any op that fell back to the CPU (PYTORCH_ENABLE_MPS_FALLBACK=1 warnings).

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-qwenedit/bin/python pipeline/mac_bench.py qwen --out OUTDIR [--n 3]
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-moss/bin/python pipeline/mac_bench.py moss --out OUTDIR [--n 3]
"""
from __future__ import annotations

import argparse, json, os, sys, time, warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

M = os.path.expanduser("~/models")
SOLDIER = ("The woman from the reference image, the same person: keep her face, hair, eyes and age exactly, and dress "
           "her in a soldier camouflage combat uniform, her face covered in green and black camouflage paint (not the "
           "clothes in the reference); light her with the scene's own light, direction and colour, so she belongs in "
           "it. She is singing, lips parted. Medium close-up from the front, the woman singing with steady resolve, her "
           "eyes fixed on the distant horizon. A sharp, blue-white flash erupts behind her, illuminating the smudges of "
           "camouflage paint on her face and making her long, wavy blond hair stand out against the dark, rainy "
           "backdrop. The light from the flash creates a high-contrast rim around her features, emphasizing her calm "
           "expression as she delivers the line, embodying the pursuit of her inner peace amidst the chaos.")
BACKGROUND = ("Quiet rain on still streets, a child drawing animals, cracked walls, glowing inner furnace, dust settling, "
              "serene freedom, muted colors, ethereal atmosphere., no people")   # noble-hunt v1 shot 0 (Spark: 46 s)

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["qwen", "moss"])
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=3, help="stills after load (the first includes warm-up)")
ap.add_argument("--photo", default=os.path.expanduser("~/expts/poptok/pieces/artists/r.amber.jpeg"))
ap.add_argument("--size", default="1024x576")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
W, H = (int(v) for v in a.size.split("x"))
fallbacks = set()
_show = warnings.showwarning


def _catch(msg, cat, *rest, **kw):
    if "fall back to run on the CPU" in str(msg) or "MPS backend" in str(msg):
        fallbacks.add(str(msg).split("'")[1] if "'" in str(msg) else str(msg)[:120])
    _show(msg, cat, *rest, **kw)


warnings.showwarning = _catch
warnings.simplefilter("always")
assert torch.backends.mps.is_available()
peak = 0


def mem():
    global peak
    peak = max(peak, torch.mps.driver_allocated_memory())


t0 = time.time()
if a.model == "qwen":
    import qwen_edit
    qwen_edit.MODEL = f"{M}/Qwen-Image-Edit-2511"
    edit = qwen_edit.load(lightning=f"{M}/Qwen-Image-Edit-2511-Lightning/"
                          "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", device="mps")
    run = lambda seed: edit([a.photo], SOLDIER, seed=seed, size=(W, H))
else:
    import keyframe_model
    keyframe_model.MOSSImage.repo = f"{M}/MOSS-Image-SFT"
    r = keyframe_model.load("moss", offload=False, device="mps")
    run = lambda seed: r.render(None, [(BACKGROUND, 1.0)], 1.0, seed=seed, size=(W, H))[0]
load_s = time.time() - t0
mem()
print(f"load {load_s:.0f}s  mps {torch.mps.driver_allocated_memory() / 2**30:.1f} GB", flush=True)
times = []
for k in range(a.n):
    t = time.time()
    im = run(1057 + k)
    torch.mps.synchronize()
    times.append(time.time() - t)
    mem()
    im.save(os.path.join(a.out, f"{a.model}_{k}.png"))
    print(f"still {k}: {times[-1]:.1f}s  peak {peak / 2**30:.1f} GB", flush=True)
res = {"model": a.model, "size": a.size, "load_s": round(load_s, 1), "still_s": [round(x, 1) for x in times],
       "steady_s": round(sum(times[1:]) / max(1, len(times) - 1), 1), "peak_mps_gb": round(peak / 2**30, 1),
       "cpu_fallback_ops": sorted(fallbacks), "torch": torch.__version__,
       "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")}
json.dump(res, open(os.path.join(a.out, f"{a.model}_bench.json"), "w"), indent=1)
print(json.dumps(res), flush=True)
