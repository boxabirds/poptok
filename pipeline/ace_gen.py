"""ACE-Step 1.5: song from a caption (music style, < 512 chars) and lyrics. Runs in the acestep image.

    python pipeline/ace_gen.py OUT --caption-file CAPTION.txt --lyrics-file LYRICS.txt [--variants xl-sft] [--seeds 1,2]
        [--duration -1] [--bpm 0] [--name song]

Lyrics may carry descriptor tags such as [Chorus - half-shouted, anthemic, gang vocals].
The 5 Hz LM (4B) plans the audio codes ("thinking") but does NOT rewrite the caption or lyrics (use_cot_caption=False,
use_cot_lyrics=False), so the brief is what drives the music. English vocals; --bpm 0 lets the LM choose.
"""
from __future__ import annotations

import argparse, json, os, shutil, time
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler
from acestep.model_downloader import ensure_lm_model

VARIANTS = {  # name: (DiT config, steps, guidance) - turbo models ignore guidance
    "xl-sft": ("acestep-v15-xl-sft", 50, 7.0),
    "xl-turbo": ("acestep-v15-xl-turbo", 8, 7.0),
    "turbo": ("acestep-v15-turbo", 8, 7.0),
}
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--variants", default="xl-sft,xl-turbo,turbo")
ap.add_argument("--seeds", default="1,2")
ap.add_argument("--duration", type=float, default=-1, help="seconds (10-600); -1 = the LM picks a length to fit the lyrics")
ap.add_argument("--lm", default="acestep-5Hz-lm-4B")
ap.add_argument("--lyrics-file", required=True)
ap.add_argument("--name", default="song", help="output file prefix")
ap.add_argument("--caption-file", required=True, help="< 512 characters")
ap.add_argument("--bpm", type=int, default=0, help="0 = let the LM choose")
ap.add_argument("--offload", action="store_true", help="CPU offload (off: run on a box where XL + the 4B LM fit)")
a = ap.parse_args()
caption = open(a.caption_file).read().strip()
lyrics = open(a.lyrics_file).read().strip()
assert len(caption) < 512 and len(lyrics) < 4096, (len(caption), len(lyrics))
os.makedirs(a.out, exist_ok=True)
ckpt = os.environ.get("ACESTEP_CHECKPOINTS_DIR", "/app/checkpoints")

print(ensure_lm_model(a.lm, checkpoints_dir=Path(ckpt)), flush=True)
llm = LLMHandler()
msg, ok = llm.initialize(checkpoint_dir=ckpt, lm_model_path=a.lm, backend="pt", device="cuda", offload_to_cpu=a.offload)
print(msg, flush=True)
if not ok:                       # without the LM, generate_music silently runs DiT-only (use_lm=False)
    raise SystemExit(f"LM {a.lm} failed to load")
dit = AceStepHandler()
log = []
for name in a.variants.split(","):
    cfg, steps, gs = VARIANTS[name]
    print(dit.initialize_service(project_root="/app", config_path=cfg, device="cuda", offload_to_cpu=a.offload), flush=True)
    for seed in (int(s) for s in a.seeds.split(",")):
        t = time.time()
        params = GenerationParams(caption=caption, lyrics=lyrics, bpm=a.bpm or None, vocal_language="en", duration=a.duration,
                                  inference_steps=steps, guidance_scale=gs, seed=seed, thinking=True,
                                  use_cot_caption=False, use_cot_lyrics=False)
        conf = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
        tmp = os.path.join(a.out, "tmp"); os.makedirs(tmp, exist_ok=True)
        r = generate_music(dit, llm, params, conf, save_dir=tmp)
        if not r.success:
            print(f"{name} seed {seed}: FAILED {r.error}", flush=True); continue
        au = r.audios[0]
        dst = os.path.join(a.out, f"{a.name}_{name}_s{seed}.wav")
        shutil.move(au["path"], dst)
        p = au.get("params", {})
        meta = {k: p.get(k) for k in ("bpm", "keyscale", "duration", "timesignature", "seed")}
        log.append({"lm": a.lm, "file": dst, "variant": name, "dit": cfg, "steps": steps, "secs": round(time.time() - t, 1), **meta})
        print(json.dumps(log[-1]), flush=True)
json.dump({"caption": caption, "lyrics": lyrics, "runs": log}, open(os.path.join(a.out, "runs.json"), "w"), indent=1)
