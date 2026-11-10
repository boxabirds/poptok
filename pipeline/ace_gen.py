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
ap.add_argument("--offload", action="store_true", help="CPU offload: models swapped in and out (recorded in runs.json)")
ap.add_argument("--codes-file", help="reuse a take's plan (its <name>.codes.txt): same melody, structure and length, new "
                "performance per seed; the planner then only fills in metadata")
a = ap.parse_args()
caption = open(a.caption_file).read().strip()
lyrics = open(a.lyrics_file).read().strip()
codes = open(a.codes_file).read().strip() if a.codes_file else ""
if codes and a.duration < 0:     # the plan fixes the length (5 codes a second); left to the planner it picks its own
    a.duration = round(codes.count("<|audio_code_") / 5, 1)   # (260 s against a 215 s plan, and ran out of memory)
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
                                  use_cot_caption=False, use_cot_lyrics=False, audio_codes=codes)
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
        if p.get("audio_codes"):                              # the planner's plan of this take, reusable with --codes-file
            open(dst[:-4] + ".codes.txt", "w").write(p["audio_codes"])
        log.append({"lm": a.lm, "file": dst, "variant": name, "dit": cfg, "steps": steps, "secs": round(time.time() - t, 1),
                    "offload": bool(a.offload), "codes_from": a.codes_file, **meta})
        print(json.dumps(log[-1]), flush=True)
rp = os.path.join(a.out, "runs.json")                      # added to, so later rounds keep earlier takes' records
prev = json.load(open(rp))["runs"] if os.path.exists(rp) else []
json.dump({"caption": caption, "lyrics": lyrics, "runs": prev + log}, open(rp, "w"), indent=1)
