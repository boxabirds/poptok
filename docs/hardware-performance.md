# Hardware performance by process step

What each box actually achieved on each step of the pipeline, measured in real runs. It's the evidence for advice on
which hardware suits which local-AI job, so keep it factual and keep filling it in: every time a step is timed on a
box, add or update the cell (value, date, source). See CLAUDE.md.

**Boxes**

| Box | Hardware | Memory for models | Software stack |
|---|---|---|---|
| gruntus | RTX 4090, x86_64 | 24 GB VRAM | CUDA 12.x, docker |
| thing1, thing2 | DGX Spark (GB10, ARM64) | 128 GB unified | CUDA 13 (sm_121), docker |
| tritus | AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S gfx1151) | 128 GB unified (~96–110 GB usable by the GPU) | ROCm 7.x, podman |
| quintus | MacBook Pro M5 Max | 128 GB unified | Metal (MPS), native uv venv |

**Rules behind the n/a cells:** no CPU offload (a model that doesn't fit in GPU memory doesn't run on that box), one GPU
job per box. So n/a means one of: *too big* (doesn't fit without offload), *no port* (the software needs CUDA/xformers
etc. and hasn't been ported), *not tried*.

## Steps

Times are per unit stated; "steady" means after warm-up. Sources: POC docs (`pocs/NN`), the log (`log.md`), run logs.

| Step | Model and settings | gruntus (4090) | thing1/2 (Spark) | tritus (Strix Halo) | quintus (M5 Max) | Notes |
|---|---|---|---|---|---|---|
| Stems + transcription | Demucs + Whisper (word timestamps), 4 min song | ~5 min/song | not tried | not tried | not tried | CPU/GPU mix; gruntus only so far |
| LLM text work (prompts, cues, transitions, scene merge) | Qwen3.8-27B Q4_K_XL, llama.cpp, 16k ctx | ~2–3 min for 40 transitions (2026-10-24) | not tried | not tried | not tried | Fits in 24 GB at Q4 |
| Segmentation (critic) | LLMDet-large (open-vocab boxes, 6 prompt passes) + SAM 2.1 hiera-large masks + per-person crop pass (arm/hand), 1280×720 frames, fp32, `docker/segment` | **3.7 s/frame** mean (2.6–6.1; 6.1 = 5 people), of which detection 2.5, SAM 0.16, person crops 1.0; peak 9.3 GB allocated; models load 3 s (2026-10-24, pocs/28, 11 frames) | not tried | not tried | not tried | Qwen crop question on each person (Qwen3.8-27B Q4 + mmproj, 1024 px crop): **1.9 s/crop** (11 crops); the whole-frame critic (3 views) 22 s/frame mean (12–47). Segment + Qwen exceed 24 GB together: run them one after the other |
| Image: MOSS-Image keyframe | MOSS-Image-SFT, 1024×576 | n/a: too big (33 GB). Was ~22 s with offload before the no-offload rule | **~45 s/still** | 386 s with defaults → **78 s** with CPU offload off (it only copies within the same unified RAM) and AOTriton attention on (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; off by default on gfx1151); TunableOp and 2 jobs at once no faster (pocs/14) | **blocked by the model's own code**: loads on MPS (19 s, 25.8 GB), but `pipeline_moss_image.py` (MOSS's custom diffusers pipeline, l. 236–237) builds the timestep sinusoid in float64 on the model's device; MPS has no float64 and `PYTORCH_ENABLE_MPS_FALLBACK=1` doesn't cover dtypes. Needs a one-line edit of that file (not made: stop rule) (2026-10-24, `pieces/_bench/mac-ports/moss*/run.log`) | ROCm 7.14 not faster (pocs/15) |
| Image: Qwen-Image-2.1 | single prompt / blended | **21.8 s** / 40.3 s | not tried | 94 s best, often 500–3600 s (pocs/16) | not tried | tritus ruled out for Qwen-Image |
| Image edit: Qwen-Image-Edit-2511 (Amber from her photo) | Lightning LoRA, 8 steps, cfg 1 | n/a: too big | **~24 s/still** (1 ref), ~61 s (3 refs); 40 steps + true CFG: 230–580 s (pocs/22) | not tried | **33.3 s/still** steady (34.3 first, 1 ref, 1024×576), load 36 s, peak 67 GB MPS; all ops on MPS, no CPU fallback needed; torch 2.14.1 / diffusers 0.42 native venv, `pipeline/mac_bench.py qwen` (2026-10-24, `pieces/_bench/mac-ports/qwen/`) → ~1.4× a Spark | Lightning: same quality, ~10× faster |
| Image edit: instruction fix of a keyframe (`fix_keys.py`) | Qwen-Image-Edit-2511 Lightning 8 steps, 1024×576 | n/a: too big | not timed | not tried | **34–35 s** (still only), **60–73 s** (still + her photo) (2026-10-24, 6 edits) | |
| Video: LTX-2.5 chain (+ 2.3 Transition LoRA) | 22B distilled bf16, 1024×576, 8 + 3 (upscale) steps | n/a: too big (~44 GB transformer + 12B Gemma) | **0.95–1.1 s/frame** (6112 frames in 97–114 min, 2026-10-24) | see the per-phase row: ~2 s/frame (`docker/ltx-rocm/`, 2026-10-24) | **1.47 s/frame**; short test 353 s vs Spark 377 s (pocs/17) | Denoising per step (2026-10-24 logs): Spark 1.00 s (stage 1) / 3.95 s (stage 2) on 65-frame clips; quintus 1.15 / 3.90 s on 49-frame clips → Mac ~25% slower per frame, compute-bound (Mac has more memory bandwidth). Denoising is only ~1/3 of the time per frame; the rest (VAE decode, where the neighbourhood attention is Triton on the Spark but an eager PyTorch fallback on the Mac; Gemma text encode per clip; frame I/O) is unprofiled. Shorter clips (slower songs) raise per-frame cost on any box. tritus: a report gives 72.8 → 49 s/step for LTX-2.3 22B bf16 with `HIP_FORCE_DEV_KERNARG=1` + `expandable_segments` (TheRock #2845), not measured here |
| Video: LTX per phase, one 65-frame scene-change clip (`ltx_phase_bench.py`, median of runs 2–3) | text encode / image cond. / denoise / decode / frame save / **total** | n/a: too big | thing1: 7.3 / 0.6 / 37.2 / 12.8 / 9.0 / **67.5 s (1.04 s/frame)** | 16–22 / 0.9 / 70.5 / 29 / 9.9 / **127–134 s (1.96–2.07 s/frame)**; first run on an empty cache +1125 s of MIOpen tuning; `HIP_FORCE_DEV_KERNARG` gave no gain (2026-10-24) | 7.6 / 1.7 / 46.9 / 22.0 / 2.0 / **81.0 s (1.25 s/frame)** | Denoise ~55% everywhere. Mac's decode is 1.7× the Spark's (no Triton: eager neighbourhood attention); Spark's PNG saving is 9 s/clip (13%) vs 2 s on the Mac; text encode repeats for every clip of a scene |
| Video: LTX chain, same full song on both boxes (head-to-head) | noble-hunt (4:02, 5818 frames, 33 scenes, same scene structure), 2026-10-24 production runs | n/a: too big | thing1: **123 min (1.27 s/frame)**, 18:03–20:06 | not run | **155 min (1.60 s/frame)**, 19:05–21:40, incl. copying frames back | Mac **1.26×** the Spark on identical work. Prompts differed (workshop vs Offline recipe), which doesn't change the compute |
| Video: one LTX transition test clip | same, 46–65 frames | n/a | not timed | not tried | ~2–3 min/clip incl. model load (2026-10-24) | |
| Lip-sync: InfiniteTalk (Wan2.1-I2V-14B) | 480p, lightx2v LoRA, 4 steps, streaming | **~62 s per second of video** (10.6 s in 661 s; 33.5–34.6 s/step), official `infinitetalk_single_int8_lora` + fp8 T5, all on GPU, no offload; fits only with memory patches (`docker/infinitetalk-x86/vram_fit.py`: drop T5/CLIP after use, bf16 scales, per-head ref-attention, sliced RoPE, tiled bf16 VAE); peak 24.08 of 24.56 GB. Quality close to bf16; slow push-in and background colour drift (2026-10-24) | **~2.4 min per second of video** (11.8 s in 28 min); 40-step full quality ~5 h for 16 s (pocs/19) | bf16 ROCm: **5.8 min per second of video** (10.7 s in 61.6 min, 197 s/step), matches the Spark render; needed an xformers→SDPA shim, contiguous q/k/v (5.6× faster attention), 3D convs rewritten as 2D (VAE 4.6× faster) (`docker/infinitetalk-rocm/`, 2026-10-24) | **blocked by dependencies** (no shims allowed): `xfuser` 0.4.1 crashes on import (`torch.cuda.get_device_name` in `xfuser/envs.py`), imported unconditionally by Wan's attention; `decord` has no macOS arm64 wheel; the macOS `xformers` wheel has no `memory_efficient_attention`; `flash-attn` has no macOS build and Wan calls `flash_attention()` directly (2026-10-24; weights not copied) | bf16 14B + T5 too big for 24 GB; int8 on the 4090 is ~2.6× faster than bf16 on a Spark |
| Lip-sync: LongCat-Video-Avatar 1.5 | distilled, 480p | n/a: too big | ran (rejected on quality: mis-articulates "r") | not tried | n/a | |
| Song generation: ACE-Step 1.5 | XL-sft + 4B LM, 150 s song | n/a: too big with the LM (>24 GB); without LM 23 s/song; 2B turbo + LM 45 s | not tried (no ARM image) | not tried | not tried | pocs/09 |
| Finish: retime, grade, mux | ffmpeg + numpy, 4 min 1024×576 | ~6 min (CPU) | not tried | not tried | not tried | CPU-bound |

## What this says so far (for advice)

- **24 GB cards (4090)** run the LLM, Whisper/Demucs and mid-size image models fast, but not today's big video and
  video-edit models at bf16 without offload (LTX 22B, Wan 14B, Qwen-Image-Edit, MOSS). With official int8 weights and
  memory patches, Wan-14B lip-sync (InfiniteTalk) fits and runs ~2.6× faster than bf16 on a DGX Spark: raw compute
  wins once the model fits, at some cost in fidelity and with no headroom.
- **128 GB unified memory (Spark, Strix Halo, M5 Max)** fits every model here at full precision. Speed differs:
  - DGX Spark: the reference; CUDA, so everything runs (with some ARM build work).
  - M5 Max: close to the Spark for LTX video (1.0–1.5×) and Qwen-Image-Edit stills (33 s vs 24 s, natively, no
    changes); MOSS blocked by one float64 line in its own pipeline file, InfiniteTalk by CUDA-only dependencies
    (xfuser, decord, xformers, flash-attn): software, not a hardware limit.
  - Strix Halo: ~1.7× slower than a Spark for MOSS once tuned (offload off, AOTriton), much slower for Qwen-Image;
    settings matter more than the chip (5× from two settings: offload off, AOTriton attention on).
