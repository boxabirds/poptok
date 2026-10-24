# POC 19: Singer lip-sync from one photo (InfiniteTalk)

**Date:** 2026-10-23 · **Status:** passed in fast mode; your verdict on how it looks pending · **Piece:** chronophobia

## Hypothesis

MeiGen-AI InfiniteTalk (Wan2.1-I2V-14B-480P + an audio adapter driven by wav2vec2) can make the singer's photo sing
a chorus of Chronophobia with believable lip-sync. Then performance shots of her can cut in with the lyric scenes.

## Test

- **Image:** `docker/infinitetalk` on thing1 (CUDA 13, torch cu130, xformers v0.0.35 built for GB10 sm_121,
  InfiniteTalk at commit 50aa0a9).
- **Inputs (`pieces/chronophobia/outputs/infinitetalk_poc/`):** `ref.jpg` (her photo), `vocals_44-61.wav` (vocal stem
  of the chorus, 44.2–60.8 s, 16 kHz mono), `input.json`.
- **Run:** `generate_infinitetalk.py --size infinitetalk-480 --sample_steps 40 --mode streaming --motion_frame 9`,
  then mux the full mix (`mix_44-61.wav`) over the result.

## What we expect

| # | Criterion | Pass if |
|---|---|---|
| 1 | Lip-sync | Mouth shapes follow the words; held notes keep the mouth open |
| 2 | Identity | Still her at the end of the 16.6 s clip |
| 3 | Motion | Head and shoulders move naturally; no frozen face or wobble |
| 4 | Cost | Fits on a Spark with no offload; render time noted per second of video |

Singing is harder than speech (sustained vowels, vibrato), so imperfect sync on held notes would be no surprise.
480p means it will need upscaling to match the 1024-wide LTX scenes.

## Getting it to run on a Spark

Seven fixes, all in `docker/infinitetalk/Dockerfile`:
1. einops, regex, yaml: imported but missing from requirements.txt.
2. decord: no aarch64 wheel; a stub that fails loudly (only used for video input).
3. xfuser: imported at load time even for one GPU.
4. `from inspect import ArgSpec`: removed in Python 3.11; the import is unused, so it's deleted.
5. misaki: needed by the bundled Kokoro TTS, imported at load time.
6. wav2vec2 refused `output_attentions` under SDPA (transformers 4.57): loaded with eager attention.
7. Wan asserts flash-attn is installed (not built for GB10): `sdpa_fallback.py` routes it to torch SDPA and keeps
   the text padding mask (checked against a reference: max difference 0.007 in bf16).

InfiniteTalk turns on **CPU offload by default on a single GPU**; every run passes `--offload_model False`.

## Result

| # | Setting | Speed | Outcome |
|---|---|---|---|
| 1 | Full quality: 40 steps, text guide 5, audio guide 4 | 114 s per step; about 5 h for the clip | Stopped after 3 steps |
| 2 | **Fast: lightx2v distillation LoRA, 4 steps, text guide 1, audio guide 2, shift 2** | 75 s per step; **39 min for 16.6 s** (about 142 s per second of video) | Finished |

Video: `pieces/chronophobia/outputs/infinitetalk_poc/chorus_480_lightx2v_mix-amber-is-born.mp4` (512×768, 25 fps, the full mix
laid over).

| # | Criterion | Verdict |
|---|---|---|
| 1 | Lip-sync | **Pass ("flying pass", user).** Mouth openness correlates with the vocal's loudness at r = 0.31 (0.08 for unrelated timings), peaking with the mouth leading the sound by 120–160 ms, which is how singers move. |
| 2 | Identity | **Pass.** Still clearly her at 16.6 s: face, hair, dress |
| 3 | Motion | Natural: head tilts, eyes closing on held notes, small shoulder moves. Background and framing stay locked, like a studio portrait |
| 4 | Cost | Fits on a Spark with no offload; about 140 s of compute per second of performance video |

Visual checks (same folder): `side_by_side.mp4` (photo left, clip right, timecode), `identity_check.jpg` (photo vs
0/4/8/12/16.4 s, with face close-ups), `motion_check.jpg` (one frame every 0.2 s, 8.0–12.6 s).

Open questions: does the full 40-step quality look better enough to pay 8× the time; the output is portrait 512×768
and needs upscaling and a 16:9 framing to cut with the LTX scenes; the singer is on a plain grey wall, so the
treatment's look has to be applied (prompt, background or a post grade).

**User review (2026-10-24):** success. Preferred over LongCat 1.5, which mis-articulates (r becomes w). InfiniteTalk is the lip-sync engine for sung inserts.
