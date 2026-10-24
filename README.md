# poptok

Lyrics and a visual treatment in, a music video out. The pictures change on the beat, and the video pulses now and
then on the real bass hits. If you don't have a song yet, poptok can make one first from a music style and the lyrics.

```
./poptok pieces/<name> --plan-only    # treatment + scene prompts (minutes): read outputs/v1/treatment.md
./poptok pieces/<name>                # keyframes, video, finish (about an hour and a half for 2-3 minutes of song)
```

## A piece

Each song gets its own folder under `pieces/`. Git ignores `pieces/`, so inputs, outputs and per-song settings stay
out of the repo. To start one, copy `pieces.example/`.

```
pieces/<name>/
  piece.env            settings for this piece: inputs, run name, video host, tuning
  inputs/              lyrics.txt (master lyrics with [Section] tags), visual-treatment.txt,
                       song.wav, or music-style.txt to generate one
  analysis/            written by poptok: stems, transcript, beats, lyrics on the song's clock, prompts
  outputs/<run>/       written by poptok: treatment.md, keys/, ltx_chain/, <name>-<run>.mp4
  outputs/song.wav     written by poptok when it generated the song
```

There are two ways in:
- **Song + lyrics + visual treatment:** set `SONG`.
- **Lyrics + music style + visual treatment:** leave `SONG` empty and set `MUSIC_STYLE`. ACE-Step 1.5 generates
  several songs, and poptok keeps the one whose sung lyrics Whisper understands best.

## Steps (`bin/make_video.sh`)

| # | step | script | where |
|---|---|---|---|
| 1 | Stems, tempo and a Whisper transcript with word timings; beat grid | `pipeline/analyse.py`, `beat_grid.py` | analysis image |
| 2 | Master lyrics placed on the song's clock (Whisper gives only the timing, never the words) | `align_lyrics.py` | host |
| 3 | Treatment for the whole video, then one scene per lyric line written from it | `treatment.py` | local LLM |
| 4 | One keyframe per bar; near-duplicate neighbouring scenes merged | `keyframes.py`, `merge_scenes.py` | keyframe image, LLM |
| 5 | Video: LTX-2.5 + Transition LoRA, one continuous chain where each scene transforms into the next | `ltx_chain.py` | LTX image, any box |
| 6 | Each scene change moved onto its downbeat; occasional pulse on real low-frequency hits; grade; mux | `retime_to_beats.py`, `finish.py` | host |

The treatment (`outputs/<run>/treatment.json`, readable as `treatment.md`) sets:
- the story arc;
- a fixed cast, places and motifs, each worded the same way in every prompt;
- the look, which leads every image prompt;
- how things move, and how one scene turns into the next;
- the text policy: at most one word per scene, copied from the lyric line, with the camera locked off for that scene;
- the colour grade, from start to end;
- the palette for each section;
- whether frames are held in pairs for a stop-motion feel.

You can edit `treatment.json` and re-run. To rewrite the scenes from your edited treatment, delete
`analysis/full-metadata-true.json` first.

## Requirements

- One NVIDIA GPU box for analysis, the LLM and keyframes (tested on an RTX 4090, 24 GB).
- For the video step, the same box or another reachable by ssh (tested on a DGX Spark) that has the same
  `~/models` layout.
- Docker images, built from `docker/` (see `docker/README.md`): `m2v-pre`, `zoommorph`, `ltx2`, and `acestep`
  (needed only to generate songs).
- A local LLM: llama.cpp `llama-server` with Qwen 3.8 27B (`LLM_SERVER`, `LLM_GGUF`). It is started for each step
  and stopped after, so the GPU only ever runs one job.
- Models in `~/models`:
  - `MOSS-Image-SFT`
  - `ltx-2.5/`, plus `ltx-2.3/ltx2.3-transition.safetensors`
  - `acestep/` (needed only to generate songs)

Each step waits while another pipeline container is using the GPU on that box: one GPU job per box.

## Software versions

As used for the offline-local v2 render (2026-10-23). Read from the images and hosts that ran each step.

| Step | Where | Software | Models |
|---|---|---|---|
| Song | `acestep` image (upstream ACE-Step Dockerfile) | ACE-Step 1.5.0; Python 3.11.17, torch 2.10.0+cu128, transformers 4.57.6, diffusers 0.37.1 | ACE-Step XL-sft |
| Analysis | `m2v-pre` image, RTX 4090 (driver 580.159.03) | Python 3.10.12, torch 2.4.1+cu121, Demucs 4.0.1a1, openai-whisper 20240930, whisper-timestamped 1.15.9, librosa 0.11.0, numpy 1.26.4 | Whisper medium |
| Prompts, scene merge | host, RTX 4090 | llama.cpp llama-server 0.6.0-dev (commit de7fa0a) | Qwen3.8-27B GGUF, UD-Q4_K_XL |
| Stills | `morph-spark` image, DGX Spark GB10 (driver 580.178.04, kernel 7.0.0-1019-nvidia) | Python 3.12.3, torch 2.14.1+cu130 (cuDNN 9.24), diffusers 0.41.0, transformers 5.20.2, accelerate 1.15.0 | MOSS-Image-SFT (no revision recorded) |
| Video | `ltx2` image, DGX Spark GB10 | Python 3.12.15, torch 2.13.0+cu132, ltx-core 1.5.0 and ltx-pipelines 1.5.0 (commit e98f6d9), transformers 5.14.1, accelerate 1.15.0 | LTX-2.5 22B distilled transformer (bf16), Gemma 4 12B text encoder with LTX-2.5 projection, LTX-2.5 video VAE (bf16), LTX-2.3 Transition LoRA |
| Finish | host (not containerised yet) | Python 3.9.16, OpenCV 4.13.0, numpy 2.0.2, FFmpeg 4.4.2 | — |
| Plumbing | all boxes | Docker 29.4.2 (4090 box), 29.6.2 (Sparks); Bash; `flock` | — |

Known inconsistencies, to tidy up:
- The finish step runs on the host, outside Docker.
- The analysis image is on torch 2.4 / CUDA 12.1, against torch 2.13–2.14 / CUDA 13 elsewhere.
- The stills and video images use different torch builds (2.14 + CUDA 13.0 and 2.13 + CUDA 13.2).

## Other tools

- `pipeline/lyric_loop.py`: re-spells words the singer garbles (generate, separate stems, transcribe, compare with the
  lyrics, try a phonetic variant) until Whisper hears them.
- `pipeline/prompts_per_line.py`: the older method, one independent prompt per lyric line plus a fixed style
  (`STYLE=` in piece.env).

Origins: prompt wording and analysis come from [music2vid](https://github.com/boxabirds/music2vid). The renderer
experiments are in morphexpts, which is not public.
