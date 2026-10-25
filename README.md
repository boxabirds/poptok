# poptok

A music video from a song and its lyrics, or from nothing but a prompt. Simulated workshops do the creative planning:
songwriters write the song, directors plan the video. Local models do the rest: ACE-Step sings, MOSS and
Qwen-Image-Edit draw the stills, LTX turns them into one continuous take, and InfiniteTalk lip-syncs the passages
where the performer sings on camera. Everything runs locally, spread over whatever GPU boxes you have.

```
./poptok pieces/<name> --plan-only    # plan only (minutes): read outputs/v1/treatment.md and the prompts
HOLD_BEFORE_VIDEO=1 ./poptok pieces/<name>   # stills, then stop for a review before the hours of video
./poptok pieces/<name>                # everything; re-running resumes where it stopped
```

## A piece

Each song gets its own folder under `pieces/`. Git ignores `pieces/`, so inputs, outputs and per-song settings stay
out of the repo. To start one, copy `pieces.example/`.

```
pieces/<name>/
  piece.env            settings for this piece: inputs, run name, hosts, tuning
  inputs/              lyrics.txt, song.wav or music-style.txt, visual-treatment.txt or brief.txt, ...
                       (the workshops write their transcripts and results here too)
  analysis/            written by poptok: stems, transcript, beats, lyrics on the song's clock
  outputs/<run>/       written by poptok: keys/ (stills), ltx_chain/, sing/ (lip-sync), <name>-<run>.mp4
  outputs/song.wav     written by poptok when it generated the song
```

There are three ways in:

| You have | piece.env | What poptok does first |
|---|---|---|
| A song and its lyrics | `SONG`, `LYRICS` | Nothing extra |
| Lyrics and a music style | `LYRICS`, `MUSIC_STYLE`, no `SONG` | ACE-Step 1.5 makes several takes; the one whose sung words Whisper understands best is kept |
| Only a prompt | `SONG_PROMPT="A song about unrequited love"` | The songwriters' workshop writes the lyrics and the music style, then the song is generated as above |

Add `WORKSHOP=1` (with `TREATMENT=inputs/visual-treatment.txt`, `CUES=inputs/lyrics-cued.txt`) and the directors'
workshop writes the visual plan, so a prompt alone is enough for a whole video. Without it, you write
`visual-treatment.txt` yourself.

### The songwriters' workshop (`pipeline/songwriters.py`)

Three songwriters, fictional composites of the Nashville-craft, pop-melody and singer-songwriter traditions, discuss
the prompt. They agree an angle, title, hook and the music, draft the chorus and the verses, and cut each other's
cliches. The result is checked against the form and retried if it doesn't fit. The default form (`SONG_FORM`) is an
intro, 4 verses interleaved with the chorus, and an outro of two choruses back to back. Written to `inputs/`:
`songwriters.md` (the transcript and the final lyrics), `lyrics.txt` and `music-style.txt`.

### The directors' workshop (`pipeline/workshop.py`)

Three music-video directors (handmade, observational and graphic traditions) plan the video from the lyrics, their
timings and an optional `inputs/brief.txt`. They're told what the machine can and can't draw. They agree:
- the **universe style**: the medium every frame is made of (oil impasto, cyanotype, daguerreotype, photographic...)
  and how light behaves in it; it leads every image prompt;
- how **concrete or abstract** the video is, overall and shot by shot;
- a shot for every line, and for every 6-8 s of instrumental passages, so no still is held too long;
- where the performer appears and where she sings (if there's a performer photo).

A plausibility pass then fixes shots a camera could never see. Written to `inputs/`: `workshop.md`,
`visual-treatment.txt`, `universe.txt` and `lyrics-cued.txt` (each line with its mode and shot note).

Lyrics can be in a songwriter's format: bare section headings with an action hint in parentheses
(`Chorus 2 (a small group has formed)`), and whole-line stage directions (`(silence)`). poptok turns them into
bracketed tags (`pipeline/normalise_lyrics.py`), and the directors follow the hints.

### A performer

`PERFORMER_PHOTO` (one photo) and `PERFORMER` (a short description) put the singer in the video. Her stills are drawn
by Qwen-Image-Edit from the photo, in modes the directors choose per line: `none`, `pov`, `part` (a hand, her back,
a reflection), `scene` (acting) and `sing`. The `sing` passages are lip-synced with InfiniteTalk from the vocal stem
and cut in. `WARDROBE` dresses her for the piece.

## Steps

| # | Step | Script | Where |
|---|---|---|---|
| 0a | Songwriters' workshop (with `SONG_PROMPT`) | `songwriters.py` | local LLM |
| 0b | Song (with no `SONG`) | `bin/make_song.sh`, `ace_gen.py`, `ace_score.py` | `acestep` image |
| 1 | Stems, tempo, Whisper word timings, beat grid; the lyrics placed on the song's clock (forced alignment) | `analyse.py`, `beat_grid.py`, `align_lyrics.py`, `force_align.py` | analysis image |
| 2 | Directors' workshop (with `WORKSHOP=1`) | `workshop.py` | local LLM |
| 3 | One prompt per line and instrumental slot, inside the universe style | `prompts_per_line.py` | local LLM |
| 4 | Stills: MOSS-Image for scenes, Qwen-Image-Edit for the performer; an automatic still check re-rolls blanks and faulty shots; near-duplicate neighbours merged; transitions planned | `keyframes.py`, `amber_keys.py`, `check_keys.py`, `merge_scenes.py`, `scenes.py` | any GPU box, LLM |
| 5 | Video: LTX-2.5 with the LTX-2.3 Transition LoRA, one continuous take where each scene turns into the next | `ltx_chain.py` | `VIDEO_HOST` (NVIDIA, AMD or Apple Silicon) |
| 6 | Scene changes moved onto downbeats; grade; effects (`FX`: e.g. light bursts on drum hits); mux | `retime_to_beats.py`, `finish.py`, `fx/` | host |
| 7 | Lip-sync of the `sing` passages, cut in from just before each sung line, effects reapplied | `sing_inserts.py`, `bin/sing_inserts.sh` | the free boxes |

Between 4 and 5, `HOLD_BEFORE_VIDEO=1` stops for a review. To fix one detail of a still (a stray object, a second
copy of the performer, a reflection), use one instruction edit (`pipeline/fix_keys.py`, Qwen-Image-Edit) and keep it
or leave it. Never loop: edit-and-regenerate loops don't converge. `pipeline/critic.py` (a vision LLM) and
`pipeline/eval_video.py` (flares and jumps frame by frame) help with the review. `docs/change-impact.md` says what a
change makes poptok redo.

## Machines

One GPU job per box at a time (a lock per box); each step waits for the box to be free. `site.env` (copy
`site.env.example`) names the boxes: `VIDEO_HOST`, `KEY_POOL`, `ROCM_HOSTS`, the lip-sync hosts. Tested on:

| Box | Runs | Notes |
|---|---|---|
| RTX 4090 (24 GB) | LLM, analysis, MOSS stills, int8 lip-sync, finish | the control box; nothing is offloaded to CPU, so the bf16 video model doesn't fit |
| DGX Spark | everything, incl. Qwen-Image-Edit stills and bf16 lip-sync | docker |
| AMD Strix Halo | MOSS stills, LTX video, lip-sync | podman, ROCm images in `docker/*-rocm` |
| M5 Max | LTX video, Qwen-Image-Edit fixes | native in uv venvs (`bin/setup_mac_renderer.sh`); MOSS and InfiniteTalk not ported |

Measured times for every step on every box are in `docs/hardware-performance.md`.

## Requirements

- One NVIDIA GPU box for analysis, the LLM and keyframes (tested on an RTX 4090, 24 GB).
- For the video step, the same box or another reachable by ssh with the same `~/models` layout (see Machines).
- Docker images, built from `docker/` (see `docker/README.md`): `m2v-pre`, `zoommorph`, `ltx2`, `infinitetalk`
  (lip-sync), `segment` (critic), and `acestep` (needed only to generate songs). The Sparks' stills image `morph-spark`
  (MOSS and Qwen-Image-Edit) is not in `docker/` yet.
- A local LLM: llama.cpp `llama-server` with Qwen 3.8 27B (`LLM_SERVER`, `LLM_GGUF`). It is started for each step
  and stopped after, so the GPU only ever runs one job.
- Models in `~/models`:
  - `MOSS-Image-SFT`
  - `ltx-2.5/`, plus `ltx-2.3/ltx2.3-transition.safetensors` (valiantcat/LTX-2.3-Transition-LORA)
  - `Qwen-Image-Edit-2511` and its Lightning LoRA (performer stills, fixes)
  - `infinitetalk/` (lip-sync)
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
- `pipeline/treatment.py` (`DIRECTOR=1`): the older whole-video treatment step, without the workshops.
- `docs/runbook-julian.md`: how this installation is run day to day; `docs/log.md`: the working log; `docs/pocs/`:
  the experiments behind each choice.

Origins: prompt wording and analysis come from [music2vid](https://github.com/boxabirds/music2vid). The renderer
experiments are in morphexpts, which is not public.
