# Runbook (Julian's setup)

How to reproduce poptok's results on this particular farm of boxes. The general pipeline is in the README; this page
is the machine-specific part: which box does what, where things live, the exact commands, and what to check.
Measured speeds per step and box: [hardware-performance.md](hardware-performance.md).
Keep it current: whenever a command, host role, path or setting changes, update this page in the same session (see
CLAUDE.md). What was done and when goes in [log.md](log.md).

## Machines

| Box | Hardware | Role | Runs as |
|---|---|---|---|
| **gruntus** | RTX 4090 24 GB, x86 | Control box: `./poptok` runs here. Analysis (`m2v-pre`), the LLM (Qwen3.8-27B in llama-server, port 8091), finish step, downloads (`hfdl`). Not in the keyframe pool: MOSS (33 GB) doesn't fit and we never offload. | docker; finish step on host Python 3.9 (known gap) |
| **thing1**, **thing2** | DGX Spark GB10, 128 GB, ARM64 | Keyframes (MOSS `morph-spark`; Amber stills with Qwen-Image-Edit), video (`ltx2`), lip-sync (`infinitetalk`). Linked by QSFP: `thing2-qsfp` from thing1, `thing1-qsfp` from thing2 (use for big copies between them). | docker, repo under `~/poptok`, models under `~/models` |
| **tritus** | Strix Halo (Ryzen AI Max+ 395, gfx1151), 128 GB | MOSS keyframes, ~78 s/still (Spark ~45 s). One job at a time (two measured no faster). | podman, `localhost/chronoedit-rocm` |
| **quintus** | MacBook M5 Max | LTX video only, native (Metal isn't reachable from a container), ~1.47 s/frame. | uv venv `.venv-ltx` in `~/expts/poptok` (`bin/setup_mac_renderer.sh`); GPU lock is the directory `~/.poptok-gpu.lock.d` |

Site settings: `site.env` in the repo (git-ignored): `KEY_POOL="thing1 thing2 tritus"`, `ROCM_HOSTS=tritus`.
Hosts: `bin/hosts.sh` (`MAC_HOSTS=quintus`, the busy-image regex, the lock path).

## Rules (the user's, non-negotiable)

- **Containers for everything** (docker; podman on tritus). Exception: quintus runs natively. Nothing installed into
  bare-metal Python.
- **No CPU offload anywhere.** If a model doesn't fit a box's GPU, run it on another box. InfiniteTalk defaults to
  offload: always pass `--offload_model False`.
- **One GPU job per box** (the flock at `${TMPDIR:-/tmp}/poptok-gpu.lock`, plus a wait while any pipeline container runs).
- **Local only**: no cloud APIs.
- **Never sudo.** HF token only in environment variables.
- **Outputs the user looks at go on the share** (`~/sambashare/expts/poptok/pieces/...`, `pieces/_bench/eval/`),
  never only `/tmp`.
- Commit and push to main at the end of each working session (user, 2026-10-24: the repo had never been pushed); otherwise only when asked.

## Making a video

A piece is a folder under `pieces/` (git-ignored): `piece.env`, `inputs/`, `analysis/`, `outputs/<RUN>/`.

```bash
cd ~/sambashare/expts/poptok
./poptok pieces/slippery-road                 # full run; re-running resumes (each step skips if its output exists)
./poptok pieces/slippery-road --run v2        # a new version alongside the old one
./poptok pieces/slippery-road --plan-only     # stop after prompts, to read/edit before hours of rendering
./poptok pieces/chronophobia --test 30        # a 30 s test copy (pieces/chronophobia-test30/)
```

The R.Amber `piece.env` settings that matter:

```bash
CUES=inputs/lyrics.txt                 # lyrics annotated with actor cues (none / pov / part / scene / sing) and [Singer: ...]
PERFORMER_PHOTO=../artists/r.amber.jpeg
VIDEO_HOST=thing2                      # or thing1, or quintus
PULSE=0                                # no bass-hit pulse: the user rejected it (all R.Amber pieces)
STYLE_WORD="oil impasto"               # optional one-word style + STYLE_BRIEF (POC 24)
FINISH_ARGS="--deflare 8 --grade color"
```

Steps (`bin/make_video.sh`), where each runs, and roughly how long for a 3.5–4 min song:

| # | Step | Where | Time |
|---|---|---|---|
| 1 | Analysis: stems, Whisper, beat grid | gruntus | ~5 min |
| 2–3 | Lyrics aligned, prompts per line (LLM), actor cues | gruntus | ~5 min |
| 4 | Keyframes: Amber stills (Qwen-Image-Edit + Lightning, ~24 s/still) and MOSS stills in parallel; still check (`check_keys.py`), up to 3 re-roll rounds; merge near-duplicate scenes | thing1/thing2/tritus | ~45–60 min |
| 4b | `scenes.json` + LLM-written A→B transitions | gruntus | ~3 min |
| 5 | LTX chain (LTX-2.5 + LTX-2.3 Transition LoRA) | `VIDEO_HOST` | ~1.6 h on a Spark (6000 frames) |
| 6 | Retime to downbeats, grade, mux → `NAME.mp4` | gruntus | ~6 min |
| 7 | Lip-sync: one InfiniteTalk clip per run of sung scenes, spliced in → `NAME.mp4` (unsynced kept as `NAME-nosync.mp4`) | Sparks (`SING_HOSTS`, default `thing1 thing2`) | ~2.4 min of compute per second of chorus |

## The directors' workshop (treatment and storyboard from the lyrics)

Set `WORKSHOP=1` in `piece.env` (with `TREATMENT=inputs/visual-treatment.txt`, `CUES=inputs/lyrics-cued.txt`), optionally
write `inputs/brief.txt` (artist, mood, references, review notes); `./poptok` runs `bin/workshop.sh` first when
`inputs/workshop.md` doesn't exist. Read `inputs/workshop.md` before rendering (`--plan-only`). Redo: `bin/workshop.sh
pieces/NAME --force`. Lyrics need `[Section]` headers in brackets (bare headers get aligned as sung lines).
The workshop needs the lyric timings: `./poptok` runs the analysis first (`bin/make_video.sh ... --analysis-only`); by
hand, run that before `bin/workshop.sh`. The directors get every line's time and a shot slot (G1, G2, ...) every ~6-8 s
of each instrumental passage of 6 s or more; `lyrics-cued.txt` then has `(instrumental M:SS.S) {mode: note}` lines.
See what they'll get without the LLM: `python3 pipeline/workshop.py pieces/NAME --dry-run`. After a workshop, check
`visual-treatment.txt` keeps the brief's look and that "sing" is only on a few passages (the LLM ignores both rules
often; fix by hand, keep the originals as `*.workshop.txt`).

## Fixing one detail in a still (single pass, no loop)

```bash
# FIXES.json: [{"key": "shot_074.png", "fix": "remove the rifle; her hands are empty"}, {"key": ..., "photo": true}]
python3 pipeline/fix_keys.py OUT FIXES.json --photo pieces/artists/r.amber.jpeg [--in-place]          # on a Spark
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-qwenedit/bin/python pipeline/fix_keys.py OUT FIXES.json --device mps --models ~/models   # quintus
```
One edit per marked still, accepted or left; never re-edited (the user: edit loops don't converge). Works for removing
or swapping one thing (an object, a stray figure, clothing, a reflection); not for reshaping a body or face (test
2026-10-24: 4 of 6 clean). Results in `OUT/keys_fixed/` with before|after sheets; `--in-place` keeps `*.orig.png`.

## Bespoke effects (per piece)

`FX=inputs/fx.json` in `piece.env`: a list of effects from `pipeline/fx/` applied at the finish step, frame by frame
(so changing them costs ~6 min, no re-render). Event selection (`fx/events.py`): drum hits from
`analysis/drum_hits.json` (`pipeline/drum_hits.py`, made in the analysis step) by pattern (`triplet` = tat-tat-tat,
`runs`, `all`), strength, and optionally only around a lyric (`"after_lyric": "noble hunt"`). Effects so far:
`bursts` (light bloom + sparks + a viewpoint knock). New effect = a new module in `pipeline/fx/` with `make(cfg, ctx)`.

## Editing scenes after a render

```bash
python3 pipeline/scenes.py frame OUT 1379 2037      # which scene a video frame is in
# edit OUT/scenes.json: prompt, cast, motion, transition_in, lipsync, reroll, locked
python3 pipeline/scenes.py apply OUT                # deletes only the affected stills and frames
./poptok pieces/NAME --run SAME_RUN                 # re-renders just those
```

Copy a run first (`cp -a outputs/v1 outputs/v2; rm outputs/v2/*.mp4`) to keep the old version.

## Lip-sync by hand

```bash
python3 pipeline/sing_inserts.py plan OUT pieces/NAME/analysis STEM     # STEM = the song file's name without extension
bin/sing_inserts.sh OUT "thing1:0,2 thing2:1,3"                          # skips inserts already rendered
python3 pipeline/sing_inserts.py splice OUT OUT/NAME-nosync.mp4 OUT/NAME.mp4 --fx pieces/NAME/inputs/fx.json --fx-analysis pieces/NAME/analysis
# plan: an insert that starts or ends mid-line is moved out to that line's start/end (+0.25 s); splice --fx puts the
# piece's effects back on the inserts (they're cut in after the finish step)
```

## Removing the pulse from an existing render (no re-render)

```bash
python3 pipeline/finish.py OUT --audio pieces/NAME/inputs/SONG.wav --analysis pieces/NAME/analysis \
  --plan shots_merged.json --frames-dir OUT/ltx_chain_beat --src-fps 24 --no-flash --no-pulse --name NAME-RUN \
  --grade treatment --step 1 --deflare 8 --grade color
```

## Self-checks

```bash
python3 pipeline/eval_video.py VIDEO.mp4            # flares and jumps, frame by frame
python3 pipeline/check_keys.py OUT --photo PHOTO    # blank stills, strangers (hair rule), faces in part shots
python3 pipeline/eval_lipsync.py CLIP.mp4 CLIP.wav  # mouth vs vocal correlation; unreliable on wide/medium shots
python3 pipeline/eval_identity.py ...               # face similarity to her photo (ignores hair: don't trust alone)
```

Plausibility critic (POC 28), on gruntus, under the GPU lock, segmentation first then Qwen (together they exceed 24 GB:
segment ~9 GB, Qwen + projector ~18 GB):

```bash
docker build -t segment docker/segment      # once; weights in ~/models/segment (llmdet_large, sam2.1-hiera-large)
docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -v ~/models/segment:/models:ro -v $PWD:/work \
    segment python pipeline/segment_check.py IMG... --out SEGDIR          # geometry flags + overlay PNG per image
# then llama-server with --mmproj (see critic.py's docstring), and:
python3 pipeline/critic.py IMG::INTENT ... --crops --segment SEGDIR --json OUT.json
```

Look at consecutive frames, not sparse contact sheets. Put sheets in `pieces/_bench/eval/`.

## Operating notes (learned the hard way)

- **Never edit a running bash script in place** (bash reads it as it goes): write `X.new`, `bash -n`, then `mv` over.
- **Killing a remote job:** killing the local ssh doesn't stop it. Stop the container / process on the remote box
  too, and remove quintus's lock directory if it's left behind.
- **Don't `pkill -f`/`pgrep -f`** with a pattern that matches your own shell; kill PIDs individually.
- **Long remote jobs on quintus:** run inside `screen -dmS NAME ...` so an ssh hangup doesn't kill them.
- **quintus has its own copy of the code** (`~/expts/poptok`); `make_video.sh` syncs `pipeline/` there before each
  video. Anything run on quintus by hand needs `rsync -a pipeline/ quintus:expts/poptok/pipeline/` first.
- **Still models on quintus (native MPS, 2026-10-24):** Qwen-Image-Edit + Lightning runs (33 s/still) in its own venv
  `~/expts/poptok/.venv-qwenedit` (weights in quintus `~/models/Qwen-Image-Edit-2511{,-Lightning}`); benchmark:
  `.venv-qwenedit/bin/python pipeline/mac_bench.py qwen --out DIR` under the lock (`pieces/_bench/mac-ports/macrun.sh`).
  MOSS (`.venv-moss`) stops on a float64 line in its own pipeline file; InfiniteTalk (`.venv-infinitetalk`, deps only)
  stops on xfuser/decord/xformers/flash-attn. Not wired into `amber_keys.py` yet.
- **zsh globs remote paths locally:** quote `"host:path/*.png"` in rsync.
- **Restarting the LLM by hand:** take the GPU lock first
  (`exec 9>/tmp/poptok-gpu.lock; flock 9; ~/.local/bin/llama-server -m <gguf> -ngl 99 -c 16384 --reasoning-budget 0 --port 8091`).
- thing1 runs an `ollama/ollama` container that isn't poptok's; it's idle (no model loaded) and harmless.

## Model notes

- **Video:** LTX-2.5 22B distilled (`~/models/ltx-2.5`); the only LTX-2.3 file is the Transition LoRA
  (`~/models/ltx-2.3/ltx2.3-transition.safetensors`, valiantcat). Its trigger is ", zhuanchang". Untested: whether
  the 2.3-trained LoRA is fully effective on 2.5.
- **Amber stills:** Qwen-Image-Edit-2511 + Lightning LoRA, 8 steps, cfg 1 (`pipeline/qwen_edit.py`).
- **Lip-sync:** InfiniteTalk fast mode (lightx2v LoRA, 4 steps, text guide 1, audio guide 2, shift 2), weights in
  `~/models/infinitetalk` on the Sparks. LongCat-Video-Avatar was rejected (mis-articulates "r").
- `HOLD_BEFORE_VIDEO=1 ./poptok pieces/NAME`: stop before the video for a still review (`pipeline/fix_keys.py`), then `./poptok pieces/NAME` to continue.
- **A piece from a prompt:** `SONG_PROMPT="..."` (plus `WORKSHOP=1`, `TREATMENT=inputs/visual-treatment.txt`, `CUES=inputs/lyrics-cued.txt`) in piece.env, then `./poptok pieces/NAME`: songwriters' workshop (`inputs/songwriters.md`, `lyrics.txt`, `music-style.txt`) → ACE-Step song → analysis → directors' workshop → video. Redo the lyrics: `WORKSHOP_PY=pipeline/songwriters.py bin/workshop.sh pieces/NAME --force`. The LLM's conversation tends to loop after ~10k characters; it's cut there and the songwriters finish the song in the next pass.
