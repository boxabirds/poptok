# Plan: poptok studio (web tools) and a heterogeneous server pool

**Status:** proposal, 2026-10-24. Direction from the user: macvideo's web tools are "a general sense of where I want to
take this" (a timeline of the video, keyframe-only views, browse by song line), plus a backend "heterogeneous server
pool" that, given machines with ssh access, discovers what each can do.

## 1. Studio: the web tools

### What macvideo has (surveyed 2026-10-24, `~/sambashare/expts/macvideo/editor/`)

React 19 + Vite frontend, FastAPI backend, SQLite. A song browser with per-stage status; a song editor with a
stage panel (run/redo with cost and time previews), a storyboard of one collapsible row per lyric line (word timings,
editable prompt fields, a picker for keyframe/clip takes), and an audio-driven preview (the audio is the playhead;
video re-syncs on drift; a keyframe thumbnail strip follows playback). Range-request asset serving, an SSE event
stream for jobs, a resident LTX renderer process. One Mac, no remote hosts, no transitions or lip-sync.

### What poptok's studio needs (in order of value)

| # | View | What it shows / does | poptok data |
|---|---|---|---|
| 1 | **Timeline** | The song as a horizontal strip: sections, lyric lines, bars/beats, scenes as blocks with their keyframe thumbnails, transitions as joints, sung passages marked; the playhead follows the audio; click anywhere to jump | `analysis/beats.json`, `*-true-segments.json`, `scenes.json`, `keys/`, `sing/inserts.json` |
| 2 | **Keyframe view** | All stills in a grid (by scene or by bar), with the critic's and still check's flags; play the song over them as a slideshow (a cheap preview before any video exists) | `keys/shot_NNN.png`, `check_keys_*.log`, critic JSON |
| 3 | **Browse by line** | Each lyric line with its cue (mode + shot note), the prompt, the still, the frames of its scene, and its section | `lyrics-cued.txt`, `shots.json`, `scenes.json` |
| 4 | **Scene editor** | Edit a scene's prompt, cue, motion, transition, lip-sync, lock; see **what the change will cost** before applying (from `docs/change-impact.md`: "this redraws 1 still and re-renders 41 s of video from 2:26, ~45 min on thing1") | `scenes.json` + `scenes.py apply` |
| 5 | **Review** | Watch the video; drop a note on a frame ("three hands") → it lands on the scene, with a frame grab; collected notes become one batched re-render | frame → scene (`scenes.py frame`) |
| 6 | **Workshop** | Read the directors' conversation and storyboard; edit the brief; re-run | `inputs/workshop.md`, `brief.txt` |
| 7 | **Jobs + pool** | What each box is doing, queue, ETA; contention with other projects (dbench) flagged for a decision | the pool (section 2) |

### How

- Reuse from macvideo: the audio-driven preview and thumbnail strip, Range asset serving, the storyboard row layout,
  the SSE job events, the take picker. Replace its stages and importer with poptok's.
- **`scenes.json` stays the source of truth** (the pipeline already reads and writes it); the server reads the piece's
  files directly and writes edits back through `scenes.py`, rather than SQLite owning the data. SQLite only for job
  history and review notes, if at all.
- The server runs on gruntus (where `./poptok` runs and the files are), behind the LAN; served to the user's browser.
- Build in slices: (1) read-only timeline + keyframe view + browse by line for any finished piece; (2) review notes;
  (3) scene editing with cost preview; (4) jobs/pool view; (5) workshop.

## 2. Heterogeneous server pool

Today the boxes are hard-wired (`bin/hosts.sh`, `site.env`, per-script docker/podman/native branches). The aim: given a
list of ssh hosts, the pool works out what each can do and schedules work onto the best box, sharing politely with
other projects.

### Three kinds of knowledge (the user's framing)

| Kind | How it's obtained | Examples |
|---|---|---|
| **a) Known from the spec** | Discovery over ssh: OS/arch, GPU vendor/model, VRAM or unified memory, CPU, RAM, disk, container runtime (docker/podman/none), CUDA/ROCm/Metal versions, which images and model weights are present | "NVIDIA, 24 GB → LLM 27B Q4 yes; MOSS 33 GB no (no offload); LTX 22B bf16 no; InfiniteTalk int8 yes" — each step declares its requirements (memory, backend, image), so fit is computed, not guessed |
| **b) Needs quick calibration** | A small qualification job per (box, step): build/pull the image, run a short benchmark (one still, one 65-frame clip, 3 s of lip-sync), record time and pass/fail | Strix Halo MOSS 386 s → 78 s once two settings were found; quintus LTX 1.47 s/frame; gruntus InfiniteTalk int8 62 s per second of video |
| **c) Only known from real runs** | Measured on production jobs and fed back: throughput on long runs, failure rates, thermal throttling, quality differences, settings that only matter at length | LTX on Strix slowing past ~100 frames; per-frame cost rising on short clips; the int8 lip-sync's slow push-in |

`docs/hardware-performance.md` is already this table filled by hand; the pool makes it data (a JSON/SQLite store the
scheduler reads), with the doc generated from it.

### Components

1. **Inventory**: `pool.yaml` (hosts, ssh names, which are shared with other projects, any pinned roles).
2. **Discovery** (`pool discover`): ssh probes → a capability record per host (hardware, runtimes, images, weights,
   free disk).
3. **Step requirements**: each pipeline step declares `{memory_gb, backend: [cuda, rocm, metal], image, weights,
   whole_gpu: bool}` and its variants (bf16 vs int8, docker vs native).
4. **Qualification** (`pool qualify STEP HOST`): runs the step's benchmark, records time, pass/fail and the settings
   that worked.
5. **Ledger**: every real job records host, step, size (frames, seconds, stills), wall time, outcome → kind (c).
6. **Scheduler**: for each job, the eligible hosts (fit + qualified), ranked by expected finish time (queue + rate ×
   size), respecting one GPU job per box and the locks; replaces `pick_host` and today's hand-written SING_HOSTS /
   KEY_POOL / VIDEO_HOST.
7. **Contention**: detects other projects' work (dbench and any non-poptok GPU process) and asks the user instead of
   competing (the "pause dbench?" decision becomes a button in the jobs view).

### Order

1. Step requirements + discovery (turns today's hand knowledge into records; cheap).
2. Ledger from the existing logs (render logs already hold per-step times).
3. Scheduler for the steps that already run on several boxes (stills, lip-sync), then video.
4. Qualification jobs for new boxes or new steps.
5. The jobs/pool view in the studio.

## Open questions for the user

- Where should the studio be reachable: LAN only (gruntus), or also from outside (tailscale)?
- Build the studio inside poptok (`studio/`) or as a shared tool for both macvideo and poptok?
- Which first: the studio's read-only views (timeline, keyframes, by line) or the pool's discovery + scheduler?
