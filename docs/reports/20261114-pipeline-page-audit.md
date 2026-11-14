# Audit: the pipeline page against the code

**Written:** 14 November 2026. **Audited:** `docs/design/20261107-pipeline.html` (artifact v25) against the main
pipeline: `poptok`, `bin/make_video.sh`, `bin/make_song.sh`, `pipeline/*`, `pipeline/steps.ini`, run logs in
`pieces/`. Three read-only passes (tasks 0-3, 4-5, 6-8 plus the end tables); the main claims re-checked by hand
(marked ✔). Paths are relative to `poptok/`.

## The two biggest problems

1. **The order is wrong.** The directors meet twice, around the universe step: `--stage story` (conversation,
   treatment `story.md`, their own universe proposal) → universe step → `--stage storyboard` (`poptok:113-129`,
   `steps.ini`). ✔ The page shows universe (4) before directors (5) and says the universe is chosen "before the
   directors meet, so they plan for a known medium". Worse: **the storyboard never reads the chosen universe**
   (`workshop.py` never opens `universe.txt`/`visual-look.txt`); the universe only reaches the per-line image prompts
   (`STYLE_WORD`/`STYLE_BRIEF`, `poptok:133-136`). ✔
2. **Some steps described are lean's (`lean/`), not the main pipeline's**: the storyboard sheet, "her expression
   named", SFace matching against her photo, Qwen "native" drawing of her. The main pipeline draws her with
   `amber_keys.py`, checks stills with `check_keys.py` (YuNet + a hair-lightness rule, no SFace, ✔), makes no sheet,
   and never runs the vision-model critic (`critic.py` is not called anywhere, ✔).

## Task by task

### 0 Inputs and the overview
- "Song prompt: Required": there are three ways in: `SONG` (your song), `LYRICS`+`MUSIC_STYLE`, or `SONG_PROMPT`
  (+`SONG_FORM`) (`poptok:9-15`). The directors run only with `WORKSHOP=1`; otherwise the per-line Offline method.
  The avatar still is optional (`PERFORMER_PHOTO`); Unrequited ran without one.
- "Leave [the brief] out and the directors write one (stage 4)": stated as fact, but not built; a missing brief
  becomes "(none)" (`workshop.py:454-455`).
- Banner "Every step below has been run on its own": two steps are marked new (never run). The banner credits the
  generated-song run to Offline; it was Unrequited (directors' workshop, 25 Oct, `docs/log.md:245`).

### 1 Create the song
- "not measured": Unrequited's songwriters took ~4½ min including model load, ~43 tokens/s
  (`pieces/unrequited/songwriters_llm.log`).

### 2 Record the song
- Missing: the sense-check threshold (`MIN_MATCH` 0.3) and up to 2 extra rounds of seeds, then stop
  (`make_song.sh`). ✔
- "43–49 s per song" is per take; the step sings 3 takes by default (`SONG_SEEDS=1,2,3`, ✔) and took ~4 min.
- Missing: a style over 500 characters is shortened by the LLM first (so stage 2 can need the LLM); BPM is read from
  the style text; lyrics are normalised to [Section] form (`normalise_lyrics.py`) between 1 and 2.
- "2B with the 4B planner": that measurement was the 2B *turbo*; the pipeline can't select it (`--variants xl-sft`
  hard-coded).
- "Length set by the song": the planner picks a length per take (195, 175, 245 s for the same lyrics).
- The sense check runs in the m2v-pre image, not acestep.

### 3 Disassemble the song
- Two tempo estimates that can disagree: `metadata.json` from the drum stem (used by prompts) and `beats.json` from
  the full mix (used by the directors). Noble Hunt: 95.7 vs 126.0 BPM. ✔
- Downbeats assume 4 beats a bar (`beat_grid.py:66-67`).
- `drum_hits.py` and `align_lyrics.py` run on the host, not in the m2v-pre image; strip 4's chips omit numpy/difflib.
- "A forced aligner was tried and dropped": it is opt-in (`FORCE_ALIGN=1`), not removed.
- `docs/hardware-performance.md:26` still says "~5 min/song" for stems + transcription. ✔

### 4 Create the singer / actor (the universe step)
- Order: see above. ✔
- Inputs: not the song prompt; it reads the lyrics, brief, the treatment (`story.md`), the directors' proposal,
  `PERFORMER`, `WARDROBE`, `PERFORMER_PHOTO`.
- "14 universes": 15 in every logged run (8 open + 6 metaphor + the directors'). ✔
- Missing: the medium check (a look fixed by the brief binds every candidate).
- "Review at two pauses": no pause is built in; pick 1 is chosen automatically and the pipeline carries on. Stopping
  at the sheet is opt-in (`./poptok --until universe`); the only built-in pause is `HOLD_BEFORE_VIDEO`. ✔
- Test shots: MOSS draws a woman *described* by `PERFORMER`; the face pass (YuNet crop, Qwen-Image-Edit, feathered
  paste, 3 tries scored by SFace) puts her face in. Timing: 40-144 s per still for the face pass, not ~24 s.
- LLM "not measured": the universe step's LLM part takes 74-84 s.
- The two "new" steps (write a brief; describe the singer with the vision model) really are unbuilt; `PERFORMER` is
  typed by hand.

### 5 Design the video scenes
- "removes shots no camera could see": the plausibility pass replaces the flagged shot's note. ✔
- "one every 6–8 s of instrumental": gaps under 6 s get none; longer gaps are cut into round(gap/7) slots (5.3-10.5
  s). ✔
- The directors get the tempo, not the beat grid; drum runs only when the piece has no FX config.
- Per-line prompts also get the cue (mode + note), the singer description and the shot's place in the story.
- Hands-on files: prompts write `<song>-visual-prompts-true.json` and `full-metadata-true.json`, not `shots.json`
  (that is stage 6); `story-scenes.json`, `storyboard.md`, `universe.workshop.txt`, `visual-look.workshop.txt` omitted.
- `treatment.py` runs only with `DIRECTOR=1` (experimental).

### 6 Create key frames
- The stranger check is YuNet + hair lightness (SFace can't tell her from a dark-haired stranger), not SFace against
  the photo. ✔
- The vision-model critic is not wired in (should be marked new). ✔
- "Redraw up to 3 times": one draw plus at most 2 redraws (3 rounds), and only blank / stranger / face-in-part
  flags trigger one; near-duplicates are only reported. ✔
- "Storyboard sheet → to your review" and "her expression named": lean only. ✔
- Shots are bars, not lines: one still per bar (73-112 stills per song), merged to 33-46 scenes by
  `merge_scenes.py` (an LLM step, at most 3 bars per scene). `shots.json` is written here, not in stage 5.

### 7 Create the scenes
- InfiniteTalk int8 runs only on gruntus; the Sparks and tritus run full-precision Wan2.1-I2V-14B-480P with the
  lightx2v 4-step LoRA. Default lip-sync hosts include tritus.
- "needs at least 3.2 s of audio": ~3.3 s (`docs/log.md`).
- LLM "not measured": ~2-3 min for 40 transitions.
- The review pause (`HOLD_BEFORE_VIDEO`) comes after the scenes/transitions LLM step, not before stage 7.

### 8 Assemble the video
- Order: retime → finish (grade, deflare, mux) → lip-sync render → splice; the singing clips are spliced into the
  graded video and are not graded themselves (FX re-applied only). Inserts span the sung line ±0.25 s, not shot
  boundaries.
- Output name: `<piece>-v<N>.mp4` (e.g. `new-parade-v4.mp4`), not `<song>.mp4`. ✔
- Light flashes on drum runs: opt-in (`FX=`), only Noble Hunt uses it.
- Missing: the beat pulse is on by default (`PULSE=1`); R.Amber pieces turn it off.
- Grade: `bw` (default), `color` or `treatment`; no per-universe grade.
- Retiming: a straight-line time map between anchors, nearest frame; the page's "delete or repeat frames" is right
  about no interpolation, which conflicts with the project rule that beat sync uses FILM.

### End tables
- 4090 / offload wording is stale: the rule is now "offload judged by cost"; MOSS on the 4090 ran ~22 s/still with
  offload (`hardware-performance.md`). The code still hard-codes no offload.
- Boxes: gruntus also runs int8 lip-sync; tritus also runs LTX and InfiniteTalk.
- ComfyUI recommendation calls stage 2 a per-shot step and stage 8 GPU-heavy; neither is.

## Not verified (no evidence found either way)
Supported-architecture lists; ACE-Step XL "4B"; memory estimates marked "estimate"; "250 GB of model weights"
(thing1 has 301 GB); "42 GB video model + 26 GB text reader" (files: ~43 and ~27 GB); ComfyUI node availability;
the M5 Max and Strix Halo image timings.
