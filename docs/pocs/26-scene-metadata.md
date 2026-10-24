# POC 26: Scenes as the unit of editing (`scenes.json`)

**Date:** 2026-10-24 · **Status:** built; used for Slippery Road v2 · **Script:** `pipeline/scenes.py`

## Why

Review notes come as frame numbers ("1379: her hand is in an impossible position"), but the pipeline thinks in bars,
shots and chain clips. Fixing one scene meant re-running everything, two hours of video for a few stills.

## What it is

`OUT/scenes.json`, built after the scene merge: one entry per scene (a run of bars sharing one keyframe, which is one
shot of the LTX chain) with its bars, times, frames, section, lyric, performer cue, keyframe prompt, keyframe file,
`transition_in` (how the previous scene becomes it), `motion`, `lipsync`, `reroll`, `notes`, `locked`.

| Command | Does |
|---|---|
| `scenes.py build OUT` | writes scenes.json (keeps earlier transitions, motion and notes for unchanged scenes) |
| `scenes.py frame OUT 1379 2037` | which scene each frame is in, or which transition it falls in (the chain changes scene 0.9 s before the downbeat) |
| `scenes.py transitions OUT` | the local LLM writes `transition_in` for every scene change (POC 27) |
| `scenes.py apply OUT` | pushes edits into shots.json / shots_merged.json, deletes the edited scenes' stills (bumping `reroll.json` for re-rolls) and only the chain frames that depend on them |

Then `./poptok pieces/NAME --run RUN` redraws only those stills (the still check runs again) and only those video
frames: the chain frames are mirrored to the video box, so it renders what's missing. `keyframes.py` now keeps an
existing shots.json (it held the edits) unless `--replan`.

## Slippery Road v2 (tactical fixes)

v2 started as a copy of v1 (stills, plans, chain frames). The songwriter's 21 frame references mapped onto 18 scenes;
22 scenes were edited (prompts, motion, casts, lip-sync flags), 39 transitions written; `apply`: 19 stills to redraw,
5,127 of 6,112 video frames to re-render (new transitions on every change touch most of the chain).
