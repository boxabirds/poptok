# POC 27: Why some scene changes cut instead of morphing (the Transition LoRA's trigger)

**Date:** 2026-10-24 · **Status:** confirmed; fix in `ltx_chain.py` and `scenes.py transitions`

## The question

Offline's transitions were "fabulous scene-specific morphs"; Slippery Road v1 had plain cross-fades (frame 614) and
hard cuts (frame 792). Were the morphs only an LTX-2.3 thing?

## Finding

No. The morphs come from the LTX-2.3 Transition LoRA (valiantcat/LTX-2.3-Transition-LORA; our file is byte-identical),
which we load on LTX-2.5. Its model card shows that every prompt (1) describes how shot A physically turns into shot B
and (2) ends with the trigger word **"zhuanchang"**. We did neither: our prompt described only the destination scene.
Offline's quick, related scene changes morphed anyway; Slippery Road's slow, unrelated ones fell back to cuts.

## Test

Three of the flagged Slippery Road changes (614 snakes, 792 into the carriage, 1056 bottles), each rendered three ways
from the same stills and seed: `pieces/_bench/transitions/*_sheet.jpg` and `.mp4`.

| Variant | Result (t01-02, the snakes) |
|---|---|
| 1. today's prompt | hard cut at ~0.7 s |
| 2. today's prompt + trigger | hard cut: the word alone does nothing |
| 3. **an A→B description + trigger** | **a real morph**: the train's window lights pour out as streaks onto the wet road and become the snakes |

The other two: **t04-05** (snakes into the carriage) behaves the same and is the clearest: variants 1 and 2 hard-cut,
variant 3 morphs (the orange light-snake grows, slithers in among the passengers and fades into the carriage).
**t08-09** (into the bottles): all three change quickly; the scenes are already alike (dark tracks), and variant 3 is
only slightly smoother. A described morph helps most where the two scenes differ.

## Fix

- `ltx_chain.py`: every scene-change prompt ends with ", zhuanchang" (`--trigger`); when the scene has a
  `transition_in` in scenes.json the prompt is "One continuous shot, no cut: <transition_in> It settles on <scene>".
- `scenes.py transitions`: the local LLM writes the A→B description for each change; `make_video.sh` runs it after the
  scene merge. Hand-written ones can replace any of them.
