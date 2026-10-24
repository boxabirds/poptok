# POC 05: One scene per distinct prompt, near-duplicates merged (C4, C5)

**Date:** 2026-10-23 · **Status:** adopted (`merge_scenes.py`) · **Piece:** offline

## Hypothesis

Repeated scenes in C3 come from giving every bar a fresh still. Merging consecutive bars with the same or nearly the
same prompt into one scene, continuing the shot through those bars with no new still, and asking for constant
lighting through each change will remove the repeats and reduce flares.

## Test

C5 on thing2: 51 bars merged into 22–23 scenes (exact duplicates plus obvious near-repeats merged by the LLM), the
even-light prompt, deflare, the low-hits pulse.

## Result

- User: "absolutely incredible — transitions are very intelligent". It became the reference video
  (`pieces/offline/outputs/C-D/C5_ltx_chain_scenes-best-20261023.mp4`).
- Flaws: garbled text ("WEDNESSSSDAY", "WIKIPDA") because the prompts came from Whisper's mishearings and asked for
  on-screen text; 9 of 22 changes still had LTX's own flares; longer scenes made many changes look like dissolves.
- The LLM still kept two vault-door close-ups as separate scenes. Merging by comparing the stills themselves is an
  open idea.
