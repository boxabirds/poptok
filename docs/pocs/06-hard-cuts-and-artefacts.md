# POC 06: One LTX clip per scene with hard cuts (stack D), and what caused its artefacts

**Date:** 2026-10-23 · **Status:** dead end for the cut style; the diagnosis led to POC 08 · **Piece:** offline

## Hypothesis

One image-to-video clip per scene, hard-cut on the downbeat, is a cleaner alternative to the continuous chain.

## Test

`D_ltx_cuts.mp4`: 25 clips, each animated from its scene's still, trimmed (not resampled) so motion plays in real
time. Then a 2×2 artefact test on the three worst scenes: new seed, camera-only prompt, text-free still.

## Result

The user found "gen 1" artefacts: a hooded figure walking backwards, crowds merging and vanishing, "PHP" shown as
"D.HP", and a random pulse. The diagnosis:

| # | Artefact | Cause |
|---|---|---|
| 1 | "D.HP" | **Transcription.** Whisper heard "PHP" as "D.H.P."; the LLM wrote a prompt asking for that text; LTX drew it |
| 2 | Backwards walking | Mostly the model: the figure moved in all variants, even with "stay still" |
| 3 | Crowds merging | The model: present in every variant, even camera-only |
| 4 | Text-free still variant | My test failed: the prompt still asked for text, so MOSS kept it |

The chain (C5) beat the cuts. Finding 1 led to POC 08: master lyrics instead of Whisper's words.
