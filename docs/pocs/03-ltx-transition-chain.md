# POC 03: One continuous LTX chain with the Transition LoRA (stack C, C3)

**Date:** 2026-10-23 · **Status:** adopted; it's the core of poptok · **Piece:** offline

## Hypothesis

LTX video with the LTX-2.3 Transition LoRA, run as a chain where each clip starts from the end of the previous one
and is steered toward the next bar's still, gives one continuous shot in which each scene transforms into the next.
That is the "morph" music2vid always wanted.

## Test

`ltx_chain.py` on thing2: one MOSS still per bar (51 bars), clips chained, scene changes aimed at downbeats.

## Result

- No visible seams. In the first 200 frames, the frame-to-frame change at clip joins (5.0–6.1) is below the median
  step (7.0).
- Changes land within −1 to +9 frames of their downbeats before retiming; ±125 ms in only 52% of bars.
- Problems: repeated scenes (consecutive bars with the same prompt each got a new still), 15 of 50 changes used a
  white light flare as the transition, and a compounding zoom punch.
- User: "C3_ltx_chain is the best so far. transitions are decent."

Fixes followed in POC 04 (finish) and POC 05 (scenes).
