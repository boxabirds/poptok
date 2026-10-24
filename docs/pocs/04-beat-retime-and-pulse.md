# POC 04: Scene changes retimed onto downbeats + an occasional bass pulse (C3b, C3c)

**Date:** 2026-10-23 · **Status:** adopted (`retime_to_beats.py`, `finish.py --punch-lowhits`) · **Piece:** offline

## Hypothesis

The music lands harder if every scene change sits exactly on a downbeat and the picture pulses gently only on real
low-frequency hits, rather than punching on every beat.

## Test

- C3b: C3 with no added flash, a non-compounding punch, LTX flares pulled down to local brightness (deflare).
- C3c (2-bar pulse): zoom peak 1.05×, fading over about 0.35 s, on the heavier of alternate downbeats (energy below
  150 Hz in the drum and bass stems).
- C3c (low hits): the pulse only on real bass hits: at most one per bar, the strongest half only, at least 1.5 s apart.

## Result

- The user's feedback: "the pulsing was too intense: best at most every bar or 2 bars on the low downbeat". An early
  D build pulsed "completely random… out of time with the music"; the low-hits rule fixed that.
- C3c_lowhits: "also very good". The low-hits pulse is the default, and the user wants it kept ("the occasional pulse
  worked really well").
- Video: `pieces/offline/outputs/C-D/C3c_chain_lowhits.mp4`.
