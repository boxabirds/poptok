# POC 08: Master lyrics on Whisper's clock (C6)

**Date:** 2026-10-23 · **Status:** adopted; C6 is the reference method · **Piece:** offline

## Hypothesis

C5's garbled imagery and text come from Whisper's mishearings. Taking the words from the master lyrics and only the
timing from Whisper, and asking for no on-screen text, removes them without losing what made C5 good.

## Test

`align_lyrics.py` puts the master lyrics on Whisper's word timings (42 lines; 5 name- or number-heavy lines have
approximate timing, which is harmless since changes snap to downbeats). Then per-line prompts (music2vid's
LINE_PROMPT, full lyrics as context, "no text"), a style string appended, MOSS one still per bar, merge, LTX-2.5
chain with the Transition LoRA, retime, low-hits pulse, black and white grade.

## Result

- Garbled text gone: no "WEDNESSAY", "WIKIPDA"; only icons (warning triangles, a padlock, a key).
- 36 scenes against C5's 23, a full 2:11, 20 pulses on real bass hits.
- The stills came out red and blue neon, but the black and white grade fixes that.
- The "no text" negative wording was too heavy; "remove all text and logos" was the better phrasing.
- User: "C6 was nearly perfect". Video: `pieces/offline/outputs/C6/C6_true_lyrics-20261023-best.mp4`.
