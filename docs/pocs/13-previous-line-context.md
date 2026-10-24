# POC 13: C6 per-line prompts + the previous line for continuity (offline-local v2)

**Date:** 2026-10-23 · **Status:** rendered; awaiting the user's verdict against C6 · **Piece:** offline-local

## Hypothesis

Following music2vid, giving the LLM the previous line and the visual it wrote for it ("for continuity: the previous
line was … and its visual was …"), alongside the full lyrics, makes neighbouring scenes follow on from each other
without overriding each line's content.

## Test

`prompts_per_line.py` with previous-line context and `--lyrics-context`; the treatment text used only as the style;
everything else as C6. 10-second test first, then the full song (47 scenes).

## Result so far

- The 10-second test looked like C6: black and white, gritty, line by line. One flaw: a white flare building up
  slowly in the first 4 s, below what deflare catches.
- The full render's stills follow the lyrics (server corridors, an orb, a crumbling stack of code, a bursting vault,
  a padlock, a fortress under lightning), and neighbouring stills follow on from each other. They have red, purple
  and gold accents that the black and white grade will remove.
- Video: `pieces/offline-local/outputs/v2/offline-local-v2.mp4` (2:12, 3160 frames, 47 clips, 21 bass-hit pulses).
- Frame-by-frame check: one white flare (86 s, +31 brightness over its surroundings), against 15 of 50 changes in C3
  and 9 of 22 in C5. The 23 clusters of big frame-to-frame change are either pulses or continuous fast motion
  (debris bursting outward at 13.6–14.9 s, the orb turning into a floor of scattered papers at 22.7 s, a figure
  running between racks at 56 s), not seams or resets.
