# POC 29: Keeping the music, changing the performance (ACE-Step plan reuse)

**Date:** 2026-11-10 · **Status:** works · **Piece:** unrequited (lyrics and style) · **Box:** gruntus, RTX 4090

**Hypothesis:** ACE-Step's planner writes a plan of the song (5 "audio codes" a second) before the music model renders
it. Rendering the same plan with new seeds should keep the song (length, structure, chords, where the lines fall) and
change only the performance, so a bad take could be re-sung without losing the music.

**Test:** XL-sft + 4B planner, 50 steps, offloading off. A base take (seed 2), its plan saved (`ace_gen.py` now writes
`<take>.codes.txt`), then three takes from that plan (`--codes-file`, seeds 3-5). Control: Unrequited's 25 October take
(same lyrics and style, its own plan). Compared with `pipeline/compare_takes.py`; files in `pieces/_bench/eval/ace-codes/`.

| Take | Length | Lyric words heard | Harmony vs base | Rhythm vs base | Sung lines starting within 1 s of the base's |
|---|---|---|---|---|---|
| Base (seed 2) | 215 s | 88.5% | 1.00 | 1.00 | — |
| Same plan, seed 3 | 215 s | 98.7% | 0.52 | 0.86 | 7 of 30 (median 1.1 s off) |
| Same plan, seed 4 | 215 s | 84.5% | 0.81 | 0.86 | 19 of 26 (median 0.5 s) |
| Same plan, seed 5 | 215 s | 100% | 0.83 | 0.80 | 25 of 37 (median 0.4 s) |
| Control (own plan) | 175 s | 85.8% | −0.06 | 0.59 | 1 of 13 (median 28.9 s) |

(Harmony: frame-by-frame correlation of which notes sound. Line starts from Whisper's segments, about 1 s resolution.)

**Result:** the plan keeps the song. Same length to the tenth of a second, the singing in the same places, mostly the
same chords (0.81-0.83; seed 3 drifted more at 0.52), against an unrelated control at −0.06. All three re-sung takes
were clear. Each re-sing took ~31 s against ~55 s for a fresh take (the planner's work is skipped).

**Found on the way:**
1. **ACE-Step doesn't repeat itself.** The same seed and settings gave a 215 s song on 10 November and a 175 s one on
   25 October, so a good take can't be recovered later: its plan must be saved when it is made (now done).
2. **With a plan supplied, the planner still picks its own length** unless told: seed 5 asked for 260 s against a 215 s
   plan and ran out of memory. `ace_gen.py` now takes the length from the plan.
3. **The sense check could fail a good take.** Whisper, transcribing the full mix, fell into a loop on an invented
   line ("We are now at the end of our drive." seven times) and scored a 99.6% take at 8.4%; its random retries made
   the score differ run to run. With `condition_on_previous_text=False` (`ace_score.py`), two runs gave identical
   scores. Scores are a little lower than before (88.5% for the take that read 99.6%), which is fine for a sense check.

**Use:** when a take fails the sense check or has a bad stretch, re-sing it from its plan rather than starting again.
The plan also fixes the structure, so it is the way to stop the planner adding sections between takes.
