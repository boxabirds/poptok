# Lyric timing: speech recogniser or forced aligner?

**Written:** 9 November 2026. **Songs:** Cuckoo, New Parade, Noble Hunt, Slippery Road, Unrequited (180 lyric lines).
**Tool:** `pipeline/compare_lyric_timing.py`. **Results:** `pieces/_bench/eval/lyric-timing/lyric-timing.json`.

## Why we checked

Step 3 times every lyric line in two ways:

- **Speech recogniser, then matching** (`pipeline/analyse.py`, `pipeline/align_lyrics.py`). Whisper writes down what it
  hears, with a time per word. Our lyrics are then matched onto its words. Lines it didn't hear are spaced out between
  their neighbours.
- **Forced aligner** (`pipeline/force_align.py`). It is given our lyrics, and finds where in the vocal stem each word
  is sung. Since 24 October its timings are the ones the pipeline uses.

We added the aligner because Whisper missed whole choruses. But the cause turned out to be our own setting: a
voice-detection filter, trained on speech, cut out the sung choruses (`docs/log.md`, 24 October 16:45). With the filter
off, Whisper heard them. So it was never shown that the aligner was needed. All five songs here were analysed with the
filter off.

## How we judged

The two methods agree (start times within 0.3 s) on most lines. For each line where they disagree, the referee cuts
the vocal stem to each method's time window and asks Whisper, afresh and without the lyrics, what is sung there:

- **Starts:** are the line's first two words heard in the 2.5 s from its claimed start? This is the main test.
- **Inside:** how many of the line's words are heard in its claimed window? This is the tie-breaker. A window that is
  too long also passes it.

The referee is Whisper again, which is a bias. But it hears each clip on its own, so it can't copy its song-wide pass.

## Results (before the fix)

| Song | Lines | Agree | Disagree, both fine | Aligner right | Whisper right | Neither clearly right |
|---|---|---|---|---|---|---|
| Cuckoo | 19 | 14 | 3 | 0 | 1 | 1 |
| New Parade | 46 | 22 | 4 | 11 | 4 | 5 |
| Noble Hunt | 33 | 18 | 4 | 5 | 1 | 5 |
| Slippery Road | 42 | 29 | 6 | 1 | 4 | 2 |
| Unrequited | 40 | 31 | 5 | 2 | 2 | 0 |
| **All** | **180** | **114** | **22** | **19** | **12** | **13** |

So **136 of 180 lines (76%) are fine either way.** Where only one method is right, the aligner wins 19 to 12. **Neither
method is safe alone. They fail in different places, for different reasons.**

### Where Whisper goes wrong: lines that follow a silence

After a silence, Whisper stretches the first word back over the whole gap. Its raw word times show this:

- "The" runs 0.0–10.8 s at the start of Unrequited;
- "The" runs 89.5–99.3 s before "rockets" in Noble Hunt.

As a result, the first line starts at 0.0 s in 4 of the 5 songs (the real starts are 10.8–22.0 s), and lines after
instrumental breaks start up to 20 s early (Noble Hunt line 22). This is systematic and easy to fix. When a word lasts
implausibly long, the line's real start is near the word's end.

### Where the aligner goes wrong: singing that departs from the written lyrics

The aligner must place every written word, in order. Where the singer repeats, drops or stretches words, it still
fits them somewhere, and its timings drift late:

- the Slippery Road outro is placed ~19 s late, where nothing is sung;
- Unrequited lines 33 and 35 are ~10–20 s late;
- the end of New Parade (repeated "together") is unreliable in both methods.

The aligner's confidence score is low on most of these, but not reliably. A rule "use the aligner when its score is
at least 0.1, otherwise Whisper" picks the right one on 24 of the 31 lines where only one is right. Always using the
aligner gets 19, and always using Whisper gets 12. With only 31 decided lines, that rule could be overfitted.

## After fixing Whisper's early starts

`align_lyrics.py` now corrects the stretched first word. For a word Whisper says lasts over 1 s, it looks at the vocal
stem from the word's start to 0.5 s past its end. If there's a silence of at least 0.5 s in there, the word starts where
the silence ends. Shorter gaps are between the syllables of a held word ("Cuckoo"), and are left alone. Whisper's
timings for the 5 songs were made again with the fix (`*-true-whisper2-segments.json`; the pipeline's own timings were
not touched), and the referee run again (`pieces/_bench/eval/lyric-timing/after-fix/lyric-timing.json`).

| Lines right (of 180) | Before the fix | After |
|---|---|---|
| Whisper alone | 148 | **162** |
| Aligner alone | 155 | 160 |
| Best of both, line by line | 167 | **170** |
| Neither | 13 | 10 |

("Aligner alone" moves a little too, because the referee only judges lines where the two disagree, and the fix changed
which lines those are.)

- **Whisper alone is now level with the aligner, and its misses are smaller.** Its remaining misses:
  - 6 are within 0.4–1.4 s, including two in Cuckoo that the fix made slightly worse (a held "Cuckoo", "Fall the");
  - 2 are 13–16 s off: New Parade lines 39–40, which it didn't hear at all and spaced out between neighbours.
- **The aligner's 10 unique misses are mostly large:** 10–20 s late at the Slippery Road outro, Unrequited lines 33 and
  35 and the end of New Parade, where the singing departs from the written lyrics.
- **Both together are right on 8 more lines than Whisper alone,** but only if something chooses per line. Today the
  pipeline always takes the aligner's timings, so it gets 160, not 170.

## Decision (D50)

One timing method: Whisper on the vocal stem, with the fix. The forced aligner is no longer run by default
(`FORCE_ALIGN=1` in `bin/make_video.sh` turns it back on). Most of both methods' large errors come from songs that
don't sing what the lyrics file says (Unrequited's extra closing chorus), which is the next thing to fix.

## What this meant before the decision

1. **Whisper (fixed) should be the default.** It's as accurate as the aligner by count and fails less badly.
2. **The aligner is worth keeping only as a second opinion,** used per line: where the two disagree by more than
   0.3 s, take the one that passes the referee's check, and flag lines that pass neither (10 here, most of them in New
   Parade's repeated outro) for review. The referee costs one short Whisper pass per disputed line (about a minute per
   song on the RTX 4090).
3. **Not yet done:** the pipeline still uses the aligner's timings by default (`bin/make_video.sh`, `force_align.py
   --merge`). Switching to the per-line choice is the next step.
4. **The design page's claim** that the aligner finds words "even where voices overlap" came from the misdiagnosis. It
   has been corrected.
