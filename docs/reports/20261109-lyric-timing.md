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

## Results

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

## What this means

1. **Keep both.** Most lines agree, and their failures don't overlap much. The aligner isn't a replacement for Whisper,
   or the other way round.
2. **Fix Whisper's long-first-word error** in `align_lyrics.py`. Once that is fixed, re-run this comparison: Whisper's
   biggest losses all come from it.
3. **Then choose line by line,** not by song: where the two disagree, use the one that passes a check like this
   referee's, and flag lines that neither passes for review.
4. **The design page's claim** that the aligner finds words "even where voices overlap" came from the misdiagnosis. It
   has been corrected.
