# POC 11: 10-second test renders from a slice of the full analysis

**Date:** 2026-10-23 · **Status:** adopted (`./poptok pieces/NAME --test SECONDS[@START]`, `slice_analysis.py`)

**Hypothesis:** a short test clip can tell us whether a method looks right before a 1.5-hour render, without
re-running stems and Whisper on the short clip.

**Test:** crop the song, cut the full song's analysis (stems, word timings, segments, beats) to the window and shift
it, then run the normal pipeline on a `pieces/NAME-testN` copy.

**Result:** works. Offline-local's 240-frame test takes about 6 minutes of LTX on a Spark. It caught the treatment
problem (POC 12) and Chronophobia's identity problem (POC 18) cheaply.
