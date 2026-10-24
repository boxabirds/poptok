# POC 09: Generating the song with ACE-Step 1.5

**Date:** 2026-10-23 · **Status:** adopted (`make_song.sh`, `ace_gen.py`, `ace_score.py`); must move to a Spark ·
**Piece:** offline

## Hypothesis

ACE-Step 1.5 can sing a long, name-heavy rap lyric intelligibly, and Whisper can rank the takes by intelligibility.

## Test

Six takes (XL-sft, XL-turbo, 2B turbo; 2 seeds each), first without, then with the 4B planning LM. Score: share of
lyric words Whisper hears in order.

## Result

| # | Model | Without LM | With 4B LM | Time per 150 s song |
|---|---|---|---|---|
| 1 | XL-sft, seed 2 | 73% (the user's `-best` pick) | **77%** | 23 s → 59 s |
| 2 | XL-sft, seed 1 | 69% | 77% | 24 s → 64 s |
| 3 | XL-turbo, seed 1 | 70% | 75% | 8 s → 46 s |
| 4 | 2B turbo, seed 2 | 44% | 70% | 6 s → 45 s |

- The LM gets the length right: 339–354 words heard for 353 written, against 250–401 without it.
- The scores have a ceiling: Whisper struggles with names like "LiteLLM" and "Zeabur".
- XL + the 4B LM is over 24 GB, so it can't run on gruntus without offload. It belongs on a Spark (no ARM image yet).
- Pick: `pieces/offline/outputs/songs/ace_nolm/offline_xl-sft_s2-best.wav`.
