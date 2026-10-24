# POC 14: Strix Halo (tritus) as a keyframe box

**Date:** 2026-10-23 · **Status:** adopted as the last box in the keyframe pool · **Host:** tritus (gfx1151, ROCm 7.13)

## Hypothesis

Tritus's slowness (386 s per MOSS still) comes from settings rather than hardware: CPU offload on shared memory,
and the slow default attention on AMD. With those fixed it can take keyframe overflow from the Sparks. Running two
jobs at once may add throughput.

## Test

The same 5 Chronophobia stills under each setup, run solo (except the pair test).

## Result

| # | Setup | Seconds per still |
|---|---|---|
| 1 | gruntus 4090 (with offload; not used any more) | ~22 |
| 2 | thing1 (DGX Spark) | 45 |
| 3 | tritus A: offload on, default attention | 386 |
| 4 | **tritus C: no offload, AOTriton attention (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`)** | **78** |
| 5 | tritus C, two jobs at once | 167 each (no gain) |
| 6 | tritus D: C + TunableOp | 77–82 (no gain) |

Setup C is 5× faster than A and is what `bin/hosts.sh` uses. Two jobs at once gives no more throughput, so tritus
stays single-job. Qwen-Image-2.1 on tritus is far too slow (POC 16). Background: `morphexpts/reports/Media generation
on Strix Halo.md`.
