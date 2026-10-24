# POC 16: Qwen-Image-2.1 on Strix Halo

**Date:** 2026-10-23 · **Status:** ruled out for now · **Host:** tritus

**Hypothesis:** with offload off and AOTriton attention, Qwen-Image-2.1 (768 px, 30 steps) is usable on tritus.
References: gruntus 21.8 s (single prompt), 40.3 s (blended).

**Result:**

| # | Setup | Frame 0 (single) | Frame 1 (blended) | Frame 2 (single) |
|---|---|---|---|---|
| 1 | A: offload on | 1067 s | — | — |
| 2 | B: offload off | 864 s | 3639 s | 608 s |
| 3 | C: offload off + AOTriton | 511 s | 2996 s | **94 s** |

The times swing wildly within one run. C's last frame (94 s, about 4× the 4090's 21.8 s) suggests most of the
earlier time is one-off warm-up or memory thrashing, not steady state. Even so, the blended frame at 50 minutes
rules tritus out for Qwen-Image; it does MOSS only. Open: a longer run to see whether C settles near 94 s.
