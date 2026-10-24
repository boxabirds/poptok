# POC 07: LTX settings permutations (LoRA strength, even light, lead)

**Date:** 2026-10-23 · **Status:** ran in part; no comparison was reported · **Piece:** offline (first ~20 s)

**Hypothesis:** one of these beats C5's settings: Transition LoRA at 0.6 instead of 1.0 (more transformation, fewer
stock flares); no even-light wording (does it actually reduce flares?); a 0.6 s lead instead of 0.9 s (tighter
sync).

**Test:** the first 9 clips of the merged plan on thing1, each variant changing one thing (`ltx_p_lora06`,
`ltx_p_noeven`, `ltx_p_lead06`).

**Result:** `p_noeven` finished and `p_lora06` started, but the comparison against C5 never got reported before work
moved on. poptok kept C5's settings (LoRA 1.0, `--even-light`, deflare 8). Open: rerun on a free box and score flares
and sync.
