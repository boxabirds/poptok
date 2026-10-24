# POC 17: The M5 Max MacBook (quintus) as an LTX-2.5 renderer

**Date:** 2026-10-23 · **Status:** passed; integration as a video host next · **Host:** quintus (M5 Max, 128 GB)

## Hypothesis

The MacBook can render poptok's LTX-2.5 chain natively on Metal (MPS), at a useful speed and with the same output, as
macvideo already does there. A Mac container can't reach the GPU, so it runs in a uv venv like macvideo.

## Test

`bin/setup_mac_renderer.sh` (uv venv, torch 2.14.1, LTX at the Sparks' commit e98f6d9). Models in `~/models/ltx-2.5`
as links to macvideo's weights, plus the current video VAE and the Transition LoRA. The same 240-frame,
4-clip offline-local test job as on thing2.

## Result

| # | Box | Time |
|---|---|---|
| 1 | thing2 (DGX Spark) | 377 s, including copying frames back and the finish |
| 2 | **quintus** | **353 s**, including model load |

Same scenes, lighting and quality; small differences in motion from different maths on different hardware. It adds
a third video box. To do: a native run path in `bin/hosts.sh` and a lock that works on macOS (no `flock`).
