# POC 00: A temporally stable renderer for music2vid (morphexpts)

**Dates:** 2026-10-21 to 22 · **Status:** concluded; superseded by the LTX chain (POC 03) · **Where:** morphexpts (not in git)

## Hypothesis

music2vid's Deforum (SD 1.5 img2img on its own previous frame, prompts scheduled from the lyrics, zoom on the beat)
can keep its per-frame flow but get a modern image model. Conditioning on the previous frame and on reference images
would make the morphs stable instead of reinventing detail every frame.

## Test

Image-model loops run frame by frame, judged on consecutive-frame continuity, not on sparse contact sheets.

## Result

| # | Renderer | Verdict |
|---|---|---|
| 1 | Z-Image img2img loop | 0/10: reinvents the image each frame |
| 2 | FLUX.2 Klein | "complete crap", dropped |
| 3 | Qwen-Image-Edit-2511 identity/zoom loop | stable: zoom "smooth, noisy but ok" |
| 4 | Qwen-Image-Edit 2-reference "grow wings from image 2" | "disaster": the object looks pasted on |
| 5 | MOSS inversion-trajectory walk | dead end: keeps the photo until the last frame, then jumps |

A frame-by-frame image loop doesn't give real morphs. A video model does, and that led to POC 03.
