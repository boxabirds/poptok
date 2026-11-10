# POC 30: One look across a video, with and without the singer

**Date:** 2026-11-10 · **Status:** done · **Piece:** cuckoo (11 lean shots, 6 with Amber, vine charcoal) · **Boxes:** thing1, thing2

**Problem:** lean draws shots without Amber with MOSS and shots with her with Qwen-Image-Edit "native" (D46), so the
video looks drawn by two artists (cuckoo, 2026-10-29). Which way of drawing gives one look without losing her?

**Test:** the same 11 shots four ways (`lean/consistency/run.py`, `sheet.py`); stills with her: the try most like her
(SFace), as `lean/keyframes.py` picks. A and B already existed in `pieces/cuckoo/lean/keyframes/`; C and D drawn here
(`pieces/_bench/lean/consistency/`). "Look gap": 5 drawing statistics (brightness, contrast, near-black, near-white,
stroke density), the distance between the average still with her and without her in units of their spread (rough).

| Row | Method | Look gap | Her face (mean of 6) | By eye |
|---|---|---|---|---|
| A | today: MOSS without her, Qwen native with her | 1.19 | 0.46 | two artists: soft grey charcoal vs. black graphic ink with splatter |
| **B** | **Qwen for every shot (RENDER=qwen)** | **0.47** | **0.46** | **one look**, but Qwen's (graphic ink, splatter), not the brief's soft vine charcoal |
| C | MOSS for every shot; Qwen puts her into MOSS's still (still first as the canvas, her photo second) | 1.42 | 0.39 | her shots became soft pencil portraits, the scene mostly wiped; still two looks |
| D | Qwen for every shot, one style still (shot 01) as reference | 1.97* | 0.41 | fails: shots without her copied the reference's forest (D02-D04, D07 near-identical) |

*D's gap is inflated: its shots without her are near-copies, so their spread is tiny.

Sheet: `pieces/_bench/review/2026-11-10/one-look-sheet.jpg` (rows A-D, shots 01-11, face scores on her shots).

**Result:** one renderer for everything (B) halves the look gap at no cost to her face; neither way of keeping MOSS's
look (C) nor anchoring Qwen to a reference (D) works as tried. The cost of B is that the whole video takes Qwen's
style, which drifts from the brief (STYLE_NOT exists for that).

**Next:** B as the default for pieces with Amber; then make Qwen hold the brief's medium (STYLE_NOT wording, or a style
LoRA trained on approved stills, as the felt room's LoRA was).
