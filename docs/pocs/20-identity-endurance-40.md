# POC 20: One character across 40 trap scenes (macvideo POC 32 on local models)

**Date:** 2026-10-23 · **Status:** identity passes; composition copies the anchors · **Script:** `pipeline/identity_bench.py`,
fixture `pipeline/fixtures/mariner40.json` (copied from macvideo POC 32)

## Hypothesis

With fixed identity anchors as references, Qwen-Image-Edit-2511 keeps one invented character (the weathered mariner)
the same person across 40 scenes built to make him drift: a woman in the foreground, bald and younger men behind
him, profile, silhouette, fog, reflection, partial face, holding a photo of a woman, then a return to the opening
shot and a final portrait.

## Test

- Still 0 by MOSS from the text brief; stills 1–2 by Qwen-Image-Edit with the stills before them as references.
  These three are the fixed anchors.
- Stills 3–39: Qwen-Image-Edit with the 3 anchors only. This is macvideo POC 33's passing policy (their rolling
  references poisoned later scenes, and so did ours in POC 18). Because each scene depends only on the anchors,
  thing1 renders 3–20 and thing2 renders 21–39 at the same time.
- 1024×576, 40 steps, true CFG 4; same brief, scene prompts and prompt suffix as macvideo.
- POC 33 isn't ported separately: it existed to strip the provenance metadata Gemini embeds in its images, which
  grew with every reuse. Local PNGs carry none, so 32 and 33 are the same test here.

## What we expect

Pass, as Gemini did: no hard failures (woman, bald, younger, clean-shaven or background figure replacing him) across
40 stills. Expected weak spots: the distractor scenes, silhouette and fog, and the treatment-free photographic look.
Speed about 580 s per still with 3 references.

## How to judge

`pieces/_bench/mariner40-anchors/contact-sheet.jpg`, scored in `assessment.md` with macvideo's POC 32 table
(1–5 per scene for same person, swaps, beard, eyes, cap, build, wardrobe, primary subject, distractor takeover).

## Result

**Speed first.** The full model (40 steps, true CFG) took about 580 s a still with 3 references. The Lightning LoRA
(`lightx2v/Qwen-Image-Edit-2511-Lightning`, 8 steps, no CFG) takes about 61 s, with the same identity hold and slightly
better prompt following on four scenes compared side by side. All 37 scenes ran with Lightning, split over both
Sparks: 21 minutes. Lightning is now the default (`pipeline/qwen_edit.py`).

Sheet: `pieces/_bench/mariner40-l8/contact-sheet.jpg`. Face recognition (`pipeline/eval_identity.py`, SFace) against
anchor 0: **41/41 faces found, similarity mean 0.86, lowest 0.41** (the window profile); none below SFace's 0.363
same-person threshold.

| # | Criterion | Verdict |
|---|---|---|
| 1 | Same person, 40 scenes | **Pass**: same beard, eyes, cap, jacket throughout, through the return-anchor and final portrait |
| 2 | Distractors | **Pass**: the red-coat woman, bald and younger men, photos of a woman and of his younger self all stay secondary |
| 3 | Framing traps | **Fail**: nearly every scene copies the anchors' pose and framing (three-quarter, hand on a rail). No real profile, silhouette, hood, small figure on a wide pier; fog weak |
| 4 | Text | One garbled sign (barber scene) |
| 5 | Artefacts | White specks on his hands in many scenes (Lightning) |

**Reading:** fixed anchors hold identity, and also hold composition. For Amber, whose shots use her photo as the
only reference, expect the photo's pose to leak. Remedies to test: anchors of the character in several poses and
framings (a character sheet), anchor images cropped to the face, or one anchor plus a pose/composition still from MOSS.
