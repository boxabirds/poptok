# POC 18: Identity from a reference photo (Qwen-Image-Edit-2511)

**Date:** 2026-10-23 · **Status:** identity passes; style and speed fail; next round needed · **Piece:** chronophobia

## Why

Chronophobia's 10-second test showed three different women in ten seconds. Every still is generated from text alone,
and text doesn't hold a face. LTX moves from one still to the next, so if the stills disagree about who she is,
the video does too. Text-only casting was also tried in the treatment "director" run (POC 12): one "Operator"
character described identically in every prompt still came out as different men.

macvideo solved this with Gemini 3.1 Flash Image and reference images (identity chain, then cast-aware references,
macvideo POCs 34 and 35). poptok must stay local, so this tests the same idea with open weights.

## Hypothesis

Qwen-Image-Edit-2511, given the singer's photo plus her two most recent stills as reference images, keeps her
recognisably the same person (face, hair, age) across scenes while following each scene's prompt and the piece's
visual treatment.

## Test

`pipeline/identity_poc.py`, in the `morph-spark` image on a DGX Spark (no offload):

```
ANCHOR=pieces/artists/r.amber.jpeg STYLE="<chronophobia visual treatment>" python3 pipeline/identity_poc.py OUT
```

- **Anchor:** `pieces/artists/r.amber.jpeg`, the singer (blond, black dress). It is a reference for every still.
- **Scenes:** five Chronophobia lines: window, cardigan slipping off its hanger, pressing flowers, paper boats in a
  drawer, hallway.
- **References per still:** the anchor + the 2 most recent stills (Qwen-Image-Edit takes up to 3 images).
- **Settings:** 1024×576, 40 steps, true CFG 4, fixed seeds.
- **Output:** one PNG per scene, `meta.json` (seconds and references per still), `sheet.jpg` (photo first).

## What we expect

| # | Criterion | Pass if |
|---|---|---|
| 1 | Identity | Same face, hair colour and age as the photo in all 5 stills |
| 2 | Lyric content | Each still shows its line's scene, not a copy of the photo with a new background |
| 3 | Treatment | The pointillist look applies to her as well as the scene |
| 4 | No drift | Still 5 is no further from the photo than still 2 (the anchor stops a rolling chain from wandering) |
| 5 | Speed | Under about 2 minutes per still on a Spark |

Most likely failure: the photo's pose and black dress get pasted into every scene ("pasted-on" look; in an
earlier morphexpts test, Qwen's 2-reference edit produced a stuck-on object). A secondary risk: the style is
weaker than with MOSS because the edit model leans toward the photographic reference.

## If it passes: casting in poptok

Use macvideo's **cast-aware minimal cover** (POCs 34–35), not a plain rolling chain. POC 33 there showed that
feeding a generated frame with a dominant distractor back in as a reference poisons later shots.

1. The local LLM writes a short cast description and tags each lyric line with who is in it (her, or nobody).
2. **Her lines:** Qwen-Image-Edit with her photo + her 2 most recent stills *that contain her*. Object-only shots
   in between don't break the chain.
3. **Lines without her:** MOSS with no references, as in C6, so she can't leak in.
4. **More performers:** one anchor photo each; a scene gets the smallest set of references that covers its cast.

## If it fails

Try, in order: anchor only (no rolling references); MOSS still + Qwen edit "put this person in"; a face LoRA trained
on a few photos of her.

## Result (round 1, thing1)

Sheet: `pieces/chronophobia/outputs/identity_poc/sheet.jpg` (photo first, then the 5 scenes).

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Identity | **Pass** | The same face, blond hair, blue eyes, age and black dress in all 4 stills she appears in, through still 5 |
| 2 | Lyric content | Mostly | Window, flowers in a book, paper boats in a drawer, hallway with eyes closed are all right. The hanger still drops her entirely and shows only the cardigan |
| 3 | Treatment | **Fail** | Photographic, not hand-tinted sepia. "Pointillist dots on skin" became literal dots on her cheeks, and "pressed-flower collage" became a wall of pinned flower cards in every scene |
| 4 | No drift | Face: pass. Scene: **fail** | Her face holds, but the rolling references copy props forward: the flower-card wall appears in 5 of 5 scenes, and two vintage portrait photos appear in stills 3–5. This is the poisoning macvideo's POC 33 warned about |
| 5 | Speed | **Fail** | 230 s (1 reference), 395 s (2), 579–589 s (3) per still against a 2-minute target. Time grows with each reference image |

**Reading:** the photo anchor does exactly what text couldn't: she is one person throughout. But the rolling
references carry the set dressing too, and Qwen-Image-Edit renders the treatment as props rather than as a look.

## Next round

1. **Anchor only** (no rolling stills): fixes the copied props and costs 230 s rather than 580 s.
2. **Treatment as a look, not props:** put the style through MOSS, which handled the treatment well in the
   Chronophobia test. Options: (a) MOSS makes the scene, then Qwen-Image-Edit swaps in her identity from the photo;
   (b) Qwen-Image-Edit makes her scene, then a light MOSS img2img pass applies the look.
3. **Fewer steps:** 40 → 20–25 to bring the time toward the target; compare quality.
