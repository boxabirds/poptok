# POC 24: One-word styles (macvideo's "filter")

**Date:** 2026-10-23 · **Status:** **passed**: strong styles reshape every frame on MOSS ·
**Script:** `pipeline/style_bench.py` · **Piece:** slippery-road

## Hypothesis

A style of a few words ("stained glass", "boxes and string", "cyberpunk") reshapes every frame, not just tints it, when
the LLM first writes a style brief from it and then writes each line's prompt inside that brief. This is how
macvideo's filter worked (pocs/19: a world brief "rendered ENTIRELY in the {filter} style", then per-line prompts with
an abstraction level 0–100), and the user found it had a massive effect, and raised the realism bar by not trying for
photorealism.

## The one change from macvideo

macvideo's brief also invents the narrator, setting and a central metaphor. That is what overrode the lyrics in the
rejected treatment run (POC 12). Here the brief may say only what the world is made of and how it's lit (materials,
textures, palette, line quality, light, camera feel); the lyric line (and later the Amber cue) decides what's in it.
The songwriter's treatment still sets setting and mood.

## Test

6 Slippery Road lines without Amber (the train in the night, snakes of light, plastic bottles in the wind, dark empty
homes, black dogs, the slippery road) × 7 rows: today's method (style appended to the prompt, no brief) and six
styles (stained glass, boxes and string, cyberpunk, cyanotype, claymation, oil impasto) at abstraction 25. MOSS on
tritus, 1024×576, same seed per line across styles. 42 stills, about 55 minutes.

## What we expect

Every style row clearly in its medium (cardboard and twine trains, leaded-glass streetlights), each column still
recognisably its lyric line, and the style row visibly stronger than the appended-word baseline. Risk: MOSS may be
weaker at strong media than Gemini was.

## How it feeds the rest

Q1 (placing Amber, POC 22) uses style words as its looks; piece.env gets `STYLE=` (a few words) and `ABSTRACTION=`.

## Result

Sheet: `pieces/_bench/styles-slippery/sheet.jpg`; briefs and prompts in `plan.json`. 42 stills on tritus, 76 s each.

| Style | Verdict |
|---|---|
| suffix (today's method) | Photographic night scenes in the treatment's blue; the baseline |
| stained glass | **Strong**: leaded glass everywhere, the snake of light in amber glass, glass dogs |
| boxes and string | **Strong**: a cardboard train, cardboard houses with cats in the windows, cardboard dogs, twine |
| claymation | **Strong**: clay train, clay bottles, clay dogs, a clay cat at a TV |
| oil impasto | **Strong**: palette-knife paint throughout |
| cyanotype | **Strong**: Prussian-blue prints |
| cyberpunk | Weak: close to the baseline (the treatment's rainy London already reads cyberpunk-ish) |

Every column is still recognisably its lyric line, so the brief restricted to materials and light did not take over
the content. macvideo's method works locally on MOSS. Next: whether Amber survives these styles (POC 22 Q1).

**User review (2026-10-24):** oil impasto is the favourite treatment; for Amber, claymation (stained glass also great).

## Full rendition: Slippery Road in stained glass (2026-10-24)

`pieces/slippery-road-glass/outputs/v1/slippery-road-glass-v1.mp4` (4:15). Same lyrics, cues and Amber as Slippery Road
v1; `STYLE_WORD="stained glass"` + a brief written for the song, in every line's LLM prompt and leading every image
prompt, hers included. Stills: `pieces/_bench/eval/slippery-glass-all-stills.jpg`.

- The style holds over the whole song, Amber included: a leaded-glass figure in Chorus 1, hair as glass streams in the
  bridge. The story survives the style (glass wine glasses at "keep the wine glasses full", glass dogs, the hand).
- Still check: nothing flagged. Video check: one flare (2:13, a white glass hand), jumps on pulses and fast motion.
- Weak: some of her later shots drift back towards photographic; Chorus 1 and the outro repeat.
