# POC 21: Three characters, kept apart (macvideo POC 34 on local models)

**Date:** 2026-10-23 · **Status:** queued after POC 20 · **Script:** `pipeline/cast_bench.py`,
fixture `pipeline/fixtures/multicast16.json` (copied from macvideo POC 34)

## Hypothesis

Choosing reference images by who is in the scene keeps three different characters (Alan, the bearded mariner; Barry,
a bald dockworker; Christy, a woman in a red coat) distinct across singles, pairs and trios. Nobody leaks into a scene
they're absent from, and no two merge into one face.

## Test

16 scenes: each character alone, every pair, the trio, each alone again after the group, the trio twice more, each
pair with the third explicitly absent, a final lineup. Three policies:

| Policy | References | Box |
|---|---|---|
| `text_only` | none: MOSS from the cast descriptions (baseline) | thing2 |
| `min_cover` | macvideo's: the fewest earlier stills that together show everyone needed. Chains through group shots (the trio at scene 15 refers to 11, which refers to 10, then 6) | thing1 |
| `anchors` | each character's first solo still, for whoever is in the scene (POC 33's lesson applied to a cast) | thing2 |

At most 3 people per scene, so at most 3 references.

## What we expect

`text_only`: plausible but faces vary between scenes. `min_cover`: holds early, risk of drift through the chained
group shots. `anchors`: the most stable. In macvideo, Gemini passed all three policies, so this fixture didn't
separate them; a local model may.

## How to judge

`pieces/_bench/multicast16/<policy>/contact-sheet.jpg` and `comparison.md`: required cast present, absent cast
leaked, identities merged.

## Result

Queued.
