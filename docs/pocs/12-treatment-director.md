# POC 12: A whole-video treatment that writes every scene (offline-local v1)

**Date:** 2026-10-23 · **Status:** rejected; now opt-in only (`DIRECTOR=1`) · **Pieces:** offline-local, chronophobia

## Hypothesis

If the LLM first writes one treatment for the whole video (story arc, fixed cast and places, look, motion,
transitions, grade, palette per section) and then writes each line's scene inside it, the video becomes more coherent
and carries the user's visual treatment better than C6's per-line prompts plus a style string.

## Test

`treatment.py`, then scenes written from it; full offline-local v1 render.

## Result

The treatment replaced the lyrics. Nearly every scene became "the Operator in a server room":

| lyric | C6 | v1 |
|---|---|---|
| "No login required… walk through the door" | server room, steel doors wide open | the Operator at a cabinet |
| "Wikipedia, Archive-dot-org, GOG goin' dark" | a network of lights going out | server racks floating in a void |

The fixed character didn't even stay the same man; text doesn't hold a face. User: "C6 was nearly perfect… Get back to
that." Lessons: the lyric decides what's in a picture; the treatment controls only how it looks; consistent people
need reference images (POC 18).
