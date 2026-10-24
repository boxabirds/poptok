# POC 23: Can the local LLM write the actor cues?

**Date:** 2026-10-23 · **Status:** works; judgement calls differ from the hand version · **Script:**
`pipeline/cast_cues.py` · **Piece:** slippery-road

## Hypothesis

Qwen 3.8 27B (llama.cpp on the 4090), given plain lyrics, the visual treatment and the user's own `[Singer: …]`
directions, can decide line by line when and how the performer appears (none, pov, part, scene, sing), follow the
story (hold her face back until a reveal), obey the user's directions, and write short staging notes, without
changing a sung word.

## Test

- Input: `pieces/slippery-road/inputs/lyrics.before-annotations.txt` (the user's lyrics with two `[Singer: …]`
  directions, no cues) and `visual-treatment.txt`.
- The LLM returns JSON (sections and line cues by number, never text); the script renders the annotated lyrics and
  refuses to write if any sung line changed.
- Reference: the hand-annotated `inputs/lyrics.txt`. It's one opinion, not ground truth.
- 3 runs at temperature 0.3; `--score` compares the mode of every line.

## Result

| Run | Time | Same mode as reference | Same "face on screen" yes/no | Lines with her face (reference 18) | Reveal placed at |
|---|---|---|---|---|---|
| 1 | 19 s | 26% | 69% | 5 | "it's me", as a hand close-up |
| 2 | 17 s | 40% | 81% | 10 | Chorus 1 |
| 3 | 18 s | 31% | 81% | 10 | "it's me", her hand on wet concrete |

Outputs: `pieces/slippery-road/analysis/cues/qwen_{1,2,3}.txt` (and `.json`).

**What it gets right, every run:** words untouched; the user's directions followed exactly (Chorus 1 sung on the
rainy street; Chorus 2 as a scene outside the flat with the key); passengers averting their eyes from the camera; the
hand callback at "It's my hand"; Verse 2 without her; Verse 3 through her eyes.

**Where it differs from the hand version:** it uses her face far less (5–10 lines against 18); it treats Chorus 2 as
acting rather than singing (a literal reading of the user's direction); it keeps the bridge as pov where the hand
version has her on screen; it ends with none, not a final sung line. Its notes are short and plain
("Passengers avert eyes from camera").

**An issue it surfaced:** run 2 puts the reveal at Chorus 1, because the user's Chorus 1 direction already shows her
face, before the lyric reveal at "it's me". The hand version has the same conflict without flagging it.

## Decision

The mechanism works: fast, safe (words can't change), obeys user directions, follows the story. Use it as the
proposer, with the user editing the result, as planned. To consider: tell it a target share of face-on-screen lines;
pick the best of several runs; richer notes.
