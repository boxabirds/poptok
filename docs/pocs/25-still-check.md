# POC 25: An automatic still check before the video step

**Date:** 2026-10-24 · **Status:** built into `bin/make_video.sh`; validated on four runs · **Script:** `pipeline/check_keys.py`

## Why

The full Slippery Road v1 had two near-white MOSS stills (they became the one white flare in the video) and two shots
where Qwen-Image-Edit drew a dark-haired stranger instead of her. Each was visible in the stills an hour before the
video step, and re-rendering a finished video chain costs two hours.

## What it does

After the keyframes, `check_keys.py --reroll` deletes flagged stills and records an attempt in `reroll.json`; the next
keyframe pass draws them again with a new seed (`1000 + shot + 7919 × attempt`), up to 3 rounds, at most 2 re-rolls a
shot. Flags:

| Flag | Rule |
|---|---|
| blank | mean brightness above 215 or contrast below 12 |
| stranger | a scene / sing shot with a lit, frontal face whose surrounding hair is dark (lightest of three regions around the face below L* 12; her photo 51) |
| face in a part shot | a part shot (hand, back, reflection) with a frontal face over 4% of the frame |
| repeat | reported only: the third and later near-identical consecutive stills |

**Why hair, not face recognition:** SFace rated the dark-haired stranger 0.40 against her photo and her own shots with
hair over her face 0.14–0.17; it ignores hair, and her long blond hair is what identifies her. Hair lightness around
the face: 18–53 on her shots, night scenes included; 1–10 on the strangers.

## Validation

| Run | Flagged | Correct? |
|---|---|---|
| Slippery Road v1 (87 stills) | shots 38, 39 blank; 63, 64 stranger | all four real; no false alarms (the upside-down puddle reflection, flagged by SFace, is not) |
| Chronophobia v3 (106) | none | an earlier version flagged shots 23–24 (her, against a dark wall); fixed by using the lightest region |
| Chronophobia 30 s test (16) | shot 5, face in a part shot | real (the mirror shot) |
| Slippery Road 45 s test (18) | none | correct |

Not caught: her face given to a second person (Chronophobia v3 shots 67–69); that needs the cues to describe a second
character.
