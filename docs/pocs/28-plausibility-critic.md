# POC 28: A plausibility critic driven by the local Qwen

**Date:** 2026-10-24 · **Status:** promising (with segmentation: 4 of 4 caught, no false alarms, on the set the rules
were tuned on); not yet in the pipeline ·
**Host:** gruntus (Qwen3.8-27B Q4 + its vision projector `mmproj-F16.gguf`, llama-server)

## Hypothesis

The local Qwen3.8-27B, given images through its vision projector, can find the physically impossible moments the
user found in Slippery Road workshop v3 (three hands, a reflection that can't exist, a tram on a plaza with no rails,
a hand with no arm) without flagging clean frames or the video's intended surreal elements. If so, it can check every
still before the video is rendered (where a re-roll costs seconds) and every scene change afterwards.

## Test

`pipeline/critic.py`: one image per call, countable questions instead of "is anything wrong?" (each person and their
hands, hands that belong to nobody, the same person twice, each vehicle and what it stands on, each reflection and
whether its geometry fits, floating/merged objects), flags decided by simple rules on the answers. Each frame is sent
whole and as two 60%-wide crops at 1.6x (`--crops`). The scene's prompt is passed as intent so that what the scene asks
for, however surreal, isn't flagged.

Labelled set: 11 frames from `slippery-road-ws-v3-nosync.mp4`: 4 with real problems (62, 146, 148, 190 s), 7 clean
(16, 48, 72, 92, 176, 200, 228 s; including the intended shadow-self face in the window, the floating bottles and the
broken-glass faces).

## Result

| Run | Problems caught | False alarms on clean frames |
|---|---|---|
| One general question (no rules, no intent) | 1 of 3 tried (reflection) | 0 of 2 |
| Countable questions, no intent | (the tram and hand frames were mis-extracted in this run) | 4 of 8: the intended faces in windows, floating bottles; plus a rails-rule bug |
| **Countable questions + crops + scene intent** | **3 of 4**: tram on plaza without rails (62 s), the reflection (148 s), hand without a visible arm (190 s) | **0 of 7** |

Missed: three hands at 146 s (her two hands at her chin plus the hand carried over from the previous scene, on the
ground). Hand counting is the weak spot.

Files: `pieces/_bench/eval/slip_ws_v3/review/critic/` (frames, `critic_v2.json` with every answer).

## Step 1: segmentation (geometry) + Qwen on person crops

`docker/segment/` (image `segment`), `pipeline/segment_check.py`, `pipeline/critic.py --segment DIR`. Weights in
`~/models/segment/` (5 GB incl. the detectors tried).

**Models and licences.** SAM 3 (Meta, text concept prompts) is gated on Hugging Face behind Meta's SAM licence; our
token gets 403 (access not requested; accepting a licence is the user's call). Used instead: open-vocabulary boxes from
**LLMDet-large** (Apache-2.0; Grounding-DINO architecture, in transformers) and masks from **SAM 2.1 hiera-large**
(Apache-2.0). Detectors compared for hands/rails on the 11 frames: Grounding DINO base (also Apache-2.0; the
"Grounded-SAM 2" default) found the tram as a "hand", missed both chin hands at 146 s and drew "rail track" around any
wet ground; MM-Grounding-DINO large and OWLv2 large were no better; LLMDet found hands best and, on an upscaled crop of
the woman at 146 s, three separate arms.

**Rules** (on masks): loose hand (touches no person; hand score ≥ 0.4); detached hand (inside a person's mask but not in
the same connected piece as the body); extra hand / extra arm (> 2 attached to one person, counted on that person's box
+25% upscaled to 1024; persons < 30% of frame height skipped); reflection beside (each puddle crop flipped vertically
and searched for faces/people; one with no real person above it on about the same vertical is flagged); people inside
window/puddle masks recorded as reflections/images and their hands not counted; vehicle without rails **advisory
only** (to `notes`). Then Qwen gets each real person's crop with "how many hands/arms; does each hand connect by an arm
to this body?".

| Frame | Label | Qwen frame critic (this run) | Geometry | Qwen person crop | Combined |
|---|---|---|---|---|---|
| 16 | clean | - | - | (no people) | OK |
| 48 | clean (intended bottles) | - | - (note: train without rails: false) | (no people) | OK |
| 62 | tram, no rails | tram on road without rails | missed (plaza markings read as rails, cover 42%) | (person too small) | **caught** |
| 72 | clean | - | - | 0 hands, 1 arm | OK |
| 92 | clean | - | - | (no people) | OK |
| 146 | three hands | "reflection doesn't fit" (wrong reason; not flagged in the v2 run) | **3 arms on her** | missed: 2 hands, 2 arms | **caught** |
| 148 | impossible reflection | reflection doesn't fit | **face/person in puddle with nobody above** | 2 hands | **caught** |
| 176 | clean | - | - | 2 hands, 2 arms | OK |
| 190 | hand, no arm | hand without arm, floating | **detached hand** | hand not connected | **caught** |
| 200 | clean (intended faces in windows) | - | - | 5 crops, all ≤ 2 hands | OK |
| 228 | clean | - | - | 0 hands, 2 arms | OK |

Geometry alone: 3 of 4, 0 of 7 false alarms. Qwen frame critic alone: 3 of 4 (146 caught only for a wrong reason).
Combined: **4 of 4, 0 of 7**. The Qwen crop question added nothing here: it reads the clasped hands at her chin as one
and calls the ground hand connected.

**Timing (gruntus, 4090):** segmentation 3.7 s/frame (2.6–6.1; detection 2.5, SAM 0.16, person crops 1.0), 9.3 GB;
Qwen 1.9 s per person crop; the whole-frame critic 22 s/frame (3 views). Segment and Qwen together exceed 24 GB, so
they run one after the other.

**Caveats.** The thresholds and rules were tuned on these same 11 frames (the rails rule was demoted, the loose-hand
score floor and the 30% arm-count height were set after seeing false alarms), so this is a fit, not a test. The arm
count is noisy on small or crowded figures. Needs a fresh labelled set (other songs, more hands/vehicles) before it
gates re-rolls.

Files: `pieces/_bench/eval/slip_ws_v3/review/critic/seg/` (per-frame JSON + overlay PNG), `critic_seg.json`.

## Next

- Hands: crop each person tightly (ask the model for boxes, or use a hand detector) and count per crop.
- Things that appear from nowhere ("a train comes out of nowhere") happen between frames: send pairs or short strips
  from each scene change.
- Wire in: start the pipeline's LLM with `--mmproj`; critic on every still after the still check (flagged → re-roll,
  or Qwen rewrites the prompt when the prompt asked for something impossible); critic on 2–3 frames per scene change
  after the chain; at most 2 rounds, then list the rest for the user.

## Held-out test (2026-10-24, after the rules were tuned)

12 frames the rules never saw: 6 faults from the user's Slippery Road v1 review plus oil Chronophobia v3's second hand,
6 frames without physical faults. Segmentation geometry + Qwen whole-frame with crops and intent
(`pieces/_bench/eval/critic_heldout/`).

| Frame | Label | Result |
|---|---|---|
| sr_3175 | six fingers | caught (as a hand without an arm) |
| sr_3366 | upside-down face that isn't her | caught (reflection doesn't fit) |
| oil_4030 | second hand forming on her chest | caught |
| sr_4330 | white strips in her hair | flagged, but for a hand, not the strips |
| sr_1841 | legless | **missed** (no rule for missing legs) |
| sr_3445 | hand not attached | **missed** (the arm runs out of frame, which reads as plausible) |
| sr_614 | clean (snakes of light) | **false alarm**: "glowing lines floating" (intended, the intent didn't suppress it) |
| sr_2037 | clean (labelled) | flagged: a semi-transparent face floating over the street; looking again, a real cross-fade ghost the user didn't mention |
| sr_4796 | clean (TV) | **false alarm**: "same person twice" (her and her face on the TV) |
| sr_792, 3057, 4658 | clean | ok |

Caught 4 of 6 (one for the wrong reason); 2 real false alarms (plus one flag that turned out to be a real ghost).
Fixes indicated: a legs/lower-body rule for full-figure shots; screens, TVs and billboards masked and excluded from
"same person twice" like windows; intent wording that covers light effects; then a bigger held-out set before the
critic is allowed to trigger re-rolls on its own. Until then: use it to shortlist frames for review, not to act alone.
