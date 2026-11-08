# 3D ground truth for AI video: the felt-room experiments (AI+World)

**Dates:** 26–28 October 2026. **Written:** 8 November 2026, from the session records, `docs/log.md`, the launchers and
the review sheets cited below.
**Machines:** two DGX Sparks (thing1, thing2) for the AI models; gruntus (RTX 4090) for Blender.

## Conclusion

**A 3D scene as ground truth works: LTX-2.5 follows it frame by frame, and the same 3D room gives the same room from
different camera moves. A world LoRA trained on that one room gives it a consistent look. Compositing the singer's
identity onto it doesn't work reliably yet.** Her face can be made right on a still (face-match score up to 0.89), but
the video model doesn't keep it. Within a few frames it redraws her as a similar-looking woman who isn't her (about 0.25–0.33).

## 1. What problem were we trying to solve?

The prompt-only pipeline ("AI") draws each shot's keyframe separately, so:

1. **The place changes between shots.** Two shots of "the same room" come out as two different rooms, because each is
   drawn fresh from words.
2. **The camera isn't controlled.** Camera moves are only described in words; LTX interprets them loosely, and pinning
   several keyframes produced walk-and-rewind motion as it blended between unrelated stills.
3. **The singer's identity drifts** from shot to shot, and within a shot.

The question: if a 3D model of the place (and of her body) fixes geometry, camera and motion, can an AI video model
supply only the *look*, so that every shot is the same place, the camera does exactly what was planned, and she stays
her? The user's prompt for the idea was Matrix-Game 4.0 (matrix-game-v4-0.github.io): a deliberately plain 3D proxy
plus a renderer fine-tuned on that one world.

## 2. Hypotheses

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | LTX-2.5 with Lightricks' Layout-To-Render add-on follows a 3D render's geometry and camera exactly, with no reinvention or rewind. | **Supported** (rounds 1, 4b, 5, control) |
| H2 | Two camera moves through the same 3D room give the same room on screen. | **Supported**, once nothing contradicts the layout (round 4b on; the LoRA) |
| H3 | The look (a "felted wool" universe) can be supplied without training: by pinned styled stills, by painting the look onto the 3D geometry, or by a reference sheet (Ingredients add-on). | **Rejected.** Pinned stills fight the layout; painted textures became a patchwork; reference sheets have no effect on the world |
| H4 | A LoRA trained on (plain 3D render → styled render) pairs of that one room teaches LTX the room's look, consistently from any camera. | **Supported** on the two test shots |
| H5 | A human figure in the layout (motion-captured 3D body) becomes a person, and once dressed like her, reads as her figure. | **Partly supported:** a person and her figure (hair, clothes, pose, motion), not her face |
| H6 | Her identity can be put into the video by compositing: her face edited into the first frame (and later frames), plus a prompt that describes her. | **Not supported yet:** holds at frame 0, lost within ~8–32 frames |
| H7 | (Untested explanation for H6) Her face is too small at LTX's half-resolution first pass, and later pinned frames are only a soft guide. | **Open:** the test (D28) never ran |

## 3. Technology and data, in the order they were used

### The pipeline

| Step | What | Technology | Data |
|---|---|---|---|
| 1 | **Build the room.** A ruined apartment room from Noble Hunt (a window, a door, a table, crates, rubble, a bench), as a low-poly blockout, plus two camera moves: **A** enters by the door and pushes toward the window where she sits; **B** starts by the window and pans back toward the door. 1920×1088, 81 frames, 24 fps. | Blender 5.2 in a container, `pipeline/world/room_blockout.py` | None. The room is **modelled by a script** (boxes and cylinders placed by hand-written code), not procedurally generated: there is no randomness or generator; the same script always builds the same room |
| 2 | **Add her body.** A MakeHuman body, dressed (long blond hair, olive uniform, flat colours per part), rigged, and driven by a motion-capture clip, seated on the floor under the window. | Blender + MPFB 2 (MakeHuman), `pipeline/world/figure.py` | MakeHuman community asset packs (hair, clothes); CMU motion capture clip 82_05 (sitting), Motionbuilder-friendly BVH release |
| 3 | **Render the layout.** First as grey clay; later (the Matrix-Game proxy) as flat colours, one per kind of thing, with black outlines, a checkered floor, a depth pass and a caption written from the scene. | Blender, `pipeline/world/flat_layout.py` | The scene from steps 1–2 |
| 4 | **Make styled training targets.** Each clay view restyled on its own into the felt world. | Qwen-Image-Edit 2511 (Lightning, 8 steps), `pipeline/world/style_views.py` | 80 "shot-like" views (looking across the room, at least 4 kinds of thing in frame); **58 kept**, 22 rejected by eye where Qwen invented a window, door or sofa (`pieces/_bench/world/flat2/views/curation.json`) |
| 5 | **Train the world LoRA.** A V2V IC-LoRA: the flat layout is the control, the styled view is the target. Initialised from the official Layout-To-Render IC-LoRA (rank 128). | LTX-2 trainer (`ltx2-trainer` container), LTX-2.5 22B **dev** model, Gemma 4 12B text encoder, on thing2 | The 58 room pairs, stills at 960×544, captions from the scene, no audio. Room only: the 40 views of her were left out (user decision 26 Oct) because the restyle made her brunette |
| 6 | **Render shots.** The shot's flat layout as the control video, the world LoRA, a styled first frame (the model card's recipe), and a text prompt. | LTX-2.5 22B **distilled** + world LoRA (checkpoint 1000), ~5 min per 81-frame clip on a Spark | Shots A and B |
| 7 | **Put her face in.** Detect her face, edit only a crop with her photo as the reference, paste back a feathered oval, keep the best of N tries by face-match score. | `pipeline/face_pass.py`: YuNet (face detector), Qwen-Image-Edit 2511, SFace (face matcher) | Her photo, `pieces/artists/r.amber.jpeg` |

### LoRA training details

From `pieces/_bench/train/v2v-ic-lora-noble-felt-room-20261026/plan.md` and the training log:

| | |
|---|---|
| Mode | V2V IC-LoRA; conditions: reference (the flat layout, always), first frame (probability 0.2) |
| Starting point | `ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors` (the trained LoRA replaces it at inference) |
| Data | 58 (flat layout, styled view) pairs, 960×544 stills |
| Settings | AdamW, bf16, no quantisation, batch 1, learning rate 1e-4 (half the default, as it fine-tunes an existing LoRA), rank/alpha 128/128 |
| Run | 1,500 steps in 197 min on thing2, ~4 s per step, peak 50 GB of 128; checkpoints every 250 steps |
| Checkpoint picked | **1000**: the validation clips at steps 500, 1000 and 1500 looked much the same (`pieces/_bench/review/2026-10-27/world-lora-steps-{A,B}.jpg`) |

### Also tried

- **Lightricks' Ingredients IC-LoRA:** a reference sheet with her face, body, uniform and a felt-room panel.
- **Painting the look onto the 3D geometry:** styled views projected back onto the mesh as textures
  (`project_texture.py`, `paint_world.py`, `paint_loop.sh`).
- **Wan 2.1 VACE 14B** as a second renderer on the same layouts.
- **TRELLIS.2-4B** (with Meta's DINOv3) to build her as a textured 3D mesh from one full-length image
  (`pipeline/world/her_mesh.py`). It produced a clean mesh (`pieces/_bench/review/2026-10-28/her-mesh-2-turntable.png`),
  but the mesh never went through LTX.

## 4. The experiments, in order

| # | Experiment | Hypothesis | Result |
|---|---|---|---|
| 1 | **Round 1:** clay layout + Layout-To-Render, with styled look frames pinned | H1, H3 | The camera follows the layout exactly: A's push and B's pan play as one continuous move, and the room holds its shape. But she became a small green toy soldier (A), a felt dummy appeared beside her (the stand-in mannequin, styled literally), and the walls were stone in B and plaster in A (each look frame was styled on its own) |
| 2 | **Rounds 2–3:** stand-in removed from the look frames; B's pan cut from ~90° to ~45°; Ingredients reference sheet | H3 | B failed again: its look frame contradicted its layout |
| 3 | **Round 4:** look painted onto the 3D room (textures projected from two views) | H2, H3 | B failed because of a pinned photographic start still. The projected texture streaked and warped away from its two views |
| 4 | **Round 4b:** the textured 3D render alone, no pinned stills | H1, H2 | **LTX follows the 3D render frame by frame; A and B show the same room.** But LTX adds almost nothing: the mannequin stays a mannequin, so the 3D scene has to carry the look |
| 5 | **Round 5:** room painted from many views, plus her animated figure (mocap); LTX with and without Ingredients; Wan VACE on the same layout | H3, H5 | Same felt room in A and B. With the sheet, her figure became a blond woman in uniform in A (small in frame); in B (large in frame) she stayed a grey figure. The floor was still a patchwork. VACE: no figure at all, just a flat green silhouette (user review), at lower resolution. The user judged the painted room "a comically incoherent mess" |
| 6 | **Control:** flat-colour layout, official Layout-To-Render, no training, with and without a reference sheet | H3 | **The room fails:** LTX copies the flat colours literally (green walls, brown floor), and the sheet changes nothing. So a world LoRA is needed. Her silhouette became a person when large, but not her (dark hair, read as a man in goggles) |
| 7 | **Dressed figure** (hair, uniform in flat colours) | H5 | LTX keeps her as a fair-haired woman in olive clothes, sitting under the window, moving with the mocap, from front and back. A slightly CG woman, not her face. The uniform reference sheet turned her dark-haired, so it was dropped |
| 8 | **Model card's recipe:** one styled first frame + layout | H2 | A whole shot in the felt world, consistent from first frame to last. Consistency *between* shots still open |
| 9 | **World LoRA training** (58 room pairs) | H4 | From step 500, both shots are in the trained world from text and layout alone: textured green felt walls, broken tiles, low hard light. **A and B match each other for the first time.** The layout is followed. Problems: A's window goes pitch black; more rubble in A than B; she is a small dark generic figure (she wasn't in the training data) |
| 10 | **D3/D9:** LoRA checkpoint 1000, distilled model, with and without a first frame | H4 | With the first frame, the LoRA tracks the base model, and B's room stays dark and consistent. Without it, the LoRA alone keeps the felt look and her pose, lit like the layout. **But the first frame's face wasn't her:** 0.42 (B) and 0.13 (A), and nothing had checked it before the run |
| 11 | **D11:** face pass with 12 tries, gate at 0.5 | H6 | Rejected both first frames (best 0.36 and 0.42): the restyle had described her with dark hair and face paint, and the face pass keeps hair and face marks |
| 12 | **D12:** her described from her photo; first frames gated at 0.5 on faces ≥ 80 px | H6 | **B: her at frame 0** (median 0.50; the user chose frame 0 as the target look), **lost once she turns away** (only 2 of 11 sampled frames show a face). **A: never her** (about 0.13): her face is ~38 px wide, too small to judge or to keep, so shot A was dropped (D18) |
| 13 | **D19 step 1:** the clip prompt rewritten to describe her from her photo (it had still said "long dark-blond hair") | H6 | Blonde and facing the camera to about frame 32 (was 8), but not her: 0.61 at frame 0, then −0.05 to 0.31 |
| 14 | **D19 step 2:** frames 16 and 32 of the D19 clip face-passed (0.31 → **0.89** and **0.87**) and pinned as extra conditioning | H6 | In the video those frames scored **0.30 and 0.26**; mean 0.33 (D19: 0.25). Same motion, same blonde woman; her face never snaps to the pinned face. **This is the latest AI+World video:** `pieces/_bench/review/2026-10-28/world-lora-1000-d19s2-B.mp4` |

## 5. How each experiment was evaluated

| Method | Used for | What it is | Limits |
|---|---|---|---|
| **Labelled review sheets** | Every round | Frames laid out in rows (one per variant) and columns (frame 1, 21, 41, 61, 81), each panel labelled (e.g. 3a), and the mp4s beside them; for drift, consecutive frames (e.g. 16–47) | Sparse frames can hide frame-to-frame changes, hence the consecutive-frame sheets |
| **Claude's evaluation by eye** | Every sheet; curating the training pairs | Claude looked at the sheets and wrote a verdict per row ("room: good", "her: fails"); rejected 22 of 80 training pairs where the target didn't match its layout | Not independent, and it missed things the user caught: the D3/D9 first frame wasn't her (Claude had reported it without checking the face); the training pairs where Qwen drew the same room whatever the view; how incoherent the painted room was. After the D3/D9 miss, the face check was made a numeric pass/fail gate (D11) |
| **The user's review** | Key decisions | Verdicts that redirected the work: "a comically incoherent mess" (painted room), "no figure at all" (VACE), "someone, but it's not Amber" (D3/D9 first frame), frame 0 of D12's shot B as the target look | — |
| **Face-match score** | D11 onward | SFace cosine similarity between the largest detected face (YuNet) and her photo, every 8 frames of each clip, printed on the sheets. Same-person threshold 0.363; gate 0.5 (her own photos score 0.55–0.92); faces narrower than 80 px reported but not judged | Measures the face only (not hair, body or "is it her" overall); unreliable on small or turned faces |
| **Layout check by eye** | H1, H2 | Does each frame match the 3D render's geometry and camera (objects in place, the camera path followed, nothing invented)? | Not measured numerically |
| **Trainer validation clips** | H4 | Shots A and B from text and layout alone at steps 0, 500, 1000 and 1500 | No first frame, full (non-distilled) model: not the inference setup, hence the D3/D9 test |

## 6. Conclusions

1. **3D ground truth works for the scene.** LTX-2.5 with Layout-To-Render reproduces a 3D render's geometry and camera
   frame by frame: no reinvention, no walk-and-rewind, and the camera does exactly what was planned in Blender.
2. **LTX copies the layout; it doesn't add a look.** Flat colours stay flat colours and a mannequin stays a mannequin.
   The look has to come from the 3D scene or from training.
3. **A per-world LoRA supplies the look consistently.** 58 pairs and ~3 hours of training on one Spark gave one felt
   room seen the same way by two different cameras: the first time two shots matched.
4. **Untrained ways of supplying the look failed:** pinned separately styled stills (each is a different room),
   textures painted onto the geometry (patchwork), reference sheets (no effect on the world).
5. **A motion-captured, dressed 3D body works as her figure** (hair, clothes, pose, motion), but not as her face.
6. **Avatar compositing doesn't work reliably yet.** Her face can be made right on a still (0.87–0.89), and frame 0 holds
   it, but the video redraws her face within ~8–32 frames, even at frames pinned to her face, and never at all
   in a wide shot.

## 7. Open, untested

- **D28: is face size the cause?** The same clip from a close shot with her face ≥ 200 px wide (~100 px at LTX's
  half-resolution first pass). If identity then holds, the rule is close shots for identity; if not, pinned frames are
  simply too loose a guide.
- **Her 3D avatar through LTX.** The TRELLIS.2 mesh of her (`her-mesh-2.glb`) was built, but never rendered in the room
  and passed to LTX.
- **Training her into the LoRA**, with targets fixed so they don't make her brunette.
- **Across a cut.** Consistency was tested within shots and between two shots of one room, not across a real edit of
  many shots and places.
- **Many-view painting** to fix the floor patchwork; superseded by the LoRA approach rather than finished.

## 8. Files

| What | Where (under `~/sambashare/expts/poptok/`) |
|---|---|
| Latest video (D19 step 2) | `pieces/_bench/review/2026-10-28/world-lora-1000-d19s2-B.mp4` |
| Identity sheets and frame strips | `pieces/_bench/review/2026-10-28/world-lora-1000-her-{A,B}.jpg`, `d19-B-frames-16-47.jpg`, `d19s2-B-keys.jpg`, `d19s2-B-frames-16-47.jpg` |
| LoRA test (checkpoint 1000) | `pieces/_bench/review/2026-10-28/world-lora-1000-{A,B}.jpg` and mp4s |
| LoRA training steps | `pieces/_bench/review/2026-10-27/world-lora-steps-{A,B}.jpg` |
| Rounds 4–5, control, dressed figure, training pairs | `pieces/_bench/review/2026-10-26/world-round4*`, `world-round5-*`, `world-control-*`, `her-dressed-*`, `world-lora-pairs*.jpg`, `world-lora-her-{1,2}.jpg` |
| Training run | `pieces/_bench/train/v2v-ic-lora-noble-felt-room-20261026/` (`plan.md`, `config.yaml`, `dataset/`, `outputs/checkpoints/`) |
| Code | `pipeline/world/{room_blockout,figure,flat_layout,style_views,her_mesh,project_texture,paint_world}.py`, `world_test*.sh`, `pipeline/face_pass.py` |
| Launchers | `pipeline/launchers/{world-lora-ckpt1000,d12-identity,d19-identity,d19-step2}.sh`, `pieces/_logs/world-*.sh`, `lora-train.sh` |
| Log | `docs/log.md`, 26 October 06:05, 06:30, 11:25 |
