# Building a world first: getting multi-shot consistency out of LTX-2.5

Researched 2026-10-26. The problem: every poptok shot is generated on its own from a keyframe still, so places and costumes get reinvented each shot. The question is what makes two or more LTX renders of the same place consistent, including leaving a place and coming back.

Provenance tags used below:
- **[V]** verified from an official model card, repo, doc or paper (read directly, mostly on 2026-10-26).
- **[L]** read from the local LTX-2 checkout at `morphexpts/ref/LTX-2` (commit e98f6d9, 2026-10-18).
- **[C]** community or vendor claim. Not independently checked.
- **[P]** my own proposal or inference. Untested.

---

## TL;DR

1. **Lightricks now ships the exact tool for this. [V]** `Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render` takes a grey clay or blocky playblast from Blender or Unreal (it carries the camera move and object placement) plus a stylised first frame (plus optional mid and last keyframes) that sets the look. It returns a shot that moves like the layout and looks like the keyframe. The weights are `ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors`; the repo is gated, so click "Agree and Access" on HF first. That gives "low-poly world + camera path + video model restyle" with official support.
2. **Identity across shots: the `Ingredients` (reference-sheet) IC-LoRA. [V]** `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` conditions on one composite sheet with panels for characters (face plus turnaround), props and the location. The model card is explicit that it is for keeping recurring characters, costumes and sets consistent.
3. **LTX has no 3D memory. Coming back to a place still needs one, and you build it outside the model. [P]** The world (the .blend) is the memory for geometry. For appearance, either reuse the keyframe from the earlier visit (same camera pose, so same look still), or project earlier LTX frames back onto the proxy mesh as textures and render that textured proxy as the next keyframe or guide. That is a hand-made version of GEN3C, Voyager and VMem-style 3D caches.

---

## 1. The landscape: 3D proxy or layout driving video diffusion

### 1a. "Low-poly world + camera path + restyle" (closest to what you want)

| Thing | What it does | Status |
|---|---|---|
| [LTX-2.5 IC-LoRA Layout-To-Render](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render) | Clay or blocking playblast in, plus a look keyframe, gives a finished shot. Keeps camera path and object placement, including the motion of simple animated shapes. Rank 128, 4000 steps, trained on 2.5 dev, inference on 2.5 distilled. Ships a ComfyUI graph `LTX-2.5_ICLoRA_Layout_To_Render_Two_Stage_Distilled.json` and a CLI recipe (below). | [V] |
| [fal LTX-2.3-3DREAL-LoRA](https://huggingface.co/fal/LTX-2.3-3DREAL-LoRA) (Lovis Odin + fal, June 2026) | IC-LoRA turning grey blockouts, viewport or game renders into photoreal video. Files `3DREAL-light`, `3DREAL-strong`, `3DREAL-strong-v2` (.safetensors), trigger word `3DREAL`. **Targets LTX-2.3**, not 2.5. Photoreal-biased, so a poor fit for stylised worlds. ([news](https://comfyui-wiki.com/en/news/2026-06-29-ltx-2-3-3dreal-lora-render-to-real)) | [V] card; quality [C] |
| [VideoFrom3D](https://arxiv.org/abs/2509.17985) (POSTECH, Sep 2025, [code](https://github.com/KIMGEONUNG/VideoFrom3D)) | Coarse geometry + camera trajectory + reference image gives a style-consistent scene video. An image model makes sparse cross-view-consistent anchor views; a video model in-betweens them with flow-based camera control and structural guidance. Same shape as "Layout-To-Render + keyframes at several frames". | [V] paper |
| [Diffusion as Shader (DaS)](https://arxiv.org/abs/2501.03847) | Conditions video diffusion on 3D tracking videos (coloured 3D points), so a mesh or animation becomes the control. Handles mesh-to-video, camera control, motion transfer. | [V] paper |
| [Playbook3D "Wan 2.2 Graybox to Rendered Sequence"](https://playbook3d.gumroad.com/l/vpjph) | Free ComfyUI graph: Blender graybox gives Wan 2.2 Fun Control plus a Flux Kontext depth LoRA. Claims character and environment consistency across a sequence without LoRA training. | [C] |
| [RunComfy "Blender to ComfyUI AI Renderer 2.0"](https://www.runcomfy.com/comfyui-workflows/blender-to-comfyui-ai-renderer-2-0-workflow-cinematic-video-output) | Blender depth, outline or pose passes into Wan 2.1 VACE 14B, with reference and start frames. | [C] |
| [Blockshot](https://www.stork.ai/en/blockshot), [Intangible](https://www.intangible.ai/answers/ai-storyboarding-previsualization) | SaaS: block the scene in 3D, set cameras, export a reference clip for an AI video model. invideo describes the generic recipe as low-poly Blender, stylise frame 1 with an image model, then animate in Kling or Seedance ([FAQ](https://invideo.io/faq/how-do-you-use-a-blender-blocking-render-as-a-camera/)). | [C] |

### 1b. Camera-controlled and 3D-informed video models (research)

| Paper | Mechanism | Relevance |
|---|---|---|
| [CameraCtrl](https://arxiv.org/abs/2404.02101) | Plücker-embedding camera adapter for T2V. | Camera only. No world memory. |
| [ReCamMaster](https://arxiv.org/abs/2503.11647) | Re-renders an existing video along a new camera path. | "Revisit the same scene from another angle", but source is a video. |
| [GEN3C](https://arxiv.org/abs/2503.03751) (NVIDIA) | Builds a **3D cache**: depth-unprojected point clouds from earlier frames, rendered along the new camera and fed as conditioning. | The canonical "world-consistent via explicit 3D cache" design. You can approximate it (section 3d). |
| [Uni3C](https://arxiv.org/abs/2504.14899) | Point-cloud renders for camera plus SMPL-X for human motion, unified in one control. | Shows camera geometry and acting can be controlled separately. |
| [Voyager (HunyuanWorld)](https://arxiv.org/abs/2506.04225) | Generates RGB+depth jointly, keeps a world cache, long-range explorable scenes. | Long-range, world-consistent. Heavy. |
| [WorldForge](https://arxiv.org/abs/2509.15130) | Training-free (inference-time) trajectory steering of pretrained video models using warped renders. | Proves warped-render guidance can be bolted onto an existing model. |
| [WorldWarp](https://arxiv.org/abs/2512.19678) | Propagates 3D geometry with asynchronous video diffusion. | Explicit-3D family. |
| [Lyra](https://arxiv.org/abs/2509.19296) (NVIDIA) | Self-distils a video model into 3D Gaussian splats. | Video becomes splat. Could turn a good LTX establishing shot into a reusable splat set. |

### 1c. Memory world models ("leave and come back")

| Paper or product | Memory type | Note |
|---|---|---|
| [WorldMem](https://arxiv.org/abs/2504.12369) | Memory bank of past frames + poses | Early revisit-consistency work. |
| [Context as Memory](https://arxiv.org/abs/2506.03141) | Retrieves past frames by FOV overlap and feeds them as context | Same idea as "use the earlier keyframe for this camera pose" (section 3d). |
| [VMem](https://arxiv.org/abs/2506.18903) | Surfel-indexed view memory | Retrieval keyed by 3D surface visibility. |
| [Video World Models with Long-term Spatial Memory](https://arxiv.org/abs/2506.05284) | Point-cloud spatial memory + episodic frames | Explicit 3D memory. |
| [AnchorWeave](https://arxiv.org/abs/2602.14941) | Retrieved *local* spatial memories | Argues one global reconstruction accumulates misalignment. |
| [I3DM](https://arxiv.org/abs/2603.23413), [WorldCrafter](https://arxiv.org/abs/2609.24984) | Implicit 3D-aware memory | WorldCrafter reports revisit LPIPS 0.255 vs 0.487 for Lyra 2.0 (per secondary summary; check the paper). |
| [OctWorld](https://arxiv.org/abs/2609.03919) | TSDF-fused octree map of generated frames | Explicit, persistent. |
| [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) (DeepMind) | Learned, real-time 720p/24fps | Consistent "for a few minutes", memory of about a minute when revisiting. Research preview. Not local. |
| [World Labs Marble](https://help.scenario.com/articles/4774055514-marble-by-world-labs-the-essentials) | Generates a Gaussian-splat world (.spz) and mesh export | A source of a navigable world. You could fly a camera through it and feed the render to Layout-To-Render or Union depth. [C] export details |

### 1d. Multi-shot without 3D

[Long Context Tuning](https://arxiv.org/abs/2503.10589), [HoloCine](https://arxiv.org/abs/2510.20822) and [Captain Cinema](https://arxiv.org/abs/2507.18634) get scene-level consistency by generating many shots jointly, or by conditioning on keyframe memory. They help with costume and identity, but give no geometric guarantee when you come back to a place. Commercial tools (Runway Gen-4 references, Luma scene references, Kling elements) work at the reference-image level. I found no evidence that any of them takes a 3D layout as input ([comparison, C](https://www.atlascloud.ai/it/blog/guides/kling-ai-vs-runway-vs-luma)).

**Takeaway:** research converges on two levers. (a) Geometry from a render of a persistent 3D thing (a point cloud, splat or mesh). (b) Appearance retrieved from earlier views of the same surfaces. Layout-To-Render gives you (a) inside LTX. You supply (b).

---

## 2. LTX-2.x control: exact files and versions

The repo README says "a LoRA only works with the model it was trained on" [L]. **However, the official LTX-2.5 ComfyUI workflows load the 2.3 Union-Control, Motion-Track and Ingredients files** on the 2.5 distilled transformer [V: `example_workflows/2.5/*.json`; the docs page [Union Control](https://docs.ltx.io/open-source-model/feature-guides/structural-control/union-control) says the 2.3 file is reused]. So some 2.3 IC-LoRAs are officially run on 2.5. Where a native 2.5 file exists, use it.

### LTX-2.5 (22B, Gemma-4 text encoder): native files on HF [V]

| Repo | File | Use |
|---|---|---|
| `Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render` | `ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors` | **Clay or blocking video plus look keyframe(s) gives the shot.** |
| `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients` | `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` | Reference sheet (characters, props, location) gives a consistent clip. |
| `Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details` | `ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors` | Detail and 4K refine, tiled fusion. |
| `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` | `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` | DFR detailing stage. |
| `Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate` | `ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors` | Removes people and reconstructs the background, so you can get an empty plate of a location from a shot. |
| Also | Restore, Deblur, Decompression, Colorization, Day-To-Night (`...-0.9`), Water-Simulation, Alpha-Gen, SDR-To-HDR, Refine; `LTX-2.5-22b-LoRA-Slow-Motion-Control`, `LTX-2.5-22b-LoRA-Cinemagraph` | Utility. |

### LTX-2.3 files used on 2.5 by the official workflows [V]

| Repo | File | Notes |
|---|---|---|
| `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | Canny + depth + pose in one LoRA. Reference at 0.5x output resolution. Paper: [AVControl](https://arxiv.org/abs/2603.24793). 2.5 workflow: `LTX-2.5_ICLoRA_Union_Control_Distilled.json`. |
| `Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control` | `ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors` | Coloured spline tracks give object motion. |
| `Lightricks/LTX-2.3-22b-IC-LoRA-Relight`, `-In-Outpainting`, `-HDR`, `-DubIt` | | |

### Older (LTX-2 19B), not for 2.5 [V]
`LTX-2-19b-IC-LoRA-{Depth,Canny,Pose}-Control`, `-Detailer`, `-Union-Control`. The camera LoRAs are `LTX-2-19b-LoRA-Camera-Control-{Dolly-In,Dolly-Out,Dolly-Left,Dolly-Right,Jib-Up,Jib-Down,Static}`. They only offer canned moves. With a 3D layout you don't need them, because the layout *is* the camera.

### Other conditioning in the local repo [L]
- **Keyframes at arbitrary frames:** `--image PATH FRAME_IDX STRENGTH [replace|guide]`. `guide` works at any pixel frame. `replace` only works at frame 0 or `1+8k`. Multiple images are allowed (`packages/ltx-pipelines/docs/conditioning.md`). Layout-To-Render puts the look still at **frame -1** (a reference token, not a displayed frame) plus optional guides at mid and last frames.
- **ICLoraPipeline** (`ltx_pipelines.ic_lora`) is distilled-only and two-stage. `--video-conditioning PATH STRENGTH` is repeatable (each path becomes its own reference item) and `--lora` is repeatable. `--stage-2-ic-lora` keeps the guide on in stage 2. `--conditioning-attention-mask MASK STRENGTH` sets per-region guide strength. `--chunked` / `--chunk-pixel-frames` / `--chunk-carry-frames` handle long clips.
- If several IC-LoRAs set different `reference_downscale_factor` values, the pipeline raises an error (`alpha_gen._reference_scale_factors`; the factors must match). Layout-To-Render and Ingredients both run at factor 1, so **stacking them is mechanically allowed. Whether the result is any good is [P], untested.**
- **Generated keyframe slots** (2.5 only): `--num-generated-keyframes N` relaxes temporal compression where motion is fast.
- **RetakePipeline** regenerates a time window inside an existing clip, so you can fix a bad stretch without losing continuity.
- **Trainer:** `packages/ltx-trainer` covers T2V/I2V LoRA training, V2V IC-LoRA (`v2v_ic_lora.yaml`) and composable `reference` + `first_frame` conditions. The local `train-model` skill in `ref/LTX-2/.claude/skills` drives it end to end.

### Layout-To-Render recipe (from the card) [V]
```
python -m ltx_pipelines.ic_lora \
  --prompt "<1-2 sentences describing the finished shot; never say clay/3D/Unreal>" \
  --video-conditioning clay.mp4 1.0 \
  --lora ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors \
  --stage-2-ic-lora \
  --image look_first.png -1 1.0 \
  --image look_mid.png 36 1.0 \
  --height 1088 --width 1920 --num-frames 81 --frame-rate 24 \
  --output-path out.mp4
```
- Stage 1 runs at 960x544 for 8 steps, stage 2 at 1920x1088 for 3 steps. CFG 1, euler_ancestral, LoRA strength 1.0.
- W and H must be divisible by 64. Frames snap to 8k+1. Prefer a 24 fps layout.
- **Never use the grey clay frame as the look image**, or the output comes out grey. Make the look image by stylising the clay frame with any image model (Z-Image, Qwen-Edit or Klein from your earlier tests).
- To hold the look across the whole shot, add a mid and a last keyframe.

### Ingredients recipe [V]
- Input is a static video of the reference sheet, looped, at least 121 frames, at output resolution (downscale 1).
- Trained bucket: 768x448, 121 frames, 24 fps. Anything else is out of distribution.
- Prompt in two labelled parts: `Reference sheet: …` / `Generated video: …`.
- Bigger panels carry over better. Each character needs a clean front close-up plus a turnaround. Black background, no text.
- ComfyUI graph: `LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json` (note: it loads the 2.3 file).

---

## 3. Practical pipeline (headless Linux)

### 3a. One world file, many cameras [P]
- Keep one `.blend` (or a procedural Python script that builds it) **per location**. Every shot is a camera path inside it, so geometry and layout are identical by construction when you leave and come back.
- Characters are low-poly stand-ins (capsules, mannequins) whose blocking is animated. Layout-To-Render keeps "the motion of simple shapes in a blocking animation" [V], so the stand-ins are where the acting is planned.
- Beat sync: drive camera keyframes from the poptok beat grid. The camera path in the layout is honoured frame by frame, so zoom and dolly accents land where you put them. Measure lag in ms as usual.

### 3b. Rendering the guide passes cheaply
- **Blender CLI:** `blender -b world.blend -P shot.py -- --cam path.json --out clay/` (Blender is not installed on gruntus. Get the official tarball from download.blender.org; no root needed).
  - *Clay pass for Layout-To-Render:* Workbench engine, solid shading, studio light, matcap or flat grey, 24 fps. Fast; on CPU it is seconds per frame at 1080p. On a headless GPU box, run EEVEE or Workbench via EGL (`--gpu-backend` / `-E BLENDER_WORKBENCH`).
  - *Depth for Union depth:* enable the Z/Mist pass, normalise with **one fixed near/far for the whole shot and the whole location** (the LTX docs ask for a consistent depth range across frames [V]). Invert to near = white.
  - *Normals and edges:* Normal pass, Freestyle lines, or Canny on the clay render.
  - Camera path: script it with `bpy` (keyframe `location` and `rotation_euler` from JSON) so poptok can generate it.
- **Lighter alternatives:** `pyrender` or `trimesh` with `PYOPENGL_PLATFORM=egl` for depth and normals of procedural meshes (pure Python, no Blender). three.js under headless Chromium (Puppeteer or Playwright) for browser-authored worlds. Gaussian-splat worlds (Marble `.spz`, Lyra output) render with gsplat or nerfstudio along the same camera JSON. Feed the RGB or depth to Union, or the RGB to Layout-To-Render.
- Size: 1920x1088 (Layout-To-Render). Union takes a reference at half resolution (ref0.5), and the workflow resizes automatically.

### 3c. Character and costume consistency
1. **Ingredients sheet** per character set and location (official, no training).
2. **Look keyframes**: produce them with the same image-model edit for every shot. Feed the previous shot's look still plus the new clay frame to an edit model ("restyle this layout like this reference") so the stills chain.
3. **Character LoRA (T2V) on LTX-2.5** with `ltx-trainer` and the `train-model` skill. Clean-Plate shows how cheap this can be: an official IC-LoRA trained on **26 clip pairs, rank 32, 3000 steps** [V]. An identity or style LoRA stacks with an IC-LoRA. The docs say standard LoRA plus IC-LoRA can be combined; start style LoRAs at 0.5-0.8 [V].
4. **Train your own "layout + sheet" IC-LoRA** [P]: pairs of (clay render of a scene, real or LTX clip). Only worth it if stacking Layout-To-Render + Ingredients fails.

### 3d. Coming back to a place: appearance memory [P]
Geometry is solved by the shared .blend. Appearance is not, because each LTX run re-hallucinates textures. Options, from cheapest up:
1. **Pose-keyed keyframe reuse** (Context-as-Memory style): store `(location, camera pose, look still, best LTX frame)` per shot. A new shot at a similar pose reuses that look still, or the earlier shot's actual output frame, as its `-1` look image plus mid guides.
2. **Projection-painted proxy** (GEN3C or VMem style, done by hand): after a shot renders, project its frames back onto the low-poly mesh from the known camera (Blender "project from view" / UV-project bake, or a small numpy rasteriser using the depth you already rendered). The proxy gradually acquires the world's textures. Render the *textured* proxy for the next shot, and use it as the look keyframe(s). Possibly also use it as the layout video; Layout-To-Render was trained on clay, so whether a textured layout helps or confuses it is untested.
3. **Clean plate** a good establishing shot with `Clean-Plate` IC-LoRA, then use those plates as the location panel on the Ingredients sheet.
4. Longer term: lift a good establishing shot into splats (Lyra-style, or Marble from stills) and fly the camera through that instead of the grey box.

---

## 4. Known limitations

| Issue | Evidence |
|---|---|
| **Acting and motion come from the layout.** Structural guides fix the camera and the gross motion. If the stand-in doesn't move, the character won't either. Layout-To-Render keeps blocking-shape motion; body acting beyond that is up to the model. | [V] card wording; acting quality [P] |
| Too rigid a guide flattens the look | Docs: IC-LoRA strength 1.0 by default; **0.5-0.8 for a softer guide and more creative freedom**, about 0.6 for soft depth. `--conditioning-attention-mask` can relax it per region, for example only lock the set while the character area stays at 0.5. [V] |
| Depth fails on flat compositions; Canny fails on soft or organic content (fur, foliage, clouds); Pose needs visible bodies | [V] Union-Control docs |
| Flicker | Docs: keep the depth range consistent and smooth; raise steps (40-50) on non-distilled paths if flicker appears. Synthetic Blender depth is temporally stable by construction, unlike estimated depth. [V]/[P] |
| Clip length | Frames snap to 8k+1. Layout-To-Render examples run 65-81 frames at 24 fps; Ingredients was trained at 121 frames. ICLoraPipeline has `--chunked` windows with carry frames, and Refine-Details runs 97-frame windows up to about 240 frames as one extent. Expect about 3-5 s per guided shot natively; longer needs chunking, with drift risk at seams. [V]/[P] |
| Resolution buckets | Ingredients: 768x448 trained (out of distribution elsewhere). Layout-To-Render: 1920x1088 known good, /64. [V] |
| **Stylisation (stained glass, felted wool)** | Layout-To-Render takes "palette, lighting, and materials" from the keyframe, so a stained-glass or felt look still should carry, and its examples change season and material over the same layout. The examples shown are photoreal, though, and no stylised examples were found. The fal 3DREAL LoRA is photoreal-biased and a poor fit. Strongly graphic styles may fight a strict depth guide; use a lower strength or Layout-To-Render rather than depth. **Untested, so A/B it.** [V]/[P] |
| LoRA version confusion | README says LoRAs are model-specific, but the official 2.5 graphs load 2.3 Union, Motion-Track and Ingredients files. Prefer native `ltx-2.5-*` files where they exist. [V]/[L] |
| Stacking two IC-LoRAs (layout + sheet) | Allowed by the code (both downscale 1). No official example. Token cost goes up with each reference. [L]/[P] |
| Gated downloads | The Layout-To-Render raw README returned "not in the authorized list" with the current HF token, while `resolve/` served the card. Accept the gate on the HF page before downloading weights. [V] |

---

## 5. Top 3 approaches for the local LTX-2.5 pipeline (ranked)

1. **Layout-To-Render on a per-location Blender world.** Procedural low-poly .blend per location, cameras scripted from the beat grid, Workbench clay render in headless Blender. For each shot, stylise its first clay frame with the image model (chain from the previous look still for consistency), then run `ic_lora` with the Layout-To-Render LoRA, `-1` look still, plus mid and last guides. Official, built for exactly this, and coming back to a place reproduces the geometry exactly.
2. **Add identity: Ingredients sheet, then a character LoRA if needed.** Build one reference sheet per song (characters, costumes, props, location plate). Test stacking it with Layout-To-Render. Add a **per-video LTX-2.5 LoRA** for the cast, costumes and props (rank 64-128, trained on bootstrapped stills plus short clips; see section 6), stacked with Layout-To-Render at 0.5-0.8. Make the look stills with an image-side LoRA or Qwen-Edit-2511 multi-reference edits.
3. **Appearance memory for revisits.** Store look stills and output frames keyed by camera pose. Next, project-paint LTX frames onto the proxy so the world accumulates texture, and render the textured proxy as the keyframe for any return visit. Union-Control depth (2.3 file on 2.5) is the fallback guide when you want looser structure than Layout-To-Render, or for stylised looks that clay locking fights.

### First experiment to run [P]
One location, two shots: A enters the room, B returns 20 s later from a different angle. Use the same .blend. Compare three set-ups: (i) independent look stills, (ii) B reuses A's look still, (iii) B's look still is rendered from the A-textured proxy. Judge by consecutive frames and by side-by-side crops of the same wall or prop.

---

## 6. A custom LoRA per video as a world-building tool

Added at the coordinator's request. Execution time is not a constraint. The same provenance tags apply: [V] verified, [L] local repo, [C] community or vendor, [P] my inference.

### 6a. What character, costume, location, prop and style LoRAs give you on LTX-2.x and Wan 2.x

**Verified, from the local trainer and official cards:**
- The `ltx-trainer` supports **videos, single images, or a mix**. Images use frame count 1 in the bucket (for example `960x544x1`). Mixed datasets need `batch_size: 1` and multiple buckets (`"960x544x1;960x544x49"`). The officially safer option is **two LoRAs, one trained on stills and one on video, stacked at inference** [L `docs/dataset-preparation.md`].
- Hardware: **80 GB+ for the standard config** (`t2v_lora.yaml`: rank 32, alpha 32, LR 1e-4, 2000 steps, batch 1). A 32 GB low-VRAM config (`t2v_lora_low_vram.yaml`) adds INT8 quantisation, an 8-bit optimiser and rank 16 [L README, configs]. A 128 GB Spark is in the 80 GB+ tier, so run the standard config with no quantisation.
- Rank guidance from the bundled `train-model` skill [L `references/mode-selector.md`]:
  - Single character, object or style: **32-64**.
  - **Multi-character world, or a dense multi-concept series: 96-128.** That is the "one LoRA per video holding the cast, costumes and props" case.
  - Motion or camera: 8-16.
  - Structural IC-LoRA: 16-32.
- Official LoRAs trained with this trainer show how small datasets can be:
  - Clean-Plate IC-LoRA: **26 clip pairs, rank 32, 3000 steps** [V].
  - Ingredients: rank 128, 12k steps, 8 GPUs, proprietary data [V].
  - Layout-To-Render: rank 128, 4000 steps [V].
- LTX's docs recommend stacking a standard LoRA with an IC-LoRA: **IC-LoRA at 1.0 plus the effect or style LoRA at 0.5-0.8, with total strength under about 2.0** [V [IC-LoRA docs](https://docs.ltx.io/open-source-model/usage-guides/ic-lo-ra), [Union docs](https://docs.ltx.io/open-source-model/feature-guides/structural-control/union-control)]. LTX's temporal-consistency guide names a **trained character LoRA** (plus shorter clips and cutting between shots) as the fix for identity drift [V-ish, [LTX blog](https://ltx.io/blog/temporal-consistency-in-ai-video)].

**Community evidence (not verified):**
- *LTX-2 character LoRA (Oxen.ai, "Yoda")* [C, [post](https://ghost.oxen.ai/how-to-train-a-ltx-2-character-lora-with-oxen-ai/)]:
  - Data and settings: 10-30 video clips of about 5 s, 960x576 at 24 fps, **rank 16, LR 2e-4, 1000 steps, about $10 of cloud GPU**.
  - Text-only and plain I2V baselines both failed. Plain I2V looked right but didn't move or sound right.
  - After 1000 steps the look and voice were "starting to" emerge. Audio learns at a different rate from video.
- *LTX-2.3 trainer guides* [C, [RunComfy](https://www.runcomfy.com/trainer/ai-toolkit/ltx-2-lora-training), [ltxworkflow](https://ltxworkflow.com/resources/tutorials/ltx-23-lora-training-guide-style-motion-ic-lora)]:
  - **2000-3000 steps** as the first pass; 4000-6000 for hard concepts.
  - Rank 32, going to 64 only if underfitting. LR 1e-4.
  - **20-50 images** for a character or style; 80-120 for very specific subjects.
  - 512 px is the floor for identity work.
- *Hotstar engineering blog* (moved from Wan 2.1 to LTX-2) [C, via search snippet; the page returned 403]:
  - Character features appear at about **1k-3k steps** and become consistent at **3k-6k**.
  - **Beyond about 9k steps the model memorises training poses**, so output gets stiff.
- *Wan 2.2* [C, [musubi-tuner discussion #455](https://github.com/kohya-ss/musubi-tuner/discussions/455), [settings guide](https://wavespeed.ai/blog/fr/posts/blog-wan-2-2-lora-training-settings/), [tutorial](https://www.stablediffusiontutorials.com/2025/10/wan2.2-lora-training.html)]:
  - Rank 16-32, 3000-5000 steps, 10-15+ stills for a character.
  - Separate high-noise and low-noise expert LoRAs (for example 1800 low-noise + 600 high-noise steps) held body and face shape slightly better.
  - **LoRAs trained on images only tend to suppress motion.** The fix is to animate the stills with an I2V model and train on short clips (about 33 frames).
- *Drift.* Practitioners report that identity holds while motion is small and drifts once the character moves a lot, even with a LoRA ([LTX-2 issue #255](https://github.com/Lightricks/LTX-2/issues/255); [RunComfy I2V drift guide](https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-i2v-character-consistency-lora)) [C].
- *Training time on a Spark.* I found no published LTX or Wan video-LoRA timing for the Spark. One forum user reports AI-Toolkit doing 3000 steps in 4 h 33 min (model unspecified) ([NVIDIA forum](https://forums.developer.nvidia.com/t/ostris-ai-toolkit-on-dgx-spark/355277)) [C].
  - **Estimate [P]:** from our own inference figures, the Spark denoises LTX-2.5 stage 1 at about 1.0 s per step on a 65-frame half-resolution clip (`poptok/docs/hardware-performance.md`). A training step costs very roughly 3-4x a forward pass (backward plus checkpointing).
  - Image-only (`960x544x1`, rank 32-64): a few seconds per step, so **about 1.5-3 h for 2-3k steps**.
  - Short video clips (`960x544x49`): perhaps 10-20 s per step, so **about 6-15 h for 2-3k steps**.
  - Measure with the skill's 50-step dry run before trusting these numbers.

### 6b. Image-model LoRA for the stills vs a video LoRA on LTX

| | Keyframe-image LoRA (Qwen-Image / Qwen-Image-Edit-2511, Z-Image, Flux) | Video LoRA on LTX-2.5 |
|---|---|---|
| What it fixes | The **first frame**, plus any mid and last guides you also generate. | Every frame. Identity is in the weights. |
| Where it breaks | After about 1-3 s of motion LTX re-interprets the face and costume. Plain I2V drift is the reported failure [C]. It only holds where a guide frame exists. | Can stiffen motion if trained on stills only or overtrained [C]. A location LoRA doesn't hold geometry (6d). |
| Cost | Cheap. Our Qwen-Edit-2511 runs at about 24 s per still on a Spark [L hardware-performance]. Image-LoRA training is well understood. | Hours per video [P]. Needs a curated dataset. |
| Combines with Layout-To-Render | Yes: these are the `-1` / mid / last look stills. | Yes: stack at 0.5-0.8 on top of the IC-LoRA [V guidance]. Untested on Layout-To-Render specifically. |

**Assessment [P]:** use **both**. They are not competing:
- An image LoRA (or a multi-reference edit model) makes the look stills consistent. That is the cheap, high-leverage step, because Layout-To-Render takes "palette, lighting, materials" from those stills.
- A video LoRA on LTX carries identity through the frames *between* the guides, and through turnarounds the stills never showed.
- I found no controlled head-to-head comparison (the searches turned up nothing). Run one: the same layout shot with (i) stills only and (ii) stills plus the LTX character LoRA. Measure face and garment similarity per frame.

### 6c. Bootstrapping a consistent training set from nothing

Community recipes [C], plus how they fit poptok [P]:
1. **Multi-reference edit model as the generator.**
   - Qwen-Image-Edit-2511 officially takes multiple reference images and claims better character consistency and less drift ([model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)) [V for the claim].
   - Community workflows turn one reference into a captioned character dataset across many described scenes ([Floyo "Single Image to Character Dataset"](https://www.floyo.ai/workflows/qwen-2511-single-image-to-character--z4vpyev94iuc)) [C].
   - We already use this for Amber from her photo (about 24 s per still, 1 reference; about 61 s with 3).
2. **3D blockout renders, restyled.** Render the low-poly character and set from many cameras (turnarounds, close-ups, the actual shot cameras) in Blender. Restyle each with the edit model, passing *the same* hero reference each time. Because the views come from one 3D source, viewpoint coverage is controlled. This is also how VideoFrom3D gets cross-view-consistent anchor views (sparse anchor views from an image model under appearance guidance) [V paper].
3. **Turn into video.** Animate the curated stills with LTX I2V (or Layout-To-Render on small moves) into 33-49-frame clips, so the LoRA learns motion and not just poses. This is the Wan community's fix for image-only motion loss [C].
4. **Curate automatically.**
   - Faces: ArcFace or InsightFace cosine similarity to the hero face (keep above a threshold).
   - Garments and props: CLIP, DINOv2 or SigLIP similarity on crops from segmentation (SAM). Drop outliers.
   - Run a VLM caption pass to tag view, shot size and costume state.
   - All of this is standard practice, but I found no single published recipe that validates it end to end [P].
5. **Iterate once.** Train v1, generate more with v1, re-curate by similarity, train v2. The risk is that drift compounds: each round can bake in the previous round's errors. Mitigate by always mixing in the original hero references and by keeping the similarity threshold fixed against round-0 references [P]. No source tested this loop (the searches found nothing).
6. **Reference-sheet output doubles as Ingredients input.** The curated turnaround panels are exactly what the Ingredients IC-LoRA sheet wants [V card].

### 6d. What a LoRA can't hold, what a 3D proxy gives, and combining them

- **A LoRA holds appearance statistics**: faces, garments, materials, palette, style, recurring props. It **does not hold metric spatial layout**: which door is left of the window, how far the altar is from the camera, what's behind you when you turn round. There is no camera-conditioned 3D representation in it.
  - No source tested a "location LoRA" for multi-view layout; the search found none [C/absence].
  - The nearest thing is Cseti's **CrossView IC-LoRA** for LTX-2.3 (rank 16, 294 UE5 camera pairs from SynCamVideo, 768x768x81) ([HF](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Prompt), [news](https://comfyui-wiki.com/en/news/2026-07-11-cseti-crossview-ic-lora)). Its authors call it a proof of concept: large viewpoint changes have to be chained in small steps. That shows how weak learned viewpoint control still is [V card / C quality].
  - The memory-world-model papers in section 1c exist because pure learned memory fails at revisits; they all add explicit or implicit 3D retrieval [V].
- **A 3D proxy holds geometry, camera and blocking exactly**, by construction. It holds no appearance. That is why Layout-To-Render needs look keyframes.
- **Combining them is the reported best practice for the *pair* of problems:**
  - Officially: IC-LoRA (structure) plus a standard LoRA (look) at 0.5-0.8 [V docs].
  - Community Blender graybox pipelines pair a structure guide with reference or LoRA conditioning: Playbook3D (Wan 2.2 Fun Control plus a Kontext depth LoRA, "consistency across sequences without LoRA training") and RunComfy Blender-to-VACE (depth, outline or pose plus reference frames) [C].
  - fal's 3DREAL strong-v2 advertises "multi-scene coherence" from blockouts [C].
  - I found **no published head-to-head** of "3D guide plus per-video LoRA" against either alone.
- **Recommended split [P]:**

| Need | Carrier |
|---|---|
| Place geometry, camera path, blocking, coming back to a place | Blender world, then Layout-To-Render (or Union depth) |
| Look of the place (materials, palette) | Look stills from the image LoRA or edit model, chained. Plus a projection-painted proxy for revisits (3d). Optionally a small "set-dressing" style LoRA. |
| Characters, costumes, props | Per-video LTX LoRA at rank 64-128 (cast plus props in one), trained on bootstrapped stills plus short clips. Plus the Ingredients sheet. |
| Global style (stained glass, felt) | Style LoRA (image side, and optionally LTX side at 0.5-0.8). Train on restyled renders of *this* world so the style LoRA doesn't import foreign geometry. |

### 6e. Per-video LoRA plan for one R.Amber song [P]
1. Build the world .blend(s) and shot cameras. Render clay turnarounds of the stand-in characters plus 30-60 set views.
2. Restyle them with Qwen-Edit-2511 using the hero refs (Amber photo, costume refs, style ref). Curate by ArcFace and DINOv2 similarity, leaving about 40-80 stills.
3. Animate about 20 of them into 49-frame LTX clips.
4. Train two LTX-2.5 LoRAs on a Spark: stills (`960x544x1`, rank 64, about 2.5k steps, about 2-3 h est.) and clips (`960x544x49`, rank 32, about 2k steps, overnight est.). Validate every 250 steps, and keep an early checkpoint because of the stiffness risk past about 6-9k steps [C].
5. Render shots with Layout-To-Render at 1.0, character LoRA(s) at 0.5-0.8, and look stills at -1, mid and last.
6. A/B against the same shots without the LoRA, judging by consecutive frames.
