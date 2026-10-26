# Photo to animatable 3D avatar (R.Amber), checked 2026-10-26

Sources: repo READMEs, LICENSE files and HF cards fetched today. Nothing was installed. "?" means I could not verify it.

**Input caveat:** `r.amber-mens-suit.jpg` is cropped at mid-thigh and her arms are crossed. Every tool below will fuse the arms into the torso. Before running any of them, generate a full-length A-pose image (front, and ideally back) from the portrait and suit photos using our existing image-edit model.

## Candidates

| Name | Licence code / weights | UK-OK | Output | Animatable how | Hardware fit | Maturity |
|---|---|---|---|---|---|---|
| **TRELLIS.2** (MS) | MIT / MIT (HF card) | yes | GLB mesh, PBR textures, up to 1536³ | Not rigged; rig in Blender | ≥24 GB, tested on A100/H100. 4090 is borderline. Spark aarch64 means building flash-attn and nvdiffrast for sm_121 (?) | Paper Dec 2025, active |
| **LHM / LHM-1B** (Alibaba) | Apache-2.0 / **conflict**: HF tag says Apache-2.0, `modelcard.md` says CC BY-NC 4.0 "cannot be used commercially" | yes, non-commercial | 3DGS bound to SMPL-X; `export_mesh` actually writes a canonical-pose **Gaussian .ply**, not a mesh | SMPL-X params, using the repo's renderer | 14–24 GB; fits the 4090 | ICCV'25, stable |
| **LHM++** (Mar 2026) | Apache-2.0 / CC BY-NC 4.0 (LICENSE_WEIGHT) | yes, non-commercial | 3DGS PLY, takes any number of views (our 5 angles) | SMPL-X | 8 GB | Some weights only on ModelScope |
| **SAM 3D Body** (Meta, Nov 2025) | SAM License, worldwide, no commercial ban found / same; MHR rig Apache-2.0 | yes (needs Trade-Controls compliance) | MHR body mesh + skeleton; `lod?.fbx` rig. Body only, no hair or clothes | Native skeleton | VRAM ?; aarch64 ? | Stable |
| **SAM 3D Objects** | SAM License | yes | Gaussian PLY (mesh ?) | none | ? | Stable |
| **DiGS-Avatar** (ECCV'26) | MIT / Google Drive student checkpoint, licence ? | ? | SMPL-X-anchored 3DGS | LBS | ? | New; needs SMPL-X and IDOL assets |
| **IDOL** (CVPR'25) | MIT / ? | ? | Gaussians (?) on SMPL-X | novel_pose | ? | Inference code released Mar 2025 |
| **SiTH** | MIT / ? | ? | Textured mesh (UV), not rigged | Repose via Editable-humans | 3090 OK | Stale (Jun 2024) |
| **PSHuman** | MIT / ? | ? | Textured mesh, not rigged | none | **>40 GB**, so Spark only | Stale (Dec 2024) |
| **SPAR3D / SF3D** | Stability Community (free under $1M revenue) | yes | UV-textured mesh | none | Small | Objects; weak on humans |
| **SVAD** | Apache-2.0 (gitee mirror) / ? | ? | 3DGS (ExAvatar) | SMPL-X | 5–6 h per avatar | Research |
| **CharacterGen** | Apache-2.0 | yes | Mesh | via UniRig | ? | Anime/VRoid only |
| **HumanNOVA** (CVPR'26) | no LICENSE file, so unusable | no | Triplane / mesh | none | ? | Code May 2026 |
| AdaHuman, Dream-Lift-Animate (NVIDIA) | No code ("coming soon" / none) | n/a | 3DGS | | | Unreleased |
| MeshLAM | No code; head only (FLAME) | n/a | | | | Unreleased |
| **Hunyuan3D-2.x** | Territory excludes **EU, UK, South Korea** | **NO** | | | | Excluded |
| UniRig (auto-rig) | MIT / weights ? | ? | FBX skeleton+skin, GLB | Generic skeleton | ≥8 GB | Active |
| Make-It-Animatable | MIT / ? | ? | Mixamo-like rig (?) | | | v2: mesh input only |

**Licensing trap:** SMPL-X itself is MPI **non-commercial only**, with commercial licences via Meshcapade. Every SMPL-X tool (LHM, IDOL, DiGS, SiTH, SVAD, GVHMR) inherits that restriction. The fully commercial-clean chain is TRELLIS.2 (MIT) + MPFB (CC0 assets) + CMU mocap.

## Recommendation

**1. TRELLIS.2 mesh skinned to our existing MPFB/cmu_mb rig (best fit, MIT)**
1. Generate a full-length, front-facing A-pose image of her in the costume, on a plain background.
2. Run TRELLIS.2 image→GLB at 1024³ on the 4090. If it OOMs, use a Spark after building the CUDA extensions for aarch64.
3. In Blender 4.2, import the GLB and decimate it to about 50–100k faces.
4. Add the MPFB body. Match its proportions and A-pose to the GLB, then use the MPFB rig, which is already the cmu_mb skeleton.
5. Add a Data Transfer modifier: vertex groups, nearest-face-interpolated, from the MPFB body to the GLB. Then add an Armature modifier and hide the MPFB body.
6. Drive it with the CMU BVH exactly as today. Nothing changes downstream.

Hair comes out as a rigid shell skinned to head/neck/spine. That is acceptable for an LTX layout pass.

**2. LHM++ / LHM-1B Gaussians (better hair and likeness, non-commercial)**
1. Feed it the A-pose image plus the five angle images (LHM++ takes multiple views).
2. Get the canonical SMPL-X Gaussian PLY.
3. Choose a motion path:
   - (a) Render the avatar with LHM's own SMPL-X animator, using AMASS's SMPL-X fits of the CMU clips. AMASS is also non-commercial. Composite the result over the Blender room.
   - (b) Import the PLY with the KIRI 3dgs-render add-on (GPL-2). Its proxy-rig bake is **experimental**. Bind it to the SMPL-X Blender add-on armature and retarget the BVH.

## Video→mocap
- **GVHMR**: ZJU licence, non-commercial, SMPL output.
- **SAM 3D Body**: per-frame MHR output, so it needs temporal smoothing and conversion from MHR to BVH (?).
- WHAM: licence not verified; it uses SMPL.

## Risks
- TRELLIS.2 needs ≥24 GB, so it may not fit the 4090 (no offload allowed). Building it on aarch64 is unproven.
- Face likeness from a single mesh pass is often soft. LTX may still drift her identity.
- Rigid hair. Clothes intersect at extreme CMU poses.
- LHM's weights licence is contradictory; treat it as CC BY-NC.
- Not verified: licences for UniRig, IDOL, DiGS and SiTH weights; VRAM for SAM 3D, DiGS and IDOL; aarch64 support for every tool.
