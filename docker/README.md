# Images

The image names are historical, and the scripts use them as they are.

| image | build | used for |
|---|---|---|
| `m2v-pre` | `docker build -t m2v-pre docker/analysis` | stems, Whisper, beat grid, lyric scoring |
| `zoommorph` | `docker build -t zoommorph docker/keyframes` | MOSS-Image keyframes |
| `ltx2` | `git clone https://github.com/Lightricks/LTX-2 ref/LTX-2 && docker build -t ltx2 -f docker/ltx/Dockerfile ref/LTX-2` | video (x86 or ARM64 such as DGX Spark; on the Spark, pass `--build-arg BASE=` an ARM64 CUDA image) |
| `segment` | `docker build -t segment docker/segment` | plausibility critic geometry: LLMDet / Grounding DINO + SAM 2.1 (`pipeline/segment_check.py`, POC 28) |
| `acestep` | `git clone https://github.com/ace-step/ACE-Step-1.5 && docker build -t acestep ACE-Step-1.5` (upstream Dockerfile) | song generation (only needed to generate songs) |

Every script runs its container as your user, with the repo mounted at `/work`.
