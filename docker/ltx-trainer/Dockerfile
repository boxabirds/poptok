# LTX-2.x trainer (Lightricks ltx-trainer, packages/ltx-trainer in ref/LTX-2) on top of the ltx2 image: LoRA training
# for per-video world/character LoRAs (pipeline/world/, 2026-10-26).
#   docker build -t ltx2-trainer docker/ltx-trainer        (on a box that has the ltx2 image)
FROM ltx2
WORKDIR /ltx
RUN uv sync --python 3.12 --package ltx-trainer --inexact && chmod -R a+rX /opt/venv
