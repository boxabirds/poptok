#!/usr/bin/env bash
# Apple Silicon renderer environment (no containers on macOS: Linux VMs cannot reach Metal). A uv virtual environment
# in this project (.venv-ltx) with PyTorch for MPS and the official LTX pipelines pinned to the same commit as the
# Sparks' ltx2 image. Adapted from macvideo's scripts/setup_renderer_env.sh: torch first, from the public index, then
# ltx-core / ltx-pipelines with --no-deps (their torch pin points at the CUDA index, which has no arm64 wheels).
#   bash bin/setup_mac_renderer.sh       (on the Mac, in ~/expts/poptok)
# Models: ~/models/ltx-2.5 (same layout as the Sparks) and ~/models/ltx-2.3/ltx2.3-transition.safetensors.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=.venv-ltx
LTX_REF=e98f6d927d83c92ef2be6aec76c8440b19306e39
uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" --no-config -q "torch>=2.11" torchaudio torchvision "transformers>=5.8.0,<5.15" \
  "mps-sdpa>=0.2.0" einops numpy av safetensors accelerate "scipy>=1.14" tqdm pillow openimageio colour-science
uv pip install --python "$VENV/bin/python" --no-config --no-deps -q \
  "git+https://github.com/Lightricks/LTX-2.git@${LTX_REF}#subdirectory=packages/ltx-core" \
  "git+https://github.com/Lightricks/LTX-2.git@${LTX_REF}#subdirectory=packages/ltx-pipelines"
"$VENV/bin/python" -c "import torch, ltx_pipelines.distilled; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
