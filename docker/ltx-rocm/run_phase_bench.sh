#!/usr/bin/env bash
# LTX phase benchmark (pipeline/ltx_phase_bench.py) on tritus in the ltx2-rocm image. Run ON tritus from ~/poptok
# (pipeline/ and pieces/_bench/ltx_phase/{a,b}.png there):
#   bash ~/poptok-build/ltx-rocm/run_phase_bench.sh tritus              # AOTriton on (image default)
#   KERNARG=1 bash ~/poptok-build/ltx-rocm/run_phase_bench.sh tritus-kernarg   # + HIP_FORCE_DEV_KERNARG, expandable segments
# Writes pieces/_bench/ltx_phase/NAME.json and NAME.log. Takes the poptok GPU lock (one GPU job per box).
set -euo pipefail
NAME=$1
EXTRA=()
[ "${KERNARG:-0}" = 1 ] && EXTRA=(-e HIP_FORCE_DEV_KERNARG=1 -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True)
CACHE=$HOME/.cache/ltx-rocm; mkdir -p "$CACHE/miopen"
flock "${TMPDIR:-/tmp}/poptok-gpu.lock" podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt label=disable --ipc=host -e HOME=/tmp -e LTX_MODELS=/models -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e MIOPEN_USER_DB_PATH=/cache/miopen -e MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor "${EXTRA[@]}" -v "$CACHE:/cache" -v "$HOME/models:/models:ro" \
  -v "$PWD:/work" -w /work localhost/ltx2-rocm \
  python pipeline/ltx_phase_bench.py pieces/_bench/ltx_phase/a.png pieces/_bench/ltx_phase/b.png \
  "pieces/_bench/ltx_phase/$NAME.json" > "pieces/_bench/ltx_phase/$NAME.log" 2>&1
