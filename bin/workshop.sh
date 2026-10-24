#!/usr/bin/env bash
# Directors' workshop step (pipeline/workshop.py): lyrics -> visual treatment + line-by-line storyboard (cued lyrics),
# with the local LLM, under this box's GPU lock.   bin/workshop.sh pieces/NAME [--force]
set -euo pipefail
cd "$(dirname "$0")/.."
LLM_SERVER=${LLM_SERVER:-$HOME/.local/bin/llama-server}
LLM_GGUF=${LLM_GGUF:-$HOME/.local/share/qwen38-27b/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf}
exec 9>"${TMPDIR:-/tmp}/poptok-gpu.lock"; flock 9
"$LLM_SERVER" -m "$LLM_GGUF" -ngl 99 -c 16384 --reasoning-budget 0 --host 127.0.0.1 --port 8091 > "$1/workshop_llm.log" 2>&1 9>&- &
L=$!; trap 'kill $L 2>/dev/null; wait $L 2>/dev/null || true' EXIT
until curl -sf http://127.0.0.1:8091/health >/dev/null; do sleep 2; done
python3 pipeline/workshop.py "$@"
