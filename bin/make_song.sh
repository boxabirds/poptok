#!/usr/bin/env bash
# Musical style + lyrics -> song (ACE-Step 1.5 XL-sft with the 4B planning LM), best of N seeds by lyric intelligibility.
#
#   bin/make_song.sh MUSIC_STYLE.txt LYRICS.txt OUT.wav [--seeds 1,2,3] [--duration auto|SECONDS]
#
# - caption: the style text; if it is over ACE-Step's 512-character limit, local Qwen 3.8 condenses it (genre, tempo,
#   instruments, vocals, mood, production kept; per-section directions dropped, as they belong in the lyric tags)
# - BPM: taken from the style text if it names one ("90-96 BPM" -> 93), else ACE-Step's LM chooses
# - lyrics: used as given ([Verse]/[Chorus] tags and descriptors such as [Chorus - gang vocals] help)
# - each seed is transcribed (Demucs vocals are NOT used here: full-mix Whisper, as in ace_score.py) and compared with
#   the lyrics; the best match is copied to OUT.wav. Work files in OUT_dir/<name>-song/.
# One GPU job at a time on this box.
set -euo pipefail
cd "$(dirname "$0")/.."
STYLE=$1; LYR=$2; OUTWAV=$3; shift 3
SEEDS=1,2,3; DUR=auto
while [ $# -gt 0 ]; do case $1 in --seeds) SEEDS=$2; shift 2;; --duration) DUR=$2; shift 2;; *) echo "unknown $1"; exit 2;; esac; done
W="$(dirname "$OUTWAV")/$(basename "${OUTWAV%.*}")-song"; mkdir -p "$W"
log() { echo "[$(date +%H:%M:%S)] $*"; }
gpu_free() { while docker ps --format '{{.Image}}' | grep -qE '^(acestep|m2v-pre|zoommorph|ltx2)'; do sleep 20; done; }

cp "$LYR" "$W/lyrics.txt"
python3 - "$STYLE" "$W/caption_raw.txt" "$W/bpm.txt" <<'EOF'
import re, sys
s = open(sys.argv[1]).read().strip(); open(sys.argv[2], "w").write(s)
m = re.search(r"(\d{2,3})\s*(?:[-–to]+\s*(\d{2,3}))?\s*BPM", s, re.I)
open(sys.argv[3], "w").write(str(round((int(m.group(1)) + int(m.group(2) or m.group(1))) / 2)) if m else "0")
EOF
if [ "$(wc -c < "$W/caption_raw.txt")" -gt 500 ]; then
  log "condensing the style to < 512 characters (Qwen 3.8)"; gpu_free
  "${LLM_SERVER:-$HOME/.local/bin/llama-server}" -m "${LLM_GGUF:-$HOME/.local/share/qwen38-27b/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf}" \
    -ngl 99 -c 8192 --reasoning-budget 0 --host 127.0.0.1 --port 8092 > "$W/llm_server.log" 2>&1 & P=$!
  until curl -sf http://127.0.0.1:8092/health >/dev/null; do sleep 2; done
  python3 - "$W/caption_raw.txt" "$W/caption.txt" <<'EOF' || { kill $P; exit 1; }
import json, sys, urllib.request
src = open(sys.argv[1]).read()
for attempt in range(3):
    q = ("Rewrite this music style description for an AI music generator in UNDER 480 characters. Keep genre, tempo, "
         "key/mood, instruments, drums and bass, vocal type and delivery, texture/FX and production/mastering. Drop "
         "directions about specific song sections (intro, hook, outro). Plain comma-separated phrases, no preamble.\n\n" + src)
    body = json.dumps({"messages": [{"role": "user", "content": q}], "temperature": 0.3, "max_tokens": 400}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions", body,
        {"Content-Type": "application/json"}), timeout=300).read())["choices"][0]["message"]["content"].strip().strip('"')
    if len(r) < 512:
        open(sys.argv[2], "w").write(r); print(len(r), "chars:", r); break
else:
    sys.exit("could not condense the style under 512 characters")
EOF
  kill $P; wait $P 2>/dev/null || true
else cp "$W/caption_raw.txt" "$W/caption.txt"; fi
BPM=$(cat "$W/bpm.txt"); D=$([ "$DUR" = auto ] && echo -1 || echo "$DUR")
log "ACE-Step XL-sft + 4B LM, seeds $SEEDS, bpm ${BPM/#0/auto}, duration $DUR"; gpu_free
docker run --gpus all --rm --user "$(id -u):$(id -g)" --ipc=host -e HOME=/tmp -e USER="$USER" -e LOGNAME="$USER" \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -e HF_HOME=/ckpt/hf -e ACESTEP_CHECKPOINTS_DIR=/ckpt -e PYTHONPATH=/app \
  -v ~/models/acestep:/ckpt -v "$PWD":/work -w /app --entrypoint /app/.venv/bin/python acestep \
  /work/pipeline/ace_gen.py "/work/$W" --variants xl-sft --seeds "$SEEDS" --duration "$D" --bpm "$BPM" \
  --caption-file "/work/$W/caption.txt" --lyrics-file "/work/$W/lyrics.txt" --name song > "$W/ace.log" 2>&1
log "scoring lyric intelligibility per seed"; gpu_free
docker run --gpus all --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/work -v "$PWD/cache":/cache -w /work \
  m2v-pre python3 pipeline/ace_score.py "$W" "$W/lyrics.txt" 2>&1 | grep -E "words heard" | tee "$W/scores.txt"
BEST=$(python3 -c "import glob,json;r=max((json.load(open(f)) for f in glob.glob('$W/*.score.json')),key=lambda x:x['match']);print(r['file'])")
cp "$BEST" "$OUTWAV"; log "best: $(basename "$BEST") -> $OUTWAV"
