#!/usr/bin/env bash
# Song -> beat-synced music video. LTX-2.5 + Transition LoRA chain, one scene per distinct lyric prompt, scene changes
# retimed onto downbeats, an occasional pulse on real bass hits, graded.
#
#   bin/make_video.sh SONG.wav --lyrics LYRICS.txt (--treatment TREATMENT.txt | --style "...") --out OUT
#        [--analysis DIR] [--name NAME] [--video-host HOST] [--plan-only] [--keep-text]
#        [--key-hosts "thing2 thing1 local tritus"] [--ltx-args "..."] [--finish-args "..."] [--merge-args "..."]
#        [--cues CUED_LYRICS --performer-photo PHOTO [--performer "how she looks"]]
# Environment: STYLE_WORD + STYLE_BRIEF (a few-word style and its brief, POC 24) are written into every prompt;
# PULSE=0 leaves out the bass-hit pulse (slow songs). FX: the piece's effects config (pipeline/fx/). WARDROBE: what the performer wears in this piece (no apostrophes).
# Normally called by ./poptok with a piece folder. Runs on the box with the 4090 (analysis, LLM, keyframes) and renders
# video on --video-host (default: this box). Keyframes go to the first free box in --key-hosts (bin/hosts.sh; default:
# the video host, then this box). Every step runs again when its inputs change and is skipped otherwise: what each
# is built from is declared in pipeline/steps.ini (bin/deps.sh), so a re-run resumes and a new storyboard reaches the
# stills.
# One GPU job at a time per box: each step waits while another pipeline container is running.
#   1 analysis   stems + Whisper with word timestamps (pipeline/analyse.py), beat grid
#   2 lyrics     master lyrics placed on Whisper's clock (align_lyrics.py); Whisper's own words are not used
#   3 prompts    --treatment: treatment.py (whole-video treatment, then one scene per lyric line from it)
#                --style:     prompts_per_line.py (one independent prompt per line + the style; the C5/C6 method)
#   4 keyframes  MOSS-Image, one per bar (keyframes.py), then merge near-duplicate scenes (merge_scenes.py)
#   5 video      LTX chain (ltx_chain.py) on --video-host
#   6 finish     retime scene changes onto downbeats, pulse, grade, mux -> OUT/NAME.mp4
#   7 lip-sync   (--performer-photo) every run of sung scenes becomes an InfiniteTalk clip from its still and the vocal,
#                rendered on gruntus (int8) and the Sparks (SING_HOSTS, default "gruntus $AMBER_HOSTS") and spliced in -> OUT/NAME.mp4; the cut
#                without them is kept as OUT/NAME-nosync.mp4. Rendered inserts are reused on a re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SONG=$1; shift
STYLE=""; LYRICS=""; NAME=""; OUT=""; ANA=""; VHOST=""; NOTEXT=--no-text; TREAT=""; PLANONLY=""; ANAONLY=""
LTX_ARGS="--even-light"; FINISH_ARGS="--deflare 8"; MERGE_ARGS=""; KEY_HOSTS=""; LCTX=""; CUES=""; PHOTO=""
PERFORMER="a young woman with long wavy blond hair"; AMBER_HOSTS=${AMBER_HOSTS:-thing1 thing2}
while [ $# -gt 0 ]; do case $1 in
  --style) STYLE=$2; shift 2;; --lyrics) LYRICS=$2; shift 2;; --name) NAME=$2; shift 2;; --out) OUT=$2; shift 2;;
  --analysis) ANA=$2; shift 2;; --video-host) VHOST=$2; shift 2;; --keep-text) NOTEXT=""; shift;;
  --treatment) TREAT=$2; shift 2;; --plan-only) PLANONLY=1; shift;; --analysis-only) ANAONLY=1; shift;;
  --key-hosts) KEY_HOSTS=$2; shift 2;; --cues) CUES=$2; shift 2;; --performer-photo) PHOTO=$2; shift 2;;
  --performer) PERFORMER=$2; shift 2;; --lyrics-context) LCTX=$2; shift 2;; --ltx-args) LTX_ARGS=$2; shift 2;; --finish-args) FINISH_ARGS=$2; shift 2;; --merge-args) MERGE_ARGS=$2; shift 2;;
  *) echo "unknown option $1"; exit 2;; esac; done
[ -n "$STYLE$TREAT" ] || { echo "--treatment or --style is required"; exit 2; }
[ -n "$LYRICS" ] || { echo "--lyrics is required (the master lyrics, with [Section] tags)"; exit 2; }
[ -n "$OUT" ] || { echo "--out is required"; exit 2; }
# cues without a performer photo: the lines' shot notes direct the stills; she is never drawn (modes none/pov only)
STEM=$(basename "${SONG%.*}"); ANA=${ANA:-$(dirname "$SONG")/$STEM-analysis}; NAME=${NAME:-$STEM}; TAG=-true
mkdir -p "$OUT" "$ANA"
SEGS=$ANA/$STEM$TAG-segments.json; LYRICS_FILE=$LYRICS
STORY=${STORY:-}                 # the workshop's story-scenes.json (./poptok sets it for WORKSHOP=1 pieces)
export ANA OUT STEM TAG SONG SEGS LYRICS_FILE CUES TREAT LCTX PHOTO STYLE PERFORMER NOTEXT MERGE_ARGS LTX_ARGS STORY  # steps.ini
KEY_HOSTS=${KEY_HOSTS:-${VHOST:-local}}          # never offload: list only boxes where MOSS (33 GB) fits
log() { echo "[$(date +%H:%M:%S)] $*"; }
source bin/hosts.sh
source bin/deps.sh                                           # need / made: step dependencies
IMAGES='^(acestep|m2v-pre|zoommorph|ltx2)'                    # the pipeline's GPU images (docker/README.md)
# one GPU job per box: a lock shared by every poptok run on this box (held from the start of a GPU step to its end),
# plus a wait for pipeline containers started by anything else
gpu_lock() { exec 9>"${TMPDIR:-/tmp}/poptok-gpu.lock"; flock 9
             while docker ps --format '{{.Image}}' | grep -qE "$IMAGES"; do sleep 20; done; }
gpu_unlock() { flock -u 9; }
analysis() { gpu_lock; docker run --gpus all --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_HOME=/cache/torch \
               -v "$ROOT":/work -v "$ROOT/cache":/cache -w /work m2v-pre "$@"; gpu_unlock; }
keyframes() { gpu_lock; docker run --gpus all --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf \
                -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e OFFLOAD=0 -v "$ROOT":/work \
                -v "$HOME/.cache/huggingface":/hf -v "$HOME/models":/models -w /work zoommorph "$@"; gpu_unlock; }
LLM_SERVER=${LLM_SERVER:-$HOME/.local/bin/llama-server}
LLM_GGUF=${LLM_GGUF:-$HOME/.local/share/qwen38-27b/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf}
LLM_URL=http://127.0.0.1:8091
llm_up() { gpu_lock; "$LLM_SERVER" -m "$LLM_GGUF" -ngl 99 -c 16384 --reasoning-budget 0 --host 127.0.0.1 --port 8091 \
             > "$OUT/llm_server.log" 2>&1 9>&- & LLM_PID=$!; until curl -sf $LLM_URL/health >/dev/null; do sleep 2; done; }
llm_down() { kill $LLM_PID; wait $LLM_PID 2>/dev/null || true; gpu_unlock; }

# 1 analysis
if need analysis; then
  log "analysis: stems + whisper"; analysis python3 pipeline/analyse.py "$SONG" --analysis "$ANA"; made; fi
if need beats; then
  log "beat grid"; analysis python3 pipeline/beat_grid.py "$SONG" "$ANA"; made; fi
if need drums; then
  python3 pipeline/drum_hits.py "$ANA" "$STEM" > /dev/null; made; fi          # deep drum hits (bursts, workshop)
# 2 lyrics
if need segments; then
  python3 pipeline/align_lyrics.py "$SONG" "$LYRICS" --analysis "$ANA"      # lyric timings: Whisper matching, then forced
  analysis python pipeline/force_align.py "$SONG" "$LYRICS" --analysis "$ANA" --merge="$TAG" > "$ANA/force_align.log" 2>&1 \
    || log "forced alignment failed ($ANA/force_align.log): keeping Whisper-matched timings"   # alignment of the known
  made                                                                      # lyrics on the vocal stem
fi
# --analysis-only: stop here (./poptok runs this before the directors' workshop, which needs the lyric timings)
[ -n "$ANAONLY" ] && { [ -n "${DEPS_DRY:-}" ] || log "analysis done: $ANA"; exit 0; }
# 3 prompts: everything that shapes a shot's prompt is an input, so a new storyboard, treatment or style rebuilds them
if need prompts; then
  llm_up
  if [ -n "$TREAT" ]; then
    log "treatment + scene prompts"
    python3 pipeline/treatment.py "$SONG" "$LYRICS" "$TREAT" --out "$OUT" --analysis "$ANA" --llm $LLM_URL \
      --tag="$TAG" > "$OUT/prompts.log" 2>&1 || { llm_down; exit 1; }
  else
    log "per-line prompts (${NOTEXT:-text allowed})"
    python3 pipeline/prompts_per_line.py "$SONG" --style "$STYLE" --api openai --llm $LLM_URL --model local \
      --segments "$ANA/$STEM$TAG-segments.json" --analysis "$ANA" --tag="$TAG" $NOTEXT ${LCTX:+--lyrics-context "$LCTX"} ${CUES:+--cues "$CUES" --performer "$PERFORMER"} ${WARDROBE:+--wardrobe "$WARDROBE"} \
      ${STYLE_WORD:+--style-word "$STYLE_WORD" --style-brief "$STYLE_BRIEF"} ${STORY:+--story "$STORY"} > "$OUT/prompts.log" 2>&1 \
      || { llm_down; exit 1; }
  fi
  llm_down; made
fi
if [ -n "$PLANONLY" ] && [ -z "${DEPS_DRY:-}" ]; then log "plan only: read $OUT/treatment.md and $OUT/prompts.log"; exit 0; fi
# 4 keyframes + scene merge. With --cues: the shots that show the performer go to amber_keys.py (Qwen-Image-Edit with her
# photo) on a Spark, at the same time as MOSS renders the rest on another box
remote_keys() {   # host, container command: code, analysis, run folder and keys go over; keys come back
  on "$1" "mkdir -p ~/poptok/$OUT ~/poptok/$ANA"
  rsync -a pipeline "$1":poptok/; rsync -a "$ANA"/ "$1":poptok/"$ANA"/; rsync -a "$OUT/shots.json" "$1":poptok/"$OUT"/
  [ -d "$OUT/keys" ] && { on "$1" "mkdir -p ~/poptok/$OUT/keys"; rsync -a --delete "$OUT/keys/" "$1":poptok/"$OUT"/keys/; }
  [ -f "$OUT/reroll.json" ] && rsync -a "$OUT/reroll.json" "$1":poptok/"$OUT"/
  [ -n "$PHOTO" ] && { on "$1" "mkdir -p ~/poptok/$(dirname "$PHOTO")"; rsync -a "$PHOTO" "$1":poptok/"$PHOTO"; }
  on "$1" "cd ~/poptok && flock $LOCK $2"
  rsync -a "$1":poptok/"$OUT"/keys "$OUT"/
}
if need keys; then
  KARGS="pipeline/keyframes.py $SONG --style \"${STYLE:-none}\" --meta full-metadata$TAG.json --analysis $ANA --out $OUT"
  eval python3 $KARGS --plan-only > "$OUT/shots.log"                    # the shot list (and each shot's cue)
  mkdir -p "$OUT/keys"
  KARGS0=$KARGS
  for ROUND in 1 2 3; do            # a pass of keyframes, then the still check (check_keys.py); re-roll flagged stills
    KARGS=$KARGS0
    AMBER_PID=""
    if [ -n "$CUES" ] && [ -n "$PHOTO" ]; then
      KARGS="$KARGS --skip-cast"
      AH=$(pick_host $AMBER_HOSTS)
      log "performer keyframes (Qwen-Image-Edit, her photo) on $AH"
      remote_keys "$AH" "docker run --gpus all --rm --user \$(id -u):\$(id -g) --ipc=host -e HOME=/tmp -v \$HOME/poptok:/work \
        -v \$HOME/models:/models -w /work morph-spark python3 pipeline/amber_keys.py $OUT --photo $PHOTO \
        --style \"${STYLE:-}\" ${STYLE_WORD:+--style-word \"$STYLE_WORD\" --style-brief \"$STYLE_BRIEF\"} ${WARDROBE:+--wardrobe \"$WARDROBE\"}" \
        > "$OUT/amber_keys.log" 2>&1 &
      AMBER_PID=$!; sleep 60                                              # let it take its box before MOSS picks one
    fi
    KH=$(pick_host $KEY_HOSTS)
    log "keyframes (MOSS) on $KH"
    if [ "$KH" = local ]; then
      eval keyframes python3 $KARGS > "$OUT/keys.log" 2>&1
    else
      remote_keys "$KH" "$(keyframe_ctr "$KH") python3 $KARGS" > "$OUT/keys.log" 2>&1
    fi
    [ -n "$AMBER_PID" ] && { wait $AMBER_PID || { log "performer keyframes failed: $OUT/amber_keys.log"; exit 1; }; }
    log "still check (round $ROUND)"
    python3 pipeline/check_keys.py "$OUT" --reroll > "$OUT/check_keys_$ROUND.log" 2>&1 \
      || { log "still check FAILED (see $OUT/check_keys_$ROUND.log); going on unchecked"; break; }
    tail -n 1 "$OUT/check_keys_$ROUND.log"
    grep -q " 0 to re-roll" "$OUT/check_keys_$ROUND.log" && break
  done
  touch "$OUT/keys/.done"; made
fi
if need merge; then
  log "merge near-duplicate scenes"; llm_up
  python3 pipeline/merge_scenes.py "$OUT" --api openai --llm $LLM_URL --model local $MERGE_ARGS > "$OUT/merge.log" 2>&1 \
    || { llm_down; exit 1; }
  llm_down; made
fi
# scenes (pipeline/scenes.py): the scene list to edit, and an LLM-written A->B morph for every scene change, which the
# chain prompts with the Transition LoRA's trigger word
if need scenes; then
  log "scenes + transitions"; llm_up
  python3 pipeline/scenes.py build "$OUT" ${CUES:+--cues "$CUES"} --segments "$SEGS" ${STORY:+--story "$STORY"} \
      > "$OUT/scenes.log" 2>&1 \
    && python3 pipeline/scenes.py transitions "$OUT" --llm $LLM_URL ${STORY:+--story "$STORY"} >> "$OUT/scenes.log" 2>&1 \
    || { llm_down; exit 1; }
  llm_down; made
fi
# HOLD_BEFORE_VIDEO=1: stop here so the stills can be reviewed (and fixed: pipeline/fix_keys.py); run again to go on
if [ -n "${HOLD_BEFORE_VIDEO:-}" ] && [ -z "${DEPS_DRY:-}" ] && [ "$(ls "$OUT/ltx_chain" 2>/dev/null | grep -c png)" -eq 0 ]; then
  log "held before video: review $OUT/keys, then run again without HOLD_BEFORE_VIDEO"; exit 0; fi
# 5 video. A fresh chain moves the old frames aside (ltx_chain.prev) and clears the video host's copy (rsync --delete),
# since ltx_chain.py and the retime skip frames that exist; scenes.py apply removes only the frames to re-render and
# accepts the rest (deps.py accept)
if need video; then
  N=$(python3 -c "import json;print(round(json.load(open('$OUT/shots.json'))['shots'][-1]['end']*24))")
  LTX="python pipeline/ltx_chain.py $OUT --merge --plan shots_merged.json $LTX_ARGS"
  mkdir -p "$OUT/ltx_chain"
  if [ -z "$VHOST" ]; then
    log "video: LTX chain here ($N frames)"; gpu_lock
    docker run --gpus all --rm --user "$(id -u):$(id -g)" --ipc=host -e HOME=/tmp -v "$ROOT":/work -v "$HOME/models":/models \
      -w /work ltx2 $LTX > "$OUT/ltx.log" 2>&1; gpu_unlock
  elif is_mac "$VHOST"; then      # native on Apple silicon (same ltx_chain.py, MPS), under ~/expts/poptok
    log "video: LTX chain on $VHOST, native ($N frames)"
    ssh "$VHOST" "mkdir -p ~/expts/poptok/$OUT"
    rsync -a pipeline "$VHOST":expts/poptok/
    rsync -a "$OUT/keys" "$OUT/shots.json" "$OUT/shots_merged.json" "$OUT/scenes.json" "$VHOST":expts/poptok/"$OUT"/
    rsync -a --delete --exclude .done "$OUT/ltx_chain/" "$VHOST":expts/poptok/"$OUT"/ltx_chain/   # the frames to keep
    ssh "$VHOST" "cd ~/expts/poptok && until mkdir ~/.poptok-gpu.lock.d 2>/dev/null; do sleep 30; done; \
      trap 'rmdir ~/.poptok-gpu.lock.d' EXIT; LTX_MODELS=\$HOME/models PYTORCH_ENABLE_MPS_FALLBACK=1 \
      .venv-ltx/bin/$LTX" > "$OUT/ltx.log" 2>&1
    rsync -a "$VHOST":expts/poptok/"$OUT"/ltx_chain "$OUT"/
  else                            # the remote keeps the same layout under ~/poptok (pipeline code + this run's folder)
    log "video: LTX chain on $VHOST ($N frames)"
    ssh "$VHOST" "mkdir -p ~/poptok/$OUT"
    rsync -a pipeline "$VHOST":poptok/
    rsync -a "$OUT/keys" "$OUT/shots.json" "$OUT/shots_merged.json" "$OUT/scenes.json" "$VHOST":poptok/"$OUT"/
    rsync -a --delete --exclude .done "$OUT/ltx_chain/" "$VHOST":poptok/"$OUT"/ltx_chain/   # the frames to keep
    ssh "$VHOST" "cd ~/poptok && $PREP while $CTRS | grep -qE '$BUSY_RE'; do sleep 30; done; \
      flock $LOCK $(ltx_ctr "$VHOST") $LTX" > "$OUT/ltx.log" 2>&1         # NVIDIA (docker) or ROCm (podman)
    rsync -a "$VHOST":poptok/"$OUT"/ltx_chain "$OUT"/
  fi
  NF=$(ls "$OUT/ltx_chain" | grep -c png || true)
  [ "$NF" -ge "$N" ] || { log "video incomplete: $NF of $N frames (see $OUT/ltx.log); run again to resume"; exit 1; }
  touch "$OUT/ltx_chain/.done"; made
fi
if [ -n "${DEPS_DRY:-}" ]; then
  echo "  finish: always runs (retime, grade, mux)"
  [ -n "$PHOTO" ] && { need lipsync || true; }
  exit 0
fi
# 6 finish
log "retime, pulse, grade, mux"
python3 pipeline/retime_to_beats.py "$OUT" --src ltx_chain --dst ltx_chain_beat --clips
GRADE=(); [ -n "$TREAT" ] && GRADE=(--grade treatment --step "$(python3 -c "import json;print(json.load(open('$OUT/shots_merged.json')).get('treatment',{}).get('frame_step',1))")")
python3 pipeline/finish.py "$OUT" --audio "$SONG" --analysis "$ANA" --plan shots_merged.json --frames-dir "$OUT/ltx_chain_beat" \
  --src-fps 24 --no-flash $([ "${PULSE:-1}" = 0 ] && echo --no-pulse || echo --punch-lowhits) --name "$NAME" "${GRADE[@]}" $FINISH_ARGS \
  ${FX:+--fx "$(dirname "$(dirname "$OUT")")/$FX"}
# lip-sync (sing_inserts.py, bin/sing_inserts.sh): the sung scenes, lip-synced from their stills, cut into the video
if [ -n "$PHOTO" ] && need lipsync; then
  python3 pipeline/sing_inserts.py plan "$OUT" "$ANA" "$STEM" --performer "$PERFORMER" > "$OUT/sing.log" 2>&1
  NS=$(python3 -c "import json;print(len(json.load(open('$OUT/sing/inserts.json'))))")
  if [ "$NS" -gt 0 ]; then
    read -ra SH <<< "${SING_HOSTS:-gruntus $AMBER_HOSTS $ROCM_HOSTS}"
    RATES=$(for h in "${SH[@]}"; do printf '%s=%s ' "$h" "$(talk_rate "$h")"; done)
    # longest insert first, each to the host that would finish it soonest (gruntus int8 ~62 s per second of video,
    # a Spark ~163 s, a Strix Halo ~348 s; bin/hosts.sh talk_rate)
    SPLAN=$(python3 -c "
import json
rs = sorted(json.load(open('$OUT/sing/inserts.json')), key=lambda r: -r['duration'])
rate = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in '$RATES'.split()}
busy, got = {h: 0.0 for h in rate}, {h: [] for h in rate}
for r in rs:
    h = min(rate, key=lambda h: busy[h] + rate[h] * r['duration'])
    busy[h] += rate[h] * r['duration']; got[h].append(r['k'])
print(' '.join(h + ':' + ','.join(map(str, sorted(ks))) for h, ks in got.items() if ks))")
    log "lip-sync: $NS sung inserts (InfiniteTalk) on $SPLAN"
    bin/sing_inserts.sh "$OUT" "$SPLAN" >> "$OUT/sing.log" 2>&1
  fi
  MISSING=$(python3 -c "
import json, os
D = '$OUT/sing'
ok = lambda k: os.path.getsize(f'{D}/it_out_{k}.mp4') > 0 if os.path.exists(f'{D}/it_out_{k}.mp4') else os.path.isdir(f'{D}/out_{k}')
print(sum(not ok(r['k']) for r in json.load(open(f'{D}/inserts.json'))))")
  if [ "$MISSING" = 0 ]; then touch "$OUT/sing/.done"; made
  else log "lip-sync: $MISSING inserts failed (see $OUT/sing.log); the video uses the rest; run again to retry them"; fi
fi
if [ -n "$PHOTO" ] && [ "$(python3 -c "import json;print(len(json.load(open('$OUT/sing/inserts.json'))))")" -gt 0 ]; then
  mv "$OUT/$NAME.mp4" "$OUT/$NAME-nosync.mp4"
  python3 pipeline/sing_inserts.py splice "$OUT" "$OUT/$NAME-nosync.mp4" "$OUT/$NAME.mp4" \
    ${FX:+--fx "$(dirname "$(dirname "$OUT")")/$FX" --fx-analysis "$ANA"} | tee -a "$OUT/sing.log"   # effects inside an insert go back on
fi
log "done: $OUT/$NAME.mp4"
