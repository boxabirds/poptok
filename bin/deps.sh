# Step dependencies, sourced by ./poptok and bin/make_video.sh. pipeline/steps.ini declares each step's inputs, target,
# outputs and mode (the graph); pipeline/deps.py decides from content hashes whether it must run. Each step is written
#   if need STEP; then
#     ... the step's commands (its recipe) ...
#     made
#   fi
# with the values steps.ini refers to ({OUT}, {CUES}, ...) exported by the script. need: true when the step must run.
# A fresh run first moves the step's old outputs aside (NAME.prev, one generation kept), so tools that skip existing
# files can't reuse them; a resumed run keeps them; a stale manual step with outputs is only reported.
# DEPS_DRY=1 (./poptok --explain): report each step, run nothing.
if [ -n "${DEPS_DRY:-}" ] && [ -z "${DEPS_PENDING:-}" ]; then export DEPS_PENDING=$(mktemp); fi
need() {
  local step=$1 dir target mode outs r=0 why o A
  mapfile -d '' A < <(python3 pipeline/steps.py args "$step") && [ ${#A[@]} -ge 4 ] \
    || { echo "deps: no usable entry for $step in pipeline/steps.ini" >&2; exit 1; }
  dir=${A[0]} target=${A[1]} mode=${A[2]} outs=${A[3]}; A=("${A[@]:4}")
  if [ -n "${DEPS_DRY:-}" ]; then                     # each step reported once (./poptok runs the analysis twice)
    grep -qxF "$dir/$step" "$DEPS_PENDING.seen" 2>/dev/null && return 1
    echo "$dir/$step" >> "$DEPS_PENDING.seen"
  fi
  why=$(python3 pipeline/deps.py check "$dir" "$step" "$target" "$mode" -- "${A[@]}") || r=$?
  case $r in
    0) [ -n "${DEPS_DRY:-}" ] && echo "  $step: $why"; until_done "$step"; return 1;;
    10|11) ;;
    *) echo "deps: check of $step failed" >&2; exit 1;;
  esac
  if [ "$mode" = manual ] && [ -e "$target" ] && [[ ",${DEPS_REDO:-}," != *",$step,"* ]]; then
    echo "  $step: $why; its outputs are kept (edited by hand?): --redo $step to run it again"; until_done "$step"; return 1
  fi
  if [ -n "${DEPS_DRY:-}" ]; then
    echo "  $step: would $([ $r = 11 ] && echo resume || echo run) ($why)"
    tr '|' '\n' <<< "$outs" >> "$DEPS_PENDING"; echo "$target" >> "$DEPS_PENDING"; return 1
  fi
  echo "[$(date +%H:%M:%S)] $step: $([ $r = 11 ] && echo "resuming, ")$why"
  if [ $r = 10 ] && [ "$mode" != manual ]; then
    IFS='|' read -ra O <<< "$outs"
    for o in "${O[@]}"; do [ -e "$o" ] || [ -L "$o" ] && { rm -rf "$o.prev"; mv "$o" "$o.prev"; }; done
  fi
  python3 pipeline/deps.py start "$dir" "$step" -- "${A[@]}"
  DEP_STEP=$step DEP_DIR=$dir
}
made() { python3 pipeline/deps.py done "$DEP_DIR" "$DEP_STEP"; until_done "$DEP_STEP"; }
# ./poptok --until STEP: stop once STEP is built or up to date (e.g. --until universe, to choose a universe before the
# storyboard, prompts and stills are made from it)
until_done() { [ -n "${DEPS_UNTIL:-}" ] && [ "$1" = "$DEPS_UNTIL" ] && [ -z "${DEPS_DRY:-}" ] && { echo "[$(date +%H:%M:%S)] stopped after $1 (--until)"; exit 0; }; return 0; }
