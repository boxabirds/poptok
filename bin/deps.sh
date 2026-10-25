# Step dependencies (pipeline/deps.py), sourced by ./poptok and bin/make_video.sh. Each step is written
#   if need STEP STAMPDIR TARGET MODE "OUTPUT|OUTPUT..." -- INPUT...; then
#     ... the step ...
#     made
#   fi
# need: true when the step must run. A fresh run first moves the step's old OUTPUTs aside (NAME.prev, one generation
# kept), so tools that skip existing files can't reuse them; a resumed run keeps them. MODE fresh / resume / manual
# (manual: a stale step with outputs is only reported). DEPS_DRY=1 (./poptok --explain): report each step, run nothing.
if [ -n "${DEPS_DRY:-}" ] && [ -z "${DEPS_PENDING:-}" ]; then export DEPS_PENDING=$(mktemp); fi
need() {
  local step=$1 dir=$2 target=$3 mode=$4 outs=$5 r=0 why o
  shift 6
  if [ -n "${DEPS_DRY:-}" ]; then                     # each step reported once (./poptok runs the analysis twice)
    grep -qxF "$dir/$step" "$DEPS_PENDING.seen" 2>/dev/null && return 1
    echo "$dir/$step" >> "$DEPS_PENDING.seen"
  fi
  why=$(python3 pipeline/deps.py check "$dir" "$step" "$target" "$mode" -- "$@") || r=$?
  case $r in
    0) [ -n "${DEPS_DRY:-}" ] && echo "  $step: $why"; return 1;;
    10|11) ;;
    *) echo "deps: check of $step failed" >&2; exit 1;;
  esac
  if [ "$mode" = manual ] && [ -e "$target" ] && [[ ",${DEPS_REDO:-}," != *",$step,"* ]]; then
    echo "  $step: $why; its outputs are kept (edited by hand?): --redo $step to run it again"; return 1
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
  python3 pipeline/deps.py start "$dir" "$step" -- "$@"
  DEP_STEP=$step DEP_DIR=$dir
}
made() { python3 pipeline/deps.py done "$DEP_DIR" "$DEP_STEP"; }
