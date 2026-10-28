# Where GPU steps run, and how. Sourced by make_video.sh.
#
# Every box has one poptok GPU lock (${TMPDIR:-/tmp}/poptok-gpu.lock). A step holds it for its whole run, so each box
# runs one GPU job at a time across all poptok runs. A box counts as free when its lock is free AND no GPU container
# (ours or anything else's) is running there.
#
# Host kinds: "local" = this box (docker); "tritus"-style ROCm boxes (podman, AMD) are listed in ROCM_HOSTS; any other
# name is an ssh host with docker + NVIDIA (e.g. a DGX Spark). Remote boxes keep the same layout under ~/poptok and
# need the models under ~/models (MOSS-Image-SFT for keyframes, ltx-2.5 + ltx-2.3 for video, infinitetalk/ for
# lip-sync). ROCm boxes run all three: MOSS (localhost/chronoedit-rocm), LTX (localhost/ltx2-rocm, docker/ltx-rocm) and
# InfiniteTalk (localhost/infinitetalk-rocm, docker/infinitetalk-rocm), with MIOpen/Triton caches in ~/.cache.

ROCM_HOSTS=${ROCM_HOSTS:-tritus}
MAC_HOSTS=${MAC_HOSTS:-quintus}          # Apple silicon: native uv venv (.venv-ltx, bin/setup_mac_renderer.sh) under
                                         # ~/expts/poptok, not containers (Metal isn't reachable from one); no flock on
                                         # macOS, so the GPU lock is a directory: ~/.poptok-gpu.lock.d
LOCK='${TMPDIR:-/tmp}/poptok-gpu.lock'
# A side job on a box the Ice Walk farm also uses: put "$SIDE" just before its flock (remote: inside the ssh command; local:
# eval it in a subshell). It marks the job waiting (/tmp/poptok-gpu.waiting/<its shell's pid>) until that shell exits,
# and the farm's workers claim no new chunk on that box while a marked job's pid is alive, so the job waits at most for
# the chunk in progress (flock is not first-come-first-served: D12 waited 68 min behind back-to-back chunks, 2026-10-28)
SIDE='mkdir -p /tmp/poptok-gpu.waiting; touch /tmp/poptok-gpu.waiting/$$; trap "rm -f /tmp/poptok-gpu.waiting/$$" EXIT;'
BUSY_RE='ltx2|morph-spark|zoommorph|m2v-pre|acestep|chronoedit-rocm|infinitetalk'

is_rocm() { [[ " $ROCM_HOSTS " == *" $1 "* ]]; }
is_mac() { [[ " $MAC_HOSTS " == *" $1 "* ]]; }
on() { local h=$1; shift; if [ "$h" = local ]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "$*"; fi; }

host_free() {
  on "$1" "flock -n $LOCK true && ! { docker ps --format '{{.Image}}' 2>/dev/null; podman ps --format '{{.Image}}' 2>/dev/null; } | grep -qE '$BUSY_RE'"
}

pick_host() {   # first free host in the list; waits until one is
  while true; do
    for h in "$@"; do host_free "$h" 2>/dev/null && { echo "$h"; return; }; done
    sleep 30
  done
}

# the keyframe container on a remote host (run from ~/poptok)
keyframe_ctr() {
  if is_rocm "$1"; then         # fast attention is off by default on this GPU; offload is pointless with shared memory
    echo "podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
      -e HOME=/tmp -e OFFLOAD=0 -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 -v \$HOME/poptok:/work \
      -v \$HOME/models:/models -w /work localhost/chronoedit-rocm"
  else
    echo "docker run --gpus all --rm --user \$(id -u):\$(id -g) --ipc=host -e HOME=/tmp -e OFFLOAD=0 \
      -v \$HOME/poptok:/work -v \$HOME/models:/models -w /work morph-spark"
  fi
}

# run before a ROCm container (outside the lock): the kernel cache folders it mounts must exist
PREP="mkdir -p ~/.cache/ltx-rocm/miopen ~/.cache/infinitetalk-rocm/miopen 2>/dev/null;"
# the GPU containers running on a host (docker and/or podman), for "wait until no pipeline container is running"
CTRS="{ docker ps --format '{{.Image}}'; podman ps --format '{{.Image}}'; } 2>/dev/null"

# ROCm: device access, fast attention on (off by default on gfx1151), kernel caches kept between runs (the first LTX
# run on an empty cache spends ~19 min tuning MIOpen)
rocm_run() {   # image, cache name
  echo "podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups \
    --security-opt label=disable --ipc=host -e HOME=/tmp -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    -e MIOPEN_USER_DB_PATH=/cache/miopen -e MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen -e TRITON_CACHE_DIR=/cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -v \$HOME/.cache/$2:/cache"
}

# the LTX video container on a remote host (run from ~/poptok)
ltx_ctr() {
  if is_rocm "$1"; then
    echo "$(rocm_run localhost/ltx2-rocm ltx-rocm) -e LTX_MODELS=/models -v \$HOME/poptok:/work -v \$HOME/models:/models:ro \
      -w /work localhost/ltx2-rocm"
  else
    echo "docker run --gpus all --rm --user \$(id -u):\$(id -g) --ipc=host -e HOME=/tmp -v \$HOME/poptok:/work \
      -v \$HOME/models:/models -w /work ltx2"
  fi
}

# the InfiniteTalk container on a remote host (run from the insert folder: /work = \$PWD)
talk_ctr() {
  if is_rocm "$1"; then
    echo "$(rocm_run localhost/infinitetalk-rocm infinitetalk-rocm) -v \$HOME/models/infinitetalk:/weights:ro -v \$PWD:/work \
      -w /work localhost/infinitetalk-rocm"
  else
    echo "docker run --gpus all --rm --user \$(id -u):\$(id -g) --ipc=host -e HOME=/tmp -v \$HOME/models/infinitetalk:/weights \
      -v \$PWD:/work -w /work infinitetalk"
  fi
}

# lip-sync speed per host, seconds of compute per second of video (docs/hardware-performance.md), for scheduling
talk_rate() { if [ "$1" = gruntus ] || [ "$1" = local ]; then echo 62; elif is_rocm "$1"; then echo 348; else echo 163; fi; }
