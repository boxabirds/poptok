# Where GPU steps run, and how. Sourced by make_video.sh.
#
# Every box has one poptok GPU lock (${TMPDIR:-/tmp}/poptok-gpu.lock). A step holds it for its whole run, so each box
# runs one GPU job at a time across all poptok runs. A box counts as free when its lock is free AND no GPU container
# (ours or anything else's) is running there.
#
# Host kinds: "local" = this box (docker); "tritus"-style ROCm boxes (podman, AMD) are listed in ROCM_HOSTS; any other
# name is an ssh host with docker + NVIDIA (e.g. a DGX Spark). Remote boxes keep the same layout under ~/poptok and
# need the models under ~/models (MOSS-Image-SFT for keyframes, ltx-2.5 + ltx-2.3 for video).

ROCM_HOSTS=${ROCM_HOSTS:-tritus}
MAC_HOSTS=${MAC_HOSTS:-quintus}          # Apple silicon: native uv venv (.venv-ltx, bin/setup_mac_renderer.sh) under
                                         # ~/expts/poptok, not containers (Metal isn't reachable from one); no flock on
                                         # macOS, so the GPU lock is a directory: ~/.poptok-gpu.lock.d
LOCK='${TMPDIR:-/tmp}/poptok-gpu.lock'
BUSY_RE='ltx2|morph-spark|zoommorph|m2v-pre|acestep|chronoedit-rocm'

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
