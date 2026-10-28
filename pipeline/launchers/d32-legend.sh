#!/usr/bin/env bash
# D32 (approved 2026-10-28): the Ice Walk layer legend. Each layer's best frame from the cached world, rendered with the
# layer on and off on gruntus; the sheet crops and zooms on the difference (pipeline/world/layer_legend.py).
#   ONLY=swifts setsid nohup bash pipeline/launchers/d32-legend.sh >> pieces/_logs/d32-legend.log 2>&1 < /dev/null &
# ONLY: the layers to (re)render (default all); the sheet always shows every layer rendered so far.
set -uo pipefail
cd ~/sambashare/expts/poptok
O=pieces/ice-walk/song; S=$O/snowbake/blendcache_snow; L=pieces/_bench/legend/ice-walk
R=pieces/_bench/review/2026-10-28; ONLY=${ONLY:-swifts,chime,drone,pad,subbass,clicks,mist,heartbeat}
# the world the farm would render, built (and cached) by the farm itself, so the legend shows the song's own world
WB=$(TAG=legend BOXES=gruntus BUILD_ONLY=1 bash pipeline/launchers/ice-song-farm.sh 2>&1 | tee /dev/stderr | sed -n 's/^world: //p')
rm -rf "${O:?}/legend"
[ -f "$WB" ] || { echo "FAILED: no world"; exit 1; }
mkdir -p $L
echo "[$(date +%H:%M)] legend $ONLY from $WB"
flock /tmp/poptok-gpu.lock docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --user $(id -u):$(id -g) \
  -v $PWD:/work -w /work blender52 -b -P pipeline/world/layer_legend.py -- render $WB $L $O/curves.json $O/layers.json \
  --gpu --snow-cache $S --only $ONLY 2>&1 | grep -E "^legend|rror|Traceback|line [0-9]" | grep -v HIPEW
grep -q . $L/picks.json 2>/dev/null || { echo "FAILED: nothing rendered"; exit 1; }
python3 pipeline/world/layer_legend.py sheet $L $R/ice-walk-layer-legend.jpg
echo "[$(date +%H:%M)] d32 done"
