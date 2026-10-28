#!/usr/bin/env bash
# D15 (approved 2026-10-28): does a second Blender on gruntus raise total output? The 4090 idles ~half of each ~12 s
# frame (CPU stages: scene/snow update, CPU denoise, save), so a second render could fill those gaps. Measured, not
# assumed (rule: one GPU job per box unless measured throughput gains): song3's chunk alone for 6 min (frames/min), then
# a second render of 24 frames alongside it, no lock; over the window from its first frame to its last, both renders'
# frames/min together vs song3's alone. Gain >= 30%: the farm moves to two workers per NVIDIA box (next run).
#   setsid nohup bash pipeline/launchers/ice-pipe-test.sh >> pieces/_logs/ice-pipe-test.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
O=pieces/ice-walk/song; S=$O/snowbake/blendcache_snow; W=$O/song3; T=pieces/_bench/ice-pipe-test
ARGS=$(sed -n "s/^ARGS='\(.*\)'$/\1/p" pieces/_logs/ice-song-farm.sh)
k=$(grep -oE "gruntus chunk [0-9]+ \(" pieces/_logs/ice-song3-chain.log | tail -1 | grep -oE "[0-9]+")
C=$W/c$k/render; rm -rf "${T:?}"; mkdir -p $T
n() { ls $1 2>/dev/null | wc -l; }
echo "[$(date +%T)] song3 on gruntus: chunk $k"
a0=$(n $C); t0=$(date +%s); sleep 360; a1=$(n $C); t1=$(date +%s)
alone=$(python3 -c "print(f'{($a1 - $a0) * 60 / ($t1 - $t0):.2f}')")
echo "[$(date +%T)] alone: $((a1 - a0)) frames in $(( (t1 - t0) / 60 )) min = $alone frames/min"
ICE="pipeline/world/ice_world.py -- $O/curves.json $T $ARGS --layers $O/layers.json --snow-cache $S --range 4801,4824"
eval "docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --user $(id -u):$(id -g) -v $PWD:/work -w /work \
  blender52 -b -P $ICE" > $T/blender.log 2>&1 &
P=$!
until [ "$(n $T/render)" -ge 1 ] || ! kill -0 $P 2>/dev/null; do sleep 2; done
b0=$(n $C); t0=$(date +%s); x0=$(n $T/render)
wait $P
b1=$(n $C); t1=$(date +%s); x1=$(n $T/render)
grep -E "ice ok|rror|Traceback" $T/blender.log | tail -2
python3 - <<PY
song, test, m = $b1 - $b0, $x1 - $x0, ($t1 - $t0) / 60
both = (song + test) / m
print(f"together: song3 {song} + test {test} frames in {m:.1f} min = {both:.2f} frames/min (song3 alone {$alone}); "
      f"gain {100 * (both / $alone - 1):+.0f}%; song3's own rate {song / m:.2f}")
PY
echo "[$(date +%T)] pipe test done"
