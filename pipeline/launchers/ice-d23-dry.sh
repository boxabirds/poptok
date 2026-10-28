#!/usr/bin/env bash
# D23 (approved 2026-10-28): after dry3 (D22: the world built once, a Blender per chunk), dry4: the same frames through
# the farm with one Blender per worker that stays up with the world open (ice_serve.py), farm pinned to the D23 commit.
# Then frames/min per box, setup per chunk and the finish spread, dry3 against dry4.
# Out: pieces/_bench/review/2026-10-28/ice-walk-dry4.mp4; pieces/_logs/ice-dry4.log
#   setsid nohup bash pipeline/launchers/ice-d23-dry.sh >> pieces/_logs/ice-d23-dry.log 2>&1 < /dev/null &
cd ~/sambashare/expts/poptok
FARM=pieces/_logs/ice-song-farm.d23.sh; git show e45cb4d:pipeline/launchers/ice-song-farm.sh > $FARM   # the D23 commit
while pgrep -f "bash pipeline/launchers/ice-d22-dry\.sh" > /dev/null; do sleep 60; done
grep -q "ice dry3 done" pieces/_logs/ice-dry3.log || { echo "$(date +%H:%M) dry3 did not finish: no dry4"; exit 1; }
echo "$(date +%H:%M) dry4 starts"
env TAG=dry4 START=2401 FRAMES=420 CH=60 bash $FARM > pieces/_logs/ice-dry4.log 2>&1
for d in dry3 dry4; do
  echo "== $d ($( [ $d = dry3 ] && echo 'D22: a Blender per chunk' || echo 'D23: a Blender per worker'))"
  grep -E "world|finish spread|^setup|ice $d done|FAILED|missing|did not start" pieces/_logs/ice-$d.log
  for h in gruntus thing1 thing2 quintus; do grep -E "^[0-9:]+ $h\.[0-9] chunk [0-9]+ done" pieces/_logs/ice-$d.log | tail -1 | grep -oE "$h so far: .*"; done
  echo "first claim $(grep -m1 -oE "^[0-9:]+ [a-z0-9]+\.[0-9] chunk 0 " pieces/_logs/ice-$d.log | cut -d' ' -f1), last finish $(grep -E "^[0-9:]+ [a-z0-9]+\.[0-9] finished" pieces/_logs/ice-$d.log | tail -1 | cut -d' ' -f1)"
done
