#!/usr/bin/env bash
# D16 (approved 2026-10-28): after song3, dry-run the farm with two workers per NVIDIA box (PER, flock -s): song frames
# 2401-2820 in 7 chunks of 60 (one per worker), through every stage, with frames/min per box in the log. Before it, the
# pieces/_logs copies of the scripts that were running (D13) become pointers to pipeline/launchers.
# Out: pieces/_bench/review/2026-10-28/ice-walk-dry3.mp4; per-box rates in pieces/_logs/ice-dry3.log
#   setsid nohup bash pipeline/launchers/ice-d16-dry.sh >> pieces/_logs/ice-d16-dry.log 2>&1 < /dev/null &
cd ~/sambashare/expts/poptok
while pgrep -f "bash pieces/_logs/(ice-song3-chain|ice-song-farm|d12-identity)\.sh" > /dev/null; do sleep 60; done
grep -q "ice song3 done" pieces/_logs/ice-song3-chain.log || { echo "$(date +%H:%M) song3 did not finish: no dry run"; exit 1; }
for s in ice-song-farm ice-song3-chain d12-identity; do
  printf '#!/usr/bin/env bash\n# moved to pipeline/launchers/%s.sh (under git, 2026-10-28)\nexec bash ~/sambashare/expts/poptok/pipeline/launchers/%s.sh "$@"\n' $s $s > pieces/_logs/$s.sh
done
echo "$(date +%H:%M) song3 done; dry3 (two workers per NVIDIA box) starts"
TAG=dry3 START=2401 FRAMES=420 CH=60 bash pipeline/launchers/ice-song-farm.sh > pieces/_logs/ice-dry3.log 2>&1
tail -2 pieces/_logs/ice-dry3.log
grep -oE "^[0-9:]+ [a-z0-9]+\.[0-9] chunk [0-9]+ done.*" pieces/_logs/ice-dry3.log
