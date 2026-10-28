#!/usr/bin/env bash
# D10 (approved 2026-10-28): flakes self-lit at 6 and the world-space mist at 0.05 (ice-walk-d8-test.jpg row 3,
# ice-walk-d8-mist2.jpg row 3). Dry run first through the same farm (song frames 2401-2640, past frame 999, all four
# boxes); only if it finishes cleanly with 240 frames, the whole song (TAG=song3; song2's chunks had the old settings).
# Out: pieces/_bench/review/2026-10-28/ice-walk-dry2.mp4, then ice-walk-song3.mp4 (+ -layers.jpg)
#   setsid nohup bash pipeline/launchers/ice-song3-chain.sh >> pieces/_logs/ice-song3-chain.log 2>&1 < /dev/null &
cd ~/sambashare/expts/poptok
L=pieces/_logs/ice-dry2.log
TAG=dry2 START=2401 FRAMES=240 CH=60 bash pipeline/launchers/ice-song-farm.sh >> $L 2>&1
grep -q "ice dry2 done" $L && ! grep -qE "FAILED|gives up|missing" $L || { echo "$(date +%H:%M) dry run did not finish cleanly: song not started"; exit 1; }
n=$(ls pieces/ice-walk/song/dry2/frames | wc -l); [ "$n" -eq 240 ] || { echo "dry run has $n frames: song not started"; exit 1; }
echo "$(date +%H:%M) dry run clean ($n frames): song3 starts"
TAG=song3 bash pipeline/launchers/ice-song-farm.sh
