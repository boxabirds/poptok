#!/usr/bin/env bash
# A progress video of a farm run so far (user 2026-10-28: "create a progress video of the first 9 chunks done so far"):
# the done chunks from the start, contiguous, with the music and the song frame number burnt in; no title card.
#   TAG=song3 setsid nohup bash pipeline/launchers/ice-progress.sh >> pieces/_logs/ice-progress.log 2>&1 < /dev/null &
set -euo pipefail
cd ~/sambashare/expts/poptok
O=pieces/ice-walk/song; TAG=${TAG:-song3}; W=$O/$TAG; R=${R:-pieces/_bench/review/2026-10-28}; T=$W/progress
rm -rf "${T:?}"; mkdir -p $T
k=0; n=0
while [ -f $W/c$k/done ]; do                            # contiguous done chunks from frame 1 (chunk files: song frames)
  for f in $W/c$k/render/*.png; do n=$((n + 1)); ln -s "$PWD/$f" $T/$(printf %04d $n).png; done
  k=$((k + 1))
done
[ $n -gt 0 ] || { echo "no done chunks"; exit 1; }
OUT=$R/ice-walk-$TAG-progress-$k-chunks.mp4
nice ffmpeg -v error -y -framerate 24 -start_number 1 -i $T/%04d.png -i "pieces/ice-walk/Ice walk.mp3" \
  -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='$TAG progress  song f %{eif\:n+1\:d}':x=12:y=12:fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.6" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -shortest $OUT
echo "$(date +%H:%M) progress: $k chunks, $n frames ($(python3 -c "print(f'{$n/24:.0f}')") s): $OUT"
