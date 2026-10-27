#!/usr/bin/env bash
# Top and tail a piece (user, 2026-10-27, Ice Walk: "Black with big text ... Then fade to black. music starts, fading
# into the video matching fade-in of 5s, End: slow fade to black of 5s as well"):
#   title card: 1 s fade in from black, 3 s hold, 1 s fade out, 1 s black, silent
#   then the video and the music fade in together over 5 s, and both fade out over the last 5 s.
#   bash pipeline/world/top_tail.sh TITLE.png VIDEO.mp4 OUT.mp4 [FADE=5]
set -eu
T=$1; V=$2; OUT=$3; F=${4:-5}
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$V")
E=$(python3 -c "print(max(0.0, $D - $F))")
ffmpeg -v error -y -loop 1 -framerate 24 -t 6 -i "$T" -f lavfi -t 6 -i anullsrc=r=48000:cl=stereo -i "$V" -filter_complex \
  "[0]scale=1920:1080,format=yuv420p,fade=in:st=0:d=1,fade=out:st=4:d=1,setsar=1[t];
   [2:v]format=yuv420p,fade=in:st=0:d=$F,fade=out:st=$E:d=$F,setsar=1[v];
   [2:a]aresample=48000,afade=in:st=0:d=$F,afade=out:st=$E:d=$F[a];
   [t][1:a][v][a]concat=n=2:v=1:a=1[vo][ao]" \
  -map "[vo]" -map "[ao]" -r 24 -c:v libx264 -pix_fmt yuv420p -crf 16 -c:a aac -b:a 192k "$OUT"
echo "$OUT: 6 s title + $D s, fades $F s"
