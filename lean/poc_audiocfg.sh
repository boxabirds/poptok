#!/usr/bin/env bash
# D41 test 1: does a stronger audio guidance shape the sung "I" (cuckoo, "why can't I", 2.54-2.80 s into the clip)
# instead of the "ooo" InfiniteTalk gave at 2.0? The s100 keyframe (D39) sung at AUDIO_CFG 3, 4, 6; lip-sync r and a
# mouth strip across "can't I come" per setting, and a 2x2 video (2.0 = D39's clip) with the song.
#   setsid nohup bash lean/poc_audiocfg.sh >> pieces/_logs/lean-poc.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
R=pieces/_bench/review/2026-10-29/lean-poc; O=pieces/_bench/lean/sweep
for G in 3 4 6; do
  [ -s $O/it_s100_g$G.mp4 ] && continue
  echo "[$(date +%H:%M)] s100 audio guidance $G on gruntus"
  AUDIO_CFG=$G MAX_FRAMES=140 bin/infinitetalk_quant.sh $O it_s100.json it_s100_g$G > $O/it_s100_g$G.log 2>&1 \
    || echo "  guidance $G FAILED (see $O/it_s100_g$G.log)"
done
for G in 3 4 6; do
  [ -s $O/it_s100_g$G.mp4 ] && python3 pipeline/eval_lipsync.py $O/it_s100_g$G.mp4 $O/vocal.wav --window 1
done
python3 lean/poc_mouth_strip.py $O $R/41-mouth-on-I.jpg s100=2.0 s100_g3=3 s100_g4=4 s100_g6=6
ffmpeg -v error -y $(for c in s100 s100_g3 s100_g4 s100_g6; do echo -i $O/it_$c.mp4; done) -i $O/mix.wav -filter_complex \
  "$(i=0; for g in 2 3 4 6; do echo -n "[$i:v]scale=640:360,setsar=1,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='$((i+1)) audio guidance $g':fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.6:x=8:y=8[v$i];"; i=$((i+1)); done)[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|640_0|0_360|640_360:shortest=1[o]" \
  -map "[o]" -map 4:a -c:v libx264 -crf 20 -c:a aac -shortest $R/41-audio-guidance.mp4
echo "[$(date +%H:%M)] D41 test 1 done: $R/41-mouth-on-I.jpg, $R/41-audio-guidance.mp4"
