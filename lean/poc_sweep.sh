#!/usr/bin/env bash
# D38 test 2: the style-intensity sweep. Keyframes on a free Spark (lean/poc_sweep.py), then each level sings the same
# 4.6 s of cuckoo ("Cuckoo / Why can't / I come home too", its vocal stem) with InfiniteTalk int8 on gruntus, then per
# level: lip-sync r (eval_lipsync.py), her face through the clip (eval_identity.py), a labelled sheet and one video of
# all six levels with the song.
#   STYLE_FILE=... setsid nohup bash lean/poc_sweep.sh >> pieces/_logs/lean-poc.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
R=pieces/_bench/review/2026-10-28/lean-poc; O=pieces/_bench/lean/sweep; P=pieces/artists/r.amber.jpeg
A=pieces/cuckoo/analysis; T0=54.0; DUR=4.6; LEVELS="0 10 25 50 75 100"
STYLE_FILE=${STYLE_FILE:?the winning wording from test 1}
mkdir -p $O $R; cp "$STYLE_FILE" $O/style.txt
if [ ! -f $O/levels.json ]; then
  H=$(pick_host thing1 thing2); echo "[$(date +%H:%M)] test 2 keyframes on $H"
  ssh $H "mkdir -p ~/poptok/$O ~/poptok/pieces/artists" && rsync -a pipeline lean $H:poptok/ &&
    rsync -a pieces/artists/ $H:poptok/pieces/artists/ && rsync -a $O/ $H:poptok/$O/
  ssh $H "cd ~/poptok && flock $LOCK $(keyframe_ctr $H) python3 lean/poc_sweep.py $O $O/style.txt $P" 2>&1 \
    | grep -E "reference|level|rror|Traceback"
  rsync -a $H:poptok/$O/ $O/
fi
[ -f $O/levels.json ] || { echo "FAILED: no keyframes"; exit 1; }
ffmpeg -v error -y -ss $T0 -t $DUR -i $A/Cuckoo-vocals.mp3 -ac 1 -ar 16000 $O/vocal.wav
ffmpeg -v error -y -ss $T0 -t $DUR -i pieces/cuckoo/inputs/Cuckoo.wav $O/mix.wav
STY=$(cat $O/style.txt)
for L in $LEVELS; do
  l=$(printf %03d $L); [ -s $O/it_$l.mp4 ] && continue
  cp $O/l$l.png $O/k$l.png
  python3 -c "
import json; s='''$STY'''
p = 'a young woman with long straight light-blonde hair sings with feeling, lips forming the words, small natural head movements.'
json.dump({'prompt': p + (f' {s}.' if $L else ''), 'cond_video': '/work/k$l.png', 'cond_audio': {'person1': '/work/vocal.wav'}}, open('$O/it_$l.json', 'w'))"
  echo "[$(date +%H:%M)] level $L% lip-sync on gruntus"
  MAX_FRAMES=$(python3 -c "import math;print(math.ceil($DUR*25)+25)") bin/infinitetalk_quant.sh $O it_$l.json it_$l > $O/it_$l.log 2>&1 \
    || echo "  level $L% FAILED (see $O/it_$l.log)"
done
for L in $LEVELS; do
  l=$(printf %03d $L); [ -s $O/it_$l.mp4 ] || continue
  python3 pipeline/eval_lipsync.py $O/it_$l.mp4 $O/vocal.wav > $O/sync_$l.txt 2>&1
  python3 pipeline/eval_identity.py $P $O/it_$l.mp4 --every 10 --json $O/id_$l.json > /dev/null 2>&1
done
python3 lean/poc_sweep_sheet.py $O $R
echo "[$(date +%H:%M)] test 2 done: $R/2-sweep.jpg, $R/2-sweep.mp4"
