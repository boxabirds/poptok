#!/usr/bin/env bash
# D39: D38 test 2's sweep, filled in. Keyframes on a free Spark (lean/poc_sweep2.py: 80-95% style, and her in full
# charcoal by face pass or from her photo), then each sings the same 4.6 s of cuckoo with InfiniteTalk int8 on
# gruntus; lip-sync (+-1 s window) and her face through each clip; one sheet of every row (D38's too) and two videos.
#   setsid nohup bash lean/poc_sweep2.sh >> pieces/_logs/lean-poc.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
R=pieces/_bench/review/2026-10-29/lean-poc; O=pieces/_bench/lean/sweep; P=pieces/artists/r.amber.jpeg
IDS=${IDS:-"080 085 090 095 f100 s100 p090 p095"}   # D39 ran IDS="095 f100 s100": the rest were photographs (no clip needed)
mkdir -p $R
if [ ! -f $O/levels2.json ]; then
  H=$(pick_host thing1 thing2); echo "[$(date +%H:%M)] D39 keyframes on $H"
  ssh $H "mkdir -p ~/poptok/$O ~/poptok/pieces/artists" && rsync -a pipeline lean $H:poptok/ &&
    rsync -a pieces/artists/ $H:poptok/pieces/artists/ && rsync -a $O/ $H:poptok/$O/
  ssh $H "cd ~/poptok && flock $LOCK $(keyframe_ctr $H) python3 lean/poc_sweep2.py $O $O/style.txt $P" 2>&1 \
    | grep -E "^k|levels2|rror|Traceback"
  rsync -a $H:poptok/$O/ $O/
fi
[ -f $O/levels2.json ] || { echo "FAILED: no D39 keyframes"; exit 1; }
STY=$(cat $O/style.txt)
for l in $IDS; do
  [ -s $O/it_$l.mp4 ] && continue
  python3 -c "
import json; s='''$STY'''
p = 'a young woman with long straight light-blonde hair sings with feeling, lips forming the words, small natural head movements.'
json.dump({'prompt': p + f' {s}.', 'cond_video': '/work/k$l.png', 'cond_audio': {'person1': '/work/vocal.wav'}}, open('$O/it_$l.json', 'w'))"
  echo "[$(date +%H:%M)] $l lip-sync on gruntus"
  MAX_FRAMES=140 bin/infinitetalk_quant.sh $O it_$l.json it_$l > $O/it_$l.log 2>&1 || echo "  $l FAILED (see $O/it_$l.log)"
done
for l in 000 010 025 050 075 100 $IDS; do
  [ -s $O/it_$l.mp4 ] || continue
  python3 pipeline/eval_lipsync.py $O/it_$l.mp4 $O/vocal.wav --window 1 > $O/sync_$l.txt 2>&1
  [ -f $O/id_$l.json ] || python3 pipeline/eval_identity.py $P $O/it_$l.mp4 --every 10 --json $O/id_$l.json > /dev/null 2>&1
done
python3 lean/poc_sweep_sheet.py $O $R 39-sweep 000,075,095,100,f100,s100=39-identity
echo "[$(date +%H:%M)] D39 done: $R/39-sweep.jpg, 39-cliff.mp4, 39-identity.mp4"
