#!/usr/bin/env bash
# poptok/lean step 4: the video. Keyframes (the user's picks) -> render_prep (shot list, motion from the directors' beats,
# an action-word transition into every keyframe) -> poptok's LTX chain (pipeline/ltx_chain.py: one continuous take, each
# change guided to its keyframe with the Transition LoRA) on a free Spark -> each keyframe where Amber sings lip-synced
# by InfiniteTalk on gruntus and laid over its span -> the song. 24 fps.
#   MAX_CLIPS=2 bash lean/render.sh pieces/cuckoo      a dry run: the first 2 clips, the cut to the review folder
#   setsid nohup bash lean/render.sh pieces/cuckoo >> pieces/_logs/lean-render.log 2>&1 < /dev/null &
#      -> PIECE/outputs/v<N>/<name>.mp4 (the next free N)
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
P=${1:?piece}; P=${P%/}; NAME=${P##*/}; O=$P/lean/render
SONG=$P/$(python3 -c "import sys;sys.path.insert(0,'pipeline');from workshop import read_env;print(read_env('$P/piece.env')['SONG'])")
VOC=$(ls $P/analysis/*-vocals.mp3 | head -1)
echo "[$(date +%H:%M)] keyframes (picks)"; bash lean/storyboard.sh $P | tail -1
echo "[$(date +%H:%M)] render prep (transitions)"; WORKSHOP_PY=lean/render_prep.py bin/workshop.sh $P | grep -E "into|render_prep|rror"
[ -f $O/scenes.json ] || { echo "FAILED: no render prep"; exit 1; }
mkdir -p $O/ltx_chain
H=$(pick_host thing1 thing2); echo "[$(date +%H:%M)] LTX chain on $H${MAX_CLIPS:+ (first $MAX_CLIPS clips)}"
ssh $H "mkdir -p ~/poptok/$O" && rsync -a pipeline $H:poptok/ && rsync -a --exclude ltx_chain $O/ $H:poptok/$O/ &&
  rsync -a $O/ltx_chain/ $H:poptok/$O/ltx_chain/
ssh $H "cd ~/poptok && flock $LOCK $(ltx_ctr $H) python pipeline/ltx_chain.py $O --plan shots.json ${MAX_CLIPS:+--max-clips $MAX_CLIPS}" \
  > $O/ltx.log 2>&1; grep -E "^clip|rror|Traceback" $O/ltx.log | tail -15
rsync -a $H:poptok/$O/ltx_chain/ $O/ltx_chain/
NF=$(ls $O/ltx_chain | grep -c png); echo "[$(date +%H:%M)] $NF frames"
# the sung keyframes: InfiniteTalk from the keyframe and the vocal over its span (25 fps out)
mkdir -p $O/sing
python3 - <<EOF > $O/sing/plan.txt
import json
for s in json.load(open("$O/shots.json"))["shots"]:
    if s["amber"] == "sings":
        print(s["i"], s["start"], s["end"])
EOF
while read -r J T0 T1; do
  [ -s $O/sing/it_$J.mp4 ] && continue
  case " ${MAX_CLIPS:+x} " in " x ") [ "$J" -ge "$MAX_CLIPS" ] && continue;; esac
  D=$(python3 -c "print(round($T1 - $T0, 3))")
  # InfiniteTalk needs >= 81 frames (3.24 s) of audio: a shorter line is sung over a window around it, padded equally,
  # and only its own span is used (cuckoo's "Cuckoo" is 2.5 s; the first try failed on it)
  PAD=$(python3 -c "print(round(max(0, (3.6 - $D) / 2), 3))"); W0=$(python3 -c "print(max(0, round($T0 - $PAD, 3)))")
  WD=$(python3 -c "print(round($D + 2 * $PAD, 3))"); echo "$J $W0" > $O/sing/window_$J.txt
  cp $O/keys/shot_$(printf %03d $J).png $O/sing/k$J.png
  ffmpeg -v error -y -ss $W0 -t $WD -i $VOC -ac 1 -ar 16000 $O/sing/v$J.wav
  python3 -c "
import json; json.dump({'prompt': 'a young woman with long, tousled, wavy dark-blonde hair sings with feeling, lips forming the words, small natural head movements. ' + json.load(open('$O/shots.json'))['style'] + '.',
 'cond_video': '/work/k$J.png', 'cond_audio': {'person1': '/work/v$J.wav'}}, open('$O/sing/it_$J.json', 'w'))"
  echo "[$(date +%H:%M)] keyframe $((J + 1)) sings: InfiniteTalk on gruntus (${D} s)"
  MAX_FRAMES=$(python3 -c "import math;print(math.ceil($WD*25)+25)") bin/infinitetalk_quant.sh $O/sing it_$J.json it_$J \
    > $O/sing/it_$J.log 2>&1 || echo "  keyframe $((J + 1)) lip-sync FAILED (see $O/sing/it_$J.log)"
done < $O/sing/plan.txt
# the cut: the chain's frames at 24 fps, each sung clip over its span, the song
IN=(-framerate 24 -i $O/ltx_chain/%05d.png); FC="[0:v]null[v0]"; k=0
while read -r J T0 T1; do
  [ -s $O/sing/it_$J.mp4 ] || continue
  IN+=(-i $O/sing/it_$J.mp4); k=$((k + 1))
  W0=$(cut -d' ' -f2 $O/sing/window_$J.txt 2>/dev/null || echo $T0)      # the clip starts at its window's start
  FC="$FC;[$k:v]fps=24,scale=1024:576,setsar=1,setpts=PTS-STARTPTS+$W0/TB[s$k];[v$((k - 1))][s$k]overlay=enable='between(t,$T0,$T1)':eof_action=pass[v$k]"
done < $O/sing/plan.txt
if [ -n "${MAX_CLIPS:-}" ]; then OUT=pieces/_bench/review/$(date +%F)/lean-$NAME/render-dry-run.mp4
else N=${OUTV:-}; [ -n "$N" ] || { N=1; while [ -e $P/outputs/v$N ]; do N=$((N + 1)); done; }   # OUTV=2: redo v2
  mkdir -p $P/outputs/v$N; OUT=$P/outputs/v$N/$NAME.mp4; fi
ffmpeg -v error -y "${IN[@]}" -i $SONG -filter_complex "$FC" -map "[v$k]" -map $((k + 1)):a -c:v libx264 -crf 18 \
  -pix_fmt yuv420p -c:a aac -b:a 256k -shortest $OUT && echo "[$(date +%H:%M)] render done: $OUT"
