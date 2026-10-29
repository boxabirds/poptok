#!/usr/bin/env bash
# poptok/macpass: macvideo's pipeline (plan -> chained keyframes -> one clip per shot -> plain cuts) with Qwen for the
# pictures and the words, LTX-2.5 on the Sparks for the clips, 1920x1088 at 24 fps.
#   STEP=keys bash macpass/run.sh pieces/cuckoo       plan + keyframes + the sheet (review before any clip)
#   STEP=keys LAST=6 bash macpass/run.sh ...           the first 6 keyframes only (a test)
#   ONLY=1,2,3 bash macpass/run.sh pieces/cuckoo      + those clips only, cut to the review folder (a dry run)
#   bash macpass/run.sh pieces/cuckoo                 everything -> PIECE/outputs/v<N>/<name>.mp4
#   (detached: setsid nohup ... >> pieces/_logs/macpass.log 2>&1 < /dev/null &)
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
P=${1:?piece}; P=${P%/}; NAME=${P##*/}; O=$P/macpass; R=pieces/_bench/review/$(date +%F)/macpass-$NAME; mkdir -p $R
SONG=$P/$(python3 -c "import sys;sys.path.insert(0,'pipeline');from workshop import read_env;print(read_env('$P/piece.env')['SONG'])")
[ -f $O/image_prompts.json ] && [ "$(python3 -c "import json;print(len(json.load(open('$O/image_prompts.json'))))")" = \
  "$(python3 -c "import json;print(len(json.load(open('$O/shots.json'))['shots']))")" ] ||
  { echo "[$(date +%H:%M)] plan"; WORKSHOP_PY=macpass/plan.py bin/workshop.sh $P | grep -E "plan|rror"; }
sync_to() { ssh $1 "mkdir -p ~/poptok/$P ~/poptok/pieces/artists" && rsync -a pipeline macpass $1:poptok/ &&
  rsync -a pieces/artists/ $1:poptok/pieces/artists/ && rsync -a $P/piece.env $1:poptok/$P/ && rsync -a $O/ $1:poptok/$O/; }
H=$(pick_host thing1 thing2); echo "[$(date +%H:%M)] keyframes on $H"; sync_to $H
ssh $H "cd ~/poptok && flock $LOCK $(keyframe_ctr $H) python3 macpass/keys.py $P ${LAST:-}" 2>&1 | grep -E "keyframe|rror|Traceback"
rsync -a $H:poptok/$O/keyframes/ $O/keyframes/
python3 macpass/sheet.py $P $R/keyframes.jpg
[ "${STEP:-}" = keys ] && exit 0
# the clips: shared across both Sparks, each its own half of the shots
echo "[$(date +%H:%M)] clips${ONLY:+ $ONLY}"
for k in 0 1; do H=$( [ $k = 0 ] && echo thing1 || echo thing2 ); sync_to $H
  ssh $H "cd ~/poptok && flock $LOCK $(ltx_ctr $H) python macpass/clips.py $P --part $k/2 ${ONLY:+--only $ONLY}" \
    > $O/clips_$k.log 2>&1 &
done; wait
for H in thing1 thing2; do rsync -a $H:poptok/$O/clips/ $O/clips/; done
grep -hE "^clip|rror|Traceback" $O/clips_*.log | tail -30
# the cut (macvideo's): each clip trimmed to its shot's span, plain cuts, the song
python3 - $O "${ONLY:-}" > $O/concat.txt <<'PY'
import json, os, sys
O, only = sys.argv[1], {int(x) for x in sys.argv[2].split(",") if x}
for s in json.load(open(f"{O}/shots.json"))["shots"]:
    f = f"{O}/clips/clip_{s['index']:03d}.mp4"
    if (not only or s["index"] in only) and os.path.exists(f):
        print(f"file '{os.path.abspath(f)}'\noutpoint {s['target_duration_s']}")
PY
if [ -n "${ONLY:-}" ]; then OUT=$R/dry-run.mp4; SS=$(python3 -c "import json;print(json.load(open('$O/shots.json'))['shots'][min(int(x) for x in '$ONLY'.split(','))-1]['start_s'])")
else N=${OUTV:-}; [ -n "$N" ] || { N=1; while [ -e $P/outputs/v$N ]; do N=$((N + 1)); done; }
  mkdir -p $P/outputs/v$N; OUT=$P/outputs/v$N/$NAME.mp4; SS=0; fi
ffmpeg -v error -y -f concat -safe 0 -i $O/concat.txt -ss $SS -i $SONG -map 0:v -map 1:a -c:v libx264 -crf 16 -r 24 \
  -pix_fmt yuv420p -c:a aac -b:a 256k -shortest $OUT && echo "[$(date +%H:%M)] done: $OUT"
