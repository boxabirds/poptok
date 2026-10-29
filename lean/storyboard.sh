#!/usr/bin/env bash
# poptok/lean: the storyboard as pictures, to review the video's shape before any clip is made. The visualiser (local
# LLM on gruntus) turns the directors' beats into pictures, MOSS draws them on a free Spark (face pass where Amber is
# in one), and one sheet shows them all in order.
#   setsid nohup bash lean/storyboard.sh pieces/cuckoo >> pieces/_logs/lean-storyboard.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
P=${1:?piece}; P=${P%/}; R=pieces/_bench/review/$(date +%F)/lean-${P##*/}
[ -f $P/lean/storyboard.json ] || { echo "no storyboard: WORKSHOP_PY=lean/directors.py bin/workshop.sh $P"; exit 1; }
mkdir -p $R
if [ ! -f $P/lean/shots.json ]; then
  echo "[$(date +%H:%M)] visualiser"; WORKSHOP_PY=lean/visualiser.py bin/workshop.sh $P
fi
[ -f $P/lean/shots.json ] || { echo "FAILED: no shots"; exit 1; }
H=$(pick_host thing1 thing2); echo "[$(date +%H:%M)] keyframes on $H"
ssh $H "mkdir -p ~/poptok/$P ~/poptok/pieces/artists" && rsync -a pipeline lean $H:poptok/ &&
  rsync -a pieces/artists/ $H:poptok/pieces/artists/ && rsync -a --exclude analysis --exclude outputs $P/ $H:poptok/$P/
ssh $H "cd ~/poptok && flock $LOCK $(keyframe_ctr $H) python3 lean/keyframes.py $P" 2>&1 | grep -E "keyframe|rror|Traceback"
rsync -a $H:poptok/$P/lean/keyframes/ $P/lean/keyframes/
python3 lean/storyboard_sheet.py $P $R/storyboard.jpg
echo "[$(date +%H:%M)] storyboard done: $R/storyboard.jpg"
