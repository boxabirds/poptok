#!/usr/bin/env bash
# Face-pass benchmark (D45): 10 fixed cases (5 treatments x 2 positions, cases.json) through the face pass's regions
# (head = current, whole, nohair), every try kept, one sheet. Draw on a Spark, hair masks on gruntus, then the cases
# split over both Sparks.
#   setsid nohup bash lean/facebench/run.sh >> pieces/_logs/facebench.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
O=pieces/_bench/lean/facebench; R=pieces/_bench/review/$(date +%F)/facebench; P=pieces/artists/r.amber.jpeg
mkdir -p $O $R
sync_to() { ssh $1 "mkdir -p ~/poptok/$O ~/poptok/pieces/artists" && rsync -a pipeline lean $1:poptok/ &&
  rsync -a pieces/artists/ $1:poptok/pieces/artists/ && rsync -a $O/ $1:poptok/$O/; }
if [ $(ls $O/*.png 2>/dev/null | grep -vc -- -) -lt 10 ]; then
  H=$(pick_host thing1 thing2); echo "[$(date +%H:%M)] drawing the cases on $H"; sync_to $H
  ssh $H "cd ~/poptok && flock $LOCK $(keyframe_ctr $H) python3 lean/facebench/draw.py $O" 2>&1 | grep -E "drawn|rror|Traceback"
  rsync -a $H:poptok/$O/ $O/
fi
echo "[$(date +%H:%M)] hair masks on gruntus"
flock $LOCK docker run --gpus all --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/work -v $HOME/models/segment:/models \
  -w /work segment python3 lean/facebench/hair.py $O 2>&1 | grep -E "hair|rror|Traceback"
for H in thing1 thing2; do sync_to $H; done
echo "[$(date +%H:%M)] face passes on thing1 (c1-o2, g1) and thing2 (g2-p2)"
ssh thing1 "cd ~/poptok && flock $LOCK $(keyframe_ctr thing1) python3 lean/facebench/run.py $O $P c1 c2 o1 o2 g1" 2>&1 \
  | grep -E "head|whole|nohair|rror|Traceback" &
ssh thing2 "cd ~/poptok && flock $LOCK $(keyframe_ctr thing2) python3 lean/facebench/run.py $O $P g2 w1 w2 p1 p2" 2>&1 \
  | grep -E "head|whole|nohair|rror|Traceback" &
wait
for H in thing1 thing2; do rsync -a $H:poptok/$O/ $O/; done
python3 lean/facebench/sheet.py $O $R
echo "[$(date +%H:%M)] facebench done: $R"
