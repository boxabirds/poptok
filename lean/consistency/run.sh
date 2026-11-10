#!/usr/bin/env bash
# One-look test (lean/consistency/run.py) on cuckoo: method C on thing1, D on thing2, in parallel; results fetched back.
#   setsid nohup bash lean/consistency/run.sh >> pieces/_logs/consistency.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
O=pieces/_bench/lean/consistency
for H in thing1 thing2; do
  ssh $H "mkdir -p ~/poptok/$O ~/poptok/pieces/cuckoo ~/poptok/pieces/artists" && rsync -a pipeline lean $H:poptok/ &&
    rsync -a pieces/artists/ $H:poptok/pieces/artists/ && rsync -a pieces/cuckoo/piece.env pieces/cuckoo/lean $H:poptok/pieces/cuckoo/
done
echo "[$(date +%H:%M)] C on thing1, D on thing2"
ssh thing1 "cd ~/poptok && flock $LOCK $(keyframe_ctr thing1) python3 lean/consistency/run.py pieces/cuckoo $O C" 2>&1 | grep -E "^C |rror|Traceback" &
ssh thing2 "cd ~/poptok && flock $LOCK $(keyframe_ctr thing2) python3 lean/consistency/run.py pieces/cuckoo $O D" 2>&1 | grep -E "^D |rror|Traceback" &
wait
for H in thing1 thing2; do rsync -a $H:poptok/$O/ $O/; done
echo "[$(date +%H:%M)] consistency test done: $O"
