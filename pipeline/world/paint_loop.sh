#!/usr/bin/env bash
# Paint a blockout view by view (2026-10-26): paint_world.py picks and renders the next view here (Blender container),
# fill_view.py paints it on a Spark (Qwen-Image-Edit), until no view paints more than 1% of the room; then the shot
# cameras and a ring of views are rendered with the finished texture (tex_<cam>.mp4, texviews/).
#   pipeline/world/paint_loop.sh OUT [HOST] [MAX_VIEWS]     (OUT holds room.blend and style.json)
set -euo pipefail
cd "$(dirname "$0")/../.."
O=$1; H=${2:-thing1}; MAX=${3:-16}
source bin/hosts.sh
blender() { docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work blender -b $O/room.blend \
  -P pipeline/world/paint_world.py -- $O "$@" 2>&1 | grep -vE '^Fra:|Saved:|Time:|^ +Time|^$|Read blend|Blender 4'; }
wait_free="while $CTRS | grep -qE '$BUSY_RE'; do sleep 20; done"
ssh $H "mkdir -p ~/poptok/$O/paint"; rsync -a pipeline $H:poptok/; rsync -a $O/style.json $H:poptok/$O/
mkdir -p $O/paint; [ -e $O/paint/views.json ] || echo '[]' > $O/paint/views.json
for i in $(seq 1 $MAX); do
  echo "[$(date +%H:%M)] next view"
  blender next | tee -a $O/paint/blender.log | grep -E '^(painted|paintable|next|done)' || true
  grep -q '^done' <(tail -n 3 $O/paint/blender.log) && break
  D=$(ls -d $O/paint/v[0-9]* | tail -n 1)
  [ -s $D/styled.png ] && { echo "view $D already painted?"; exit 1; }
  rsync -a $D $H:poptok/$O/paint/
  ssh $H "cd ~/poptok && $wait_free; flock $LOCK $(keyframe_ctr $H) python3 pipeline/world/fill_view.py $D $O/style.json" \
    >> $O/paint/fill.log 2>&1 || { echo "fill failed: $O/paint/fill.log"; exit 1; }
  rsync -a $H:poptok/$D/ $D/
  python3 -c "import json; v = json.load(open('$O/paint/views.json')); v.append(json.load(open('$D/view.json'))); \
json.dump(v, open('$O/paint/views.json', 'w'), indent=1)"
done
echo "[$(date +%H:%M)] render"
blender render | tee -a $O/paint/blender.log | grep -E '^painted' || true
for d in $O/tex_*/; do n=$(basename $d); ffmpeg -v error -y -framerate 24 -i $d/%04d.png -c:v libx264 -pix_fmt yuv420p -crf 12 $O/$n.mp4; done
echo "[$(date +%H:%M)] paint done"
