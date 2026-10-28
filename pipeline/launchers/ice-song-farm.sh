#!/usr/bin/env bash
# Ice Walk, the whole song, POC 13's settings, rendered across four boxes on the same Blender (5.2.0) and the same CPU
# denoiser (OIDN 2.5.0), checked to give the same pixels (mean difference 0.03-0.08/255): gruntus (4090, docker
# blender52), thing1 + thing2 (DGX Spark GB10, docker blender52 = the community aarch64 build + OIDN 2.5.0 built for
# aarch64), quintus (M5 Max, Blender 5.2.0.app). User 2026-10-27: "I have 2 petaflops of processing power and want to use
# it"; "top and tail it".
# Chunks from a queue: each box's worker claims the next chunk (atomic mkdir on gruntus), renders it on its own box from
# the one snow bake (--snow-cache, so chunks join seamlessly), copies the frames back and marks it done; a failed chunk
# is released for another box. A preflight on every box must show the layers, the sun and the range before any chunk.
# Dry run first, same script, short clip (user 2026-10-27: "sort out the basic pipeline with a short clip before wasting
# so much compute time"):
#   TAG=dry START=2401 FRAMES=240 CH=60 bash pipeline/launchers/ice-song-farm.sh   (START: past frame 999, where the snow
#   froze on 2026-10-28; a dry run from frame 1 never reached it)
# The song (detached, resumable: re-launch after a crash or reboot):
#   cd ~/sambashare/expts/poptok && setsid nohup bash pipeline/launchers/ice-song-farm.sh >> pieces/_logs/ice-song-farm.log 2>&1 < /dev/null &
set -u
cd ~/sambashare/expts/poptok
O=pieces/ice-walk/song; R=${R:-pieces/_bench/review/2026-10-28}; S=$O/snowbake/blendcache_snow; RO=poptok-render/$O
TAG=${TAG:-song}; W=$O/$TAG; RW=poptok-render/$W        # W: this run's chunks, claims, frames (RW: on a remote box)
NS=$(python3 -c "import json;print(len(json.load(open('$O/curves.json'))['energy']))")   # the song's frames
S0=${START:-1}; N=${FRAMES:-$(( NS - S0 + 1 ))}; E=$(( S0 + N - 1 ))      # this run: song frames S0..E
CH=${CH:-240}; K=$(( (N + CH - 1) / CH ))
OUTMP4=$R/ice-walk$([ "$TAG" = song ] || echo "-$TAG").mp4
BOXES=${BOXES:-"gruntus thing1 thing2 quintus"}
# ONE line: a newline here ended the command (eval / the remote shell) and a run rendered without the sun or the layers
ARGS='--res 1920x1080 --samples 48 --gpu --theme icechrome --snow 15000 --swifts --mist 0.05 --flake-glow 6 --mist-color 0.7,0.82,1.0 --sun-el 16 --sky 1.0,0.15,0.4 --exposure -1.0 --look AgX\ -\ Punchy --sun-color 1.0,0.93,0.85 --sun-energy 3,3 --ice-absorb 1.6 --crack-glow 80 --chime-glow 40 --halo 6 --drift 1.0 --cloud-sweep 100'
[ -d "$S" ] || { echo "no snow bake at $S"; exit 1; }
rm -rf "${W:?}/claims"; mkdir -p $W/claims              # claims without a done chunk are stale after a restart

count() {                                               # frames in a render dir on box $1 (macOS wc pads with spaces)
  if [ $1 = gruntus ]; then ls $2 2>/dev/null | wc -l | tr -d ' '
  else ssh -o BatchMode=yes $1 "ls ~/poptok-render/$2 2>/dev/null | wc -l" | tr -d ' '; fi
}

sync_inputs() {                                         # the script, curves, layers and snow bake to a remote box
  ssh -o BatchMode=yes $1 "mkdir -p ~/$RO/snowbake ~/poptok-render/pipeline/world" &&
  rsync -a pipeline/world/ice_world.py $1:poptok-render/pipeline/world/ &&
  rsync -a $O/curves.json $O/layers.json $1:$RO/ && rsync -a --delete $S $1:$RO/snowbake/
}

render() {                                              # render clip frames A..B into dir C on box H
  local H=$1 C=$2 A=$3 B=$4
  local ICE="pipeline/world/ice_world.py -- $O/curves.json $C $ARGS --layers $O/layers.json --snow-cache $S --range $A,$B"
  case $H in
    gruntus) eval "flock /tmp/poptok-gpu.lock docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
               --user $(id -u):$(id -g) -v $PWD:/work -w /work blender52 -b -P $ICE" ;;      # eval: the look's spaces
    thing1|thing2) ssh -o BatchMode=yes $H "cd ~/poptok-render && flock /tmp/poptok-gpu.lock docker run --rm --gpus all \
               -v ~/poptok-blender/cmake-make:/opt/b:ro -v ~/poptok-blender/oidn-lib:/opt/oidn:ro \
               -e LD_LIBRARY_PATH=/opt/oidn:/opt/b/libExt:/opt/b/lib --user \$(id -u):\$(id -g) -v \$PWD:/work -w /work \
               blender52 -b -P $ICE" ;;
    quintus) ssh -o BatchMode=yes $H "cd ~/poptok-render && ~/poptok-blender/Blender52.app/Contents/MacOS/Blender -b -P $ICE" ;;
  esac 2>&1 | grep -E "ice ok|ice check|rror|Traceback" | grep -v HIPEW
}

side_waiting() {                                        # a marked side job ($SIDE, bin/hosts.sh) is waiting on box $1
  # answers on stdout, not the exit status: ssh to quintus always exits 0 (checked 2026-10-28: ssh quintus 'exit 3' -> 0)
  local c='for f in /tmp/poptok-gpu.waiting/*; do [ -e "$f" ] && kill -0 "${f##*/}" 2>/dev/null && echo waiting && break; done'
  if [ $1 = gruntus ]; then bash -c "$c"; else ssh -o BatchMode=yes $1 "bash -c '$c'"; fi | grep -q waiting
}

worker() {                                              # claim chunks until none are left
  local H=$1 k A B C n fails=0
  for ((k=0; k<K; k++)); do
    C=$W/c$k; [ -f "$C/done" ] && continue
    while side_waiting $H; do sleep 20; done            # D14: a side job on this box goes before the next chunk
    mkdir $W/claims/c$k 2>/dev/null || continue         # another box has it
    A=$(( S0 + k * CH )); B=$(( S0 + (k + 1) * CH - 1 )); (( B > E )) && B=$E
    echo "$(date +%H:%M) $H chunk $k ($A-$B) start"
    render $H $C $A $B
    if [ $H != gruntus ]; then mkdir -p $C/render && rsync -a $H:$RW/c$k/render/ $C/render/; fi
    n=$(count gruntus $C/render)
    if [ "$n" -eq $(( B - A + 1 )) ]; then
      touch $C/done; echo "$(date +%H:%M) $H chunk $k done"
      [ $H != gruntus ] && ssh -o BatchMode=yes $H "rm -rf ~/$RW/c$k"
    else
      echo "$(date +%H:%M) $H chunk $k FAILED ($n frames); released"; rmdir $W/claims/c$k
      fails=$((fails + 1)); (( fails >= 2 )) && { echo "$(date +%H:%M) $H gives up after 2 failures"; return; }
    fi
  done
  echo "$(date +%H:%M) $H finished"
}

echo "$(date +%H:%M) farm $TAG start: song frames $S0-$E ($N), $K chunks of $CH, boxes: $BOXES"
for H in $BOXES; do [ $H = gruntus ] || sync_inputs $H || { echo "$H: input sync FAILED"; exit 1; }; done
# preflight: every box builds the full scene and renders 2 frames; its report must show the layers, the sun and the
# range, and exactly 2 frames must come out, or nothing starts
for H in $BOXES; do (render $H $W/pre_$H 2400 2401 > $W/pre_$H.log; echo "frames $(count $H $W/pre_$H/render)" >> $W/pre_$H.log) & done
wait
for H in $BOXES; do
  grep -q "range 2400 2401 ; layers: halo=1 clouds=1 drift=1 chime=[1-9][0-9]* crack=[1-9][0-9]* swift_bird=105 ; sun_el 16.0 look AgX - Punchy" $W/pre_$H.log \
    && grep -qx "frames 2" $W/pre_$H.log \
    && python3 - $W/pre_$H.log <<'EOF' || { echo "$(date +%H:%M) preflight FAILED on $H:"; cat $W/pre_$H.log; exit 1; }
import re, sys
t = open(sys.argv[1]).read()
s = re.search(r"snow m/s ([\d.]+) ([\d.]+)", t); w = re.search(r"swifts in frame (\d+)/(\d+)", t)
ok = s and min(float(s[1]), float(s[2])) > 1.0 and w and (int(w[2]) == 0 or int(w[1]) >= 0.7 * int(w[2]))
sys.exit(0 if ok else 1)            # the snow moving at frame 2400 (it froze past 999), the swifts in frame
EOF
done
echo "$(date +%H:%M) preflight ok on: $BOXES"
for H in $BOXES; do worker $H & done
wait
for ((k=0; k<K; k++)); do [ -f $W/c$k/done ] || { echo "chunk $k missing; re-launch to finish it"; exit 1; }; done
dirs=(); for ((k=0; k<K; k++)); do dirs+=("$W/c$k/render"); done
python3 pipeline/world/stitch_chunks.py $W/frames $N $CH 0 --start $S0 "${dirs[@]}"     # frames 1..N = song frames S0..E
ffmpeg -v error -y -framerate 24 -start_number 1 -i $W/frames/%04d.png -ss $(python3 -c "print(($S0 - 1) / 24)") -i "pieces/ice-walk/Ice walk.mp3" \
  -c:v libx264 -pix_fmt yuv420p -crf 16 -c:a aac -b:a 192k -shortest $W/body.mp4
bash pipeline/world/top_tail.sh pieces/ice-walk/title/title.png $W/body.mp4 $OUTMP4
python3 pipeline/world/layer_sheet.py $W/frames $O/curves.json $O/layers.json ${OUTMP4%.mp4}-layers.jpg --tag $TAG --start $S0
echo "$(date +%H:%M) ice $TAG done: $OUTMP4"
