#!/usr/bin/env bash
# Ice Walk, the whole song, POC 13's settings, rendered across four boxes on the same Blender (5.2.0) and the same CPU
# denoiser (OIDN 2.5.0), checked to give the same pixels (mean difference 0.03-0.08/255): gruntus (4090, docker
# blender52), thing1 + thing2 (DGX Spark GB10, docker blender52 = the community aarch64 build + OIDN 2.5.0 built for
# aarch64), quintus (M5 Max, Blender 5.2.0.app). User 2026-10-27: "I have 2 petaflops of processing power and want to use
# it"; "top and tail it".
# Chunks from a queue: each box's worker claims the next chunk (atomic mkdir on gruntus), renders it on its own box from
# the one snow bake (--snow-cache, so chunks join seamlessly), copies the frames back and marks it done; a failed chunk
# is released for another box. A preflight on every box must show the layers, the sun and the range before any chunk.
# D22 (2026-10-28): the world is built once per run, not per chunk (each chunk spent 4.3 min rebuilding the same world,
# 197 s of it in bpy.ops hummocks): ice_world.py --build-only on gruntus into $O/worldcache/<content hash>/, reused while
# the script, curves, layers, settings and Blender version are unchanged; every box renders its ranges from that .blend
# with ice_render.py.
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
OUTMP4=$R/ice-walk$([ "$TAG" = song ] || echo "-$TAG").mp4
BOXES=${BOXES:-"gruntus thing1 thing2 quintus"}
# D16 (2026-10-28): chunk workers per box. Two Blenders on the 4090 gave +30% frames/min together (5.0 -> 6.5,
# ice-pipe-test.log): one renders on the GPU while the other is in its CPU stages (scene/snow update, CPU denoise, save).
# The farm's renders share each box's GPU lock (flock -s), so a side job's exclusive flock still waits for both, and the
# D14 mark stops new claims while it waits. quintus: 1 (its GPU use can't be measured without sudo).
PER=${PER:-"gruntus:2 thing1:2 thing2:2 quintus:1"}
per() { local w; for w in $PER; do [ ${w%%:*} = $1 ] && { echo ${w#*:}; return; }; done; echo 1; }
# D20 (2026-10-28): the chunks get smaller at the end. With 240-frame chunks throughout, song3's last three went to the
# Sparks at 14:51-15:48 and gruntus and quintus sat idle from 16:13 while they finished. Now the body is chunks of CH,
# then the last TAILR rounds of every worker are chunks of CT (CH/4), so the last workers finish about one small chunk
# apart. Chunk k is song frames CA[k]..CB[k]; the plan is kept in $W/plan, and a re-launch must find the same plan.
CH=${CH:-240}; CT=${CT:-$(( CH / 4 ))}; TAILR=${TAILR:-2}
NW=0; for H in $BOXES; do NW=$(( NW + $(per $H) )); done
NB=$(( (N - TAILR * NW * CT) / CH )); (( NB < 0 )) && NB=0  # body chunks of CH; the rest of the frames in chunks of CT
CA=(); CB=(); a=$S0
while (( a <= E )); do
  c=$(( ${#CA[@]} < NB ? CH : CT )); b=$(( a + c - 1 )); (( b > E )) && b=$E
  CA+=($a); CB+=($b); a=$(( b + 1 ))
done
K=${#CA[@]}
# ONE line: a newline here ended the command (eval / the remote shell) and a run rendered without the sun or the layers
ARGS='--res 1920x1080 --samples 48 --gpu --theme icechrome --snow 15000 --swifts --mist 0.05 --flake-glow 6 --mist-color 0.7,0.82,1.0 --sun-el 16 --sky 1.0,0.15,0.4 --exposure -1.0 --look AgX\ -\ Punchy --sun-color 1.0,0.93,0.85 --sun-energy 3,3 --ice-absorb 1.6 --crack-glow 80 --chime-glow 40 --halo 6 --drift 1.0 --cloud-sweep 100'
[ -d "$S" ] || { echo "no snow bake at $S"; exit 1; }
WORLD="$O/curves.json WC $ARGS --layers $O/layers.json --snow-cache $S"   # ice_world.py's arguments, WC: the output dir
BV=$(docker run --rm blender52 --version 2>/dev/null | head -1)
KEY=$( (cat pipeline/world/ice_world.py pipeline/world/ice_range.py $O/curves.json $O/layers.json; echo "$WORLD"; echo "$BV") | sha256sum | cut -c1-16)
WC=$O/worldcache/$KEY; WB=$WC/scene_render.blend         # the world, built once, every chunk's scene
if [ -f $WB ]; then echo "$(date +%H:%M) world cached: $WB"
else
  t=$(date +%s); rm -rf "${WC:?}.tmp"
  eval "docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work blender52 -b -P pipeline/world/ice_world.py -- ${WORLD/WC/$WC.tmp} --build-only" 2>&1 \
    | grep -E "ice built|rror|Traceback" | grep -v HIPEW
  [ -f $WC.tmp/scene_render.blend ] || { echo "world build FAILED"; exit 1; }
  mv $WC.tmp $WC; echo "$(date +%H:%M) world built in $(( $(date +%s) - t )) s: $WB ($BV)"
fi
rm -rf "${W:?}/claims"; mkdir -p $W/claims              # claims without a done chunk are stale after a restart
PLAN=$(for ((k=0; k<K; k++)); do echo "c$k ${CA[k]} ${CB[k]}"; done)
if [ -f $W/plan ]; then [ "$(cat $W/plan)" = "$PLAN" ] || { echo "$W/plan differs from this launch's chunks (CH, CT, TAILR, PER): finish it with the same settings or use a new TAG"; exit 1; }
else echo "$PLAN" > $W/plan; fi

count() {                                               # frames in a render dir on box $1 (macOS wc pads with spaces)
  if [ $1 = gruntus ]; then ls $2 2>/dev/null | wc -l | tr -d ' '
  else ssh -o BatchMode=yes $1 "ls ~/poptok-render/$2 2>/dev/null | wc -l" | tr -d ' '; fi
}

sync_inputs() {                                         # the script, curves, layers and snow bake to a remote box
  ssh -o BatchMode=yes $1 "mkdir -p ~/$RO/snowbake ~/poptok-render/pipeline/world ~/poptok-render/$WC" &&
  rsync -a pipeline/world/ice_world.py pipeline/world/ice_range.py pipeline/world/ice_render.py $1:poptok-render/pipeline/world/ &&
  rsync -a $O/curves.json $O/layers.json $1:$RO/ && rsync -a --delete $S $1:$RO/snowbake/ && rsync -a $WB $1:poptok-render/$WC/
}

render() {                                              # render clip frames A..B into dir C on box H
  local H=$1 C=$2 A=$3 B=$4
  local ICE="pipeline/world/ice_render.py -- $WB $C --range $A,$B --gpu --snow-cache $S${ONLY:+ --only $ONLY}"   # D22
  case $H in
    gruntus) eval "flock -s /tmp/poptok-gpu.lock docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
               --user $(id -u):$(id -g) -v $PWD:/work -w /work blender52 -b -P $ICE" ;;      # eval: the look's spaces
    thing1|thing2) ssh -o BatchMode=yes $H "cd ~/poptok-render && flock -s /tmp/poptok-gpu.lock docker run --rm --gpus all \
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

worker() {                                              # claim chunks until none are left (box, worker number)
  local H=$1 I=$2 k A B C n t fails=0
  for ((k=0; k<K; k++)); do
    C=$W/c$k; [ -f "$C/done" ] && continue
    while side_waiting $H; do sleep 20; done            # D14: a side job on this box goes before the next chunk
    mkdir $W/claims/c$k 2>/dev/null || continue         # another box has it
    echo "$(date +%s) $H.$I" > $W/claims/c$k/t          # when and who: the setup per chunk at the end
    A=${CA[k]}; B=${CB[k]}
    echo "$(date +%H:%M) $H.$I chunk $k ($A-$B) start"; t=$(date +%s)
    render $H $C $A $B
    if [ $H != gruntus ]; then mkdir -p $C/render && rsync -a $H:$RW/c$k/render/ $C/render/; fi
    n=$(count gruntus $C/render)
    if [ "$n" -eq $(( B - A + 1 )) ]; then
      touch $C/done; echo $n >> $W/rate_$H
      echo "$(date +%H:%M) $H.$I chunk $k done ($(( ($(date +%s) - t) / 60 )) min); $H so far: $(rate $H) frames/min"
      [ $H != gruntus ] && ssh -o BatchMode=yes $H "rm -rf ~/$RW/c$k"
    else
      echo "$(date +%H:%M) $H.$I chunk $k FAILED ($n frames); released"; rm -rf "${W:?}/claims/c$k"
      fails=$((fails + 1)); (( fails >= 2 )) && { echo "$(date +%H:%M) $H.$I gives up after 2 failures"; return; }
    fi
  done
  echo "$(date +%H:%M) $H.$I finished"; date +%s > $W/fin_$H.$I
}

rate() {                                                # box $1's frames/min since the workers started (all its workers)
  python3 -c "print(f'{sum(map(int, open(\"$W/rate_$1\").read().split())) * 60 / ($(date +%s) - $TW):.2f}')"
}

echo "$(date +%H:%M) farm $TAG start: song frames $S0-$E ($N), $K chunks ($NB of $CH, then $(( K - NB )) of $CT or less), workers: $PER"
for H in $BOXES; do [ $H = gruntus ] || sync_inputs $H || { echo "$H: input sync FAILED"; exit 1; }; done
[ -n "${BUILD_ONLY:-}" ] && { echo "world: $WB"; exit 0; }   # the world built and on every box, nothing rendered
if [ -n "${CHECK_FRAMES:-}" ]; then                     # these clip frames on every box from the built world, then stop
  for H in $BOXES; do
    ( ONLY=$CHECK_FRAMES render $H $W/check_$H ${CHECK_FRAMES%%,*} ${CHECK_FRAMES%%,*}
      [ $H = gruntus ] || { mkdir -p $W/check_$H/render && rsync -a $H:$RW/check_$H/render/ $W/check_$H/render/; } ) &
  done
  wait; echo "checked: $W/check_*/render (world $WB)"; exit 0
fi
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
rm -f $W/rate_* $W/fin_*; TW=$(date +%s)
for H in $BOXES; do for ((i=1; i<=$(per $H); i++)); do worker $H $i & sleep 5; done; done
wait
python3 - $W/fin_* <<'EOF'                               # D20: how far apart the workers finished
import os, sys
t = {os.path.basename(f)[4:]: int(open(f).read()) for f in sys.argv[1:]}
a, b = min(t, key=t.get), max(t, key=t.get)
print(f"finish spread: {(t[b] - t[a]) / 60:.0f} min (first {a}, last {b})")
EOF
python3 - $W $K <<'EOF'                                  # D22: claim to first frame, per chunk (the world load + checks)
import glob, os, statistics as S, sys
W, K = sys.argv[1], int(sys.argv[2]); by = {}
for k in range(K):
    try: t, who = open(f"{W}/claims/c{k}/t").read().split()
    except (OSError, ValueError): continue
    fs = [os.path.getmtime(f) for f in glob.glob(f"{W}/c{k}/render/*.png")]
    if fs: by.setdefault(who.split(".")[0], []).append(min(fs) - int(t))
for h, v in sorted(by.items()):
    print(f"setup {h}: median {S.median(v):.0f} s to the first frame over {len(v)} chunks (min {min(v):.0f}, max {max(v):.0f})")
EOF
for ((k=0; k<K; k++)); do [ -f $W/c$k/done ] || { echo "chunk $k missing; re-launch to finish it"; exit 1; }; done
dirs=(); for ((k=0; k<K; k++)); do dirs+=("$W/c$k/render"); done
python3 pipeline/world/stitch_chunks.py $W/frames $N $CH 0 --start $S0 "${dirs[@]}"     # frames 1..N = song frames S0..E; overlap 0: each frame from the chunk that has it
ffmpeg -v error -y -framerate 24 -start_number 1 -i $W/frames/%04d.png -ss $(python3 -c "print(($S0 - 1) / 24)") -i "pieces/ice-walk/Ice walk.mp3" \
  -c:v libx264 -pix_fmt yuv420p -crf 16 -c:a aac -b:a 192k -shortest $W/body.mp4
bash pipeline/world/top_tail.sh pieces/ice-walk/title/title.png $W/body.mp4 $OUTMP4
python3 pipeline/world/layer_sheet.py $W/frames $O/curves.json $O/layers.json ${OUTMP4%.mp4}-layers.jpg --tag $TAG --start $S0
echo "$(date +%H:%M) ice $TAG done: $OUTMP4"
