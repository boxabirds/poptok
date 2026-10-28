#!/usr/bin/env bash
# TRELLIS.2 avatar test (decision D4, approved 2026-10-28): her A-pose photo (front_2_rgba.png, background removed) ->
# a textured GLB on thing1 (DGX Spark: TRELLIS.2-4B needs more than the 4090's 24 GB without offload, which we never do),
# under its GPU lock, peak memory sampled; then an 8-view turntable rendered on gruntus (blender52) to judge it.
# Out: pieces/_bench/review/2026-10-28/her-mesh-turntable.png, her.glb next to it.
#   setsid nohup bash pipeline/launchers/her-mesh.sh >> pieces/_logs/her-mesh.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
H=thing1; I=${IMG:-pieces/_bench/world/her3d/front_2_rgba.png}; O=${OUT:-pieces/_bench/world/her3d/mesh1}; R=pieces/_bench/review/2026-10-28
V=${TAG:-}                              # review names: her-mesh$V-turntable.png, her-mesh$V.glb
mkdir -p $O $R
ssh $H "mkdir -p ~/poptok/$O ~/poptok/pipeline/world" && rsync -a pipeline/world/her_mesh.py $H:poptok/pipeline/world/ && rsync -a $I $H:poptok/$I
echo "[$(date +%H:%M)] her mesh on $H"
# memory: the Spark's GPU shares system memory, so the peak of used system memory over the run, sampled every 5 s
ssh $H "nohup bash -c 'b=\$(free -m | awk \"/Mem:/{print \\\$3}\"); p=0; while [ ! -f /tmp/her-mesh.end ]; do u=\$(free -m | awk \"/Mem:/{print \\\$3}\"); [ \$u -gt \$p ] && p=\$u; sleep 5; done; echo \"base \$b MB peak \$p MB\" > /tmp/her-mesh.mem' >/dev/null 2>&1 & rm -f /tmp/her-mesh.end /tmp/her-mesh.mem"
t0=$(date +%s)
ssh $H "cd ~/poptok && while $CTRS | grep -qE '$BUSY_RE'; do sleep 20; done; flock $LOCK docker run --gpus all --rm --user \$(id -u):\$(id -g) --ipc=host \
  -e HOME=/tmp -e HF_HOME=/models/hf -e HF_HUB_OFFLINE=1 -v \$HOME/poptok:/work -v \$HOME/models:/models -w /work trellis2 \
  python3 pipeline/world/her_mesh.py $I $O --seed 1" > $O/her_mesh.log 2>&1
rc=$?; ssh $H "touch /tmp/her-mesh.end"; sleep 7
echo "  exit $rc, $(( ($(date +%s) - t0) / 60 )) min, memory: $(ssh $H 'cat /tmp/her-mesh.mem 2>/dev/null')"
[ $rc -eq 0 ] || { echo "her mesh FAILED: $(grep -E 'Error|error' $O/her_mesh.log | tail -2)"; exit 1; }
rsync -a $H:poptok/$O/her.glb $O/
flock /tmp/poptok-gpu.lock docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --user $(id -u):$(id -g) -v $PWD:/work -w /work \
  blender52 -b -P pipeline/world/turntable.py -- $O/her.glb $O/turntable 2>&1 | grep -E "mesh bounds|turntable ok|rror"
python3 - <<EOF
from PIL import Image, ImageDraw
t = [Image.open(f"$O/turntable/view{k}.png").convert("RGB") for k in range(1, 9)]
s = Image.new("RGB", (t[0].width * 8, t[0].height))
for k, im in enumerate(t):
    d = ImageDraw.Draw(im); d.rectangle((0, 0, 120, 22), fill=0); d.text((5, 5), f"{k + 1}  {45 * k} deg", fill=(255, 255, 0))
    s.paste(im, (im.width * k, 0))
s.save("$R/her-mesh$V-turntable.png")
EOF
[ -s $R/her-mesh$V-turntable.png ] && cp $O/her.glb $R/her-mesh$V.glb
echo "[$(date +%H:%M)] her mesh done"
