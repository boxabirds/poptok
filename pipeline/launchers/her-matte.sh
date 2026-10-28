#!/usr/bin/env bash
# D7 (approved 2026-10-28): a clean cut-out of her A-pose photo (BiRefNet, matte.py) on thing1, then TRELLIS.2 again
# (her-mesh.sh) and a new turntable. Queued behind the Ice Walk dry run on thing1 by the GPU lock.
# Out: pieces/_bench/review/2026-10-28/her-mesh-2-turntable.png, her-matte-2.png
#   setsid nohup bash pipeline/launchers/her-matte.sh >> pieces/_logs/her-matte.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
source bin/hosts.sh
H=thing1; D=pieces/_bench/world/her3d; R=pieces/_bench/review/2026-10-28
rsync -a pipeline/world/matte.py $H:poptok/pipeline/world/ && rsync -a $D/front_2.png $H:poptok/$D/
CTR="docker run --gpus all --rm --user \$(id -u):\$(id -g) --ipc=host -e HOME=/tmp -e HF_HOME=/models/hf -v \$HOME/poptok:/work -v \$HOME/models:/models -w /work trellis2"
echo "[$(date +%H:%M)] BiRefNet weights (public, MIT)"
ssh $H "$CTR python3 -c \"from huggingface_hub import snapshot_download; print(snapshot_download('ZhengPeng7/BiRefNet'))\"" 2>&1 | tail -1
echo "[$(date +%H:%M)] matte (waits for the GPU lock)"
ssh $H "cd ~/poptok && flock $LOCK $CTR python3 pipeline/world/matte.py $D/front_2.png $D/front_2_matte.png" 2>&1 | grep -E "matte ok|Error" | tail -2
rsync -a $H:poptok/$D/front_2_matte.png $D/ || { echo "matte FAILED"; exit 1; }
python3 - <<EOF || { echo "matte not clean: stop"; exit 1; }
from PIL import Image; import numpy as np, sys
a = np.asarray(Image.open("$D/front_2_matte.png"))[..., 3]
c = np.mean([a[:16, :16].mean(), a[:16, -16:].mean(), a[-16:, :16].mean(), a[-16:, -16:].mean()])
print("transparent %.2f, corner alpha %.0f" % ((a < 8).mean(), c)); sys.exit(0 if c < 10 and (a < 8).mean() > 0.4 else 1)
EOF
python3 -c "
from PIL import Image
im = Image.open('$D/front_2_matte.png'); bg = Image.new('RGBA', im.size, (255, 0, 255, 255)); bg.alpha_composite(im)
bg.convert('RGB').save('$R/her-matte-2.png')"                     # on magenta, so any leftover background shows
IMG=$D/front_2_matte.png OUT=$D/mesh2 TAG=-2 bash pipeline/launchers/her-mesh.sh
echo "[$(date +%H:%M)] her matte done"
