#!/usr/bin/env bash
# D22 (approved 2026-10-28): after song3, the farm with the world built once (cached .blend, ice_render.py per chunk),
# D16's two workers per NVIDIA box and D20's smaller tail chunks. First every box renders clip frames 2401 and 4400 from
# the built world and each must match song3's frames (rendered per chunk the old way) to the cross-box noise level;
# then dry3: song frames 2401-2820 through every stage, with frames/min per box, setup per chunk and the finish spread.
# Before it, the pieces/_logs copies of the scripts that were running (D13) become pointers to pipeline/launchers.
# Out: pieces/_bench/review/2026-10-28/ice-walk-dry3.mp4; pieces/_logs/ice-dry3.log
#   setsid nohup bash pipeline/launchers/ice-d22-dry.sh >> pieces/_logs/ice-d22-dry.log 2>&1 < /dev/null &
cd ~/sambashare/expts/poptok
while pgrep -f "bash (pieces/_logs/(ice-song3-chain|ice-song-farm)|pipeline/launchers/ice-song3-boost)\.sh" > /dev/null; do sleep 60; done
grep -q "ice song3 done" pieces/_logs/ice-song3-chain.log || { echo "$(date +%H:%M) song3 did not finish: no dry run"; exit 1; }
for s in ice-song-farm ice-song3-chain d12-identity; do
  printf '#!/usr/bin/env bash\n# moved to pipeline/launchers/%s.sh (under git, 2026-10-28)\nexec bash ~/sambashare/expts/poptok/pipeline/launchers/%s.sh "$@"\n' $s $s > pieces/_logs/$s.sh
done
F="START=2401 FRAMES=420 CH=60"; W=pieces/ice-walk/song/dry3
echo "$(date +%H:%M) song3 done; the world built once and checked on every box"
env TAG=dry3 $F CHECK_FRAMES=2401,4400 bash pipeline/launchers/ice-song-farm.sh || { echo "box check FAILED"; exit 1; }
python3 - $W <<'EOF' || { echo "$(date +%H:%M) a box does not match song3: no dry run"; exit 1; }
import glob, sys, numpy as np
from PIL import Image
W, bad = sys.argv[1], 0
for f in (2401, 4400):
    ref = np.asarray(Image.open(glob.glob(f"pieces/ice-walk/song/song3/c*/render/{f:04d}.png")[0]).convert("RGB"), float)
    for p in sorted(glob.glob(f"{W}/check_*/render/{f:04d}.png")):
        d = np.abs(np.asarray(Image.open(p).convert("RGB"), float) - ref).mean()
        print(f"frame {f} {p.split('/')[-3][6:]}: mean diff from song3 {d:.3f}/255"); bad += d > 0.15
n = len(glob.glob(f"{W}/check_*/render/*.png"))
print(f"{n} check frames"); sys.exit(1 if bad or n != 8 else 0)
EOF
echo "$(date +%H:%M) dry3 starts"
env TAG=dry3 $F bash pipeline/launchers/ice-song-farm.sh > pieces/_logs/ice-dry3.log 2>&1
grep -E "world|preflight|finish spread|^setup|ice dry3 done|FAILED|missing" pieces/_logs/ice-dry3.log
grep -oE "^[0-9:]+ [a-z0-9]+\.[0-9] chunk [0-9]+ done.*" pieces/_logs/ice-dry3.log | tail -4
