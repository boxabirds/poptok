#!/usr/bin/env bash
# D5 (approved 2026-10-28): the directors pick the next R.Amber song for the procedural world and its look
# (pick_song.py, the LLM on gruntus under the GPU lock, between the song2 farm's chunks), then the song is prepared for
# layer analysis: curves (music_curves.py), Demucs stems of the whole song with the Ice Walk detectors as a first pass
# (music_layers.py), and an overview of every stem's spectrogram over the whole song, for picking out its layers.
# The layers themselves are then identified by hand from the overview (as for Ice Walk), before any POC.
# Out: pieces/_bench/review/2026-10-28/next-director.md, next-stems.png
#   setsid nohup bash pipeline/launchers/ice-next-pick.sh >> pieces/_logs/ice-next-pick.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/sambashare/expts/poptok
D=pieces/_bench/ice-next; R=pieces/_bench/review/2026-10-28; mkdir -p $D $R
LLM_SERVER=$HOME/.local/bin/llama-server
LLM_GGUF=$HOME/.local/share/qwen38-27b/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
if [ ! -s $D/director.json ]; then
  echo "[$(date +%H:%M)] directors (waits for gruntus's GPU lock)"
  ( exec 9>/tmp/poptok-gpu.lock; flock 9
    "$LLM_SERVER" -m "$LLM_GGUF" -ngl 99 -c 16384 --reasoning-budget 0 --host 127.0.0.1 --port 8091 > $D/llm_server.log 2>&1 9>&- &
    P=$!; for i in $(seq 1 120); do curl -sf http://127.0.0.1:8091/health > /dev/null && break; sleep 2; done
    python3 pipeline/world/pick_song.py $D; rc=$?
    kill $P; wait $P 2> /dev/null; exit $rc ) || { echo "pick FAILED"; exit 1; }
fi
cp $D/director.md $R/next-director.md
P=$(python3 -c "import json;print(json.load(open('$D/director.json'))['piece'])")
A=$(python3 -c "import json;print(json.load(open('$D/director.json'))['audio'])")
S=$(python3 -c "import json;print(json.load(open('$D/director.json'))['seconds'])")
O=pieces/$P-world/song; mkdir -p $O
echo "[$(date +%H:%M)] $P: curves"
python3 pipeline/world/music_curves.py "$A" $O/curves.json 2>&1 | tail -1
echo "[$(date +%H:%M)] $P: stems + first-pass layers (waits for the GPU lock)"
flock /tmp/poptok-gpu.lock docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -v ~/.cache/torch:/cache/torch \
  -e TORCH_HOME=/cache/torch -v $PWD:/work -w /work --entrypoint python m2v-pre pipeline/world/music_layers.py "$A" $O/layers.json \
  --start 0 --dur $S 2>&1 | grep -v "%|" | tail -2
[ -s $O/stems/htdemucs_6s/win/other.wav ] || { echo "stems FAILED"; exit 1; }
python3 - <<EOF
import numpy as np, soundfile as sf
from scipy.signal import stft
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
names = ["drums", "bass", "guitar", "piano", "vocals", "other"]
fig, ax = plt.subplots(len(names), 1, figsize=(26, 20), sharex=True)
for k, n in enumerate(names):
    x, sr = sf.read(f"$O/stems/htdemucs_6s/win/{n}.wav"); x = x.mean(1)
    f, t, Z = stft(x, sr, nperseg=4096, noverlap=4096 - 1024)
    ax[k].pcolormesh(t, f, 20 * np.log10(np.abs(Z) + 1e-7), vmin=-100, vmax=-25, shading="auto", cmap="gray_r")
    ax[k].set_yscale("symlog", linthresh=200); ax[k].set_ylim(30, 16000)
    ax[k].set_ylabel(f"{k + 1} {n}\nHz", fontsize=13)
ax[-1].set_xlabel("seconds"); ax[0].set_title("$P: Demucs stems over the whole song (rows 1-6)", loc="left", fontsize=16)
plt.tight_layout(); plt.savefig("$R/next-stems.png", dpi=60)
EOF
echo "[$(date +%H:%M)] next pick done: $P ($R/next-director.md, $R/next-stems.png)"
