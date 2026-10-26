#!/usr/bin/env bash
# The universe step (pipeline/universe.py): candidates and picks with the local LLM here (under this box's GPU lock),
# then the stills of each pick on a Spark (AMBER_HOSTS; Qwen-Image-Edit with her photo), then the sheet.
#   bin/universe.sh pieces/NAME [--force]
set -euo pipefail
cd "$(dirname "$0")/.."
PIECE=$1; shift
source bin/hosts.sh
WORKSHOP_PY=pipeline/universe.py bin/workshop.sh "$PIECE" propose "$@"
rm -rf "$PIECE/inputs/universes"/pick*.png                       # stills of the new picks, not of earlier ones
H=$(pick_host ${AMBER_HOSTS:-thing1 thing2})
echo "[$(date +%H:%M:%S)] universe stills on $H"
ssh "$H" "mkdir -p ~/poptok/$PIECE/inputs ~/poptok/pieces/artists"
rsync -a pipeline "$H":poptok/; rsync -a pieces/artists/ "$H":poptok/pieces/artists/
rsync -a "$PIECE/piece.env" "$H":poptok/"$PIECE"/; rsync -a --delete "$PIECE/inputs/" "$H":poptok/"$PIECE"/inputs/
ssh "$H" "cd ~/poptok && while $CTRS | grep -qE '$BUSY_RE'; do sleep 20; done; \
  flock $LOCK $(keyframe_ctr "$H") python3 pipeline/universe.py $PIECE render" > "$PIECE/universe_render.log" 2>&1 \
  || { echo "universe stills failed: $PIECE/universe_render.log"; exit 1; }
rsync -a "$H":poptok/"$PIECE"/inputs/universes/ "$PIECE/inputs/universes/"
python3 pipeline/universe.py "$PIECE" sheet
