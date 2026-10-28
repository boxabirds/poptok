#!/usr/bin/env bash
# D17 (approved 2026-10-28): second chunk workers for the running song3 farm (D16's two-per-box, applied midstream):
# one more on gruntus and thing1 now, on thing2 once D12 has finished (another render beside LTX could run its shared
# memory out). They claim from song3's own queue (atomic mkdir claims) with the running farm's own render/worker code
# (read from pieces/_logs/ice-song-farm.sh, not edited), without the GPU lock: the farm's worker holds it, and the two
# share the GPU (+30% measured, ice-pipe-test.log). If a boost worker still has the last chunk when the farm checks,
# the farm stops with "chunk N missing"; then, once the boost workers are done, the farm is re-run for song3: it
# skips every done chunk and only stitches (appended to ice-song3-chain.log, where the waiting D16 dry run looks).
#   setsid nohup bash pipeline/launchers/ice-song3-boost.sh >> pieces/_logs/ice-song3-boost.log 2>&1 < /dev/null &
cd ~/sambashare/expts/poptok
F=pieces/_logs/ice-song-farm.sh; TAG=song3
eval "$(sed -n '18,26p' $F)"                            # the run's variables and ARGS, as the farm has them
eval "$(sed -n '/^count()/,/^}/p; /^render()/,/^}/p; /^worker()/,/^}/p' $F | sed 's#flock /tmp/poptok-gpu.lock ##')"
type render | grep -q flock && { echo "render still takes the lock: stop"; exit 1; }
echo "$(date +%H:%M) boost song3: $K chunks; second workers on gruntus, thing1 (thing2 after D12)"
worker gruntus &
worker thing1 &
( while pgrep -f "d12-identity\.sh" > /dev/null; do sleep 30; done; echo "$(date +%H:%M) D12 finished: thing2 boost worker"; worker thing2 ) &
wait
echo "$(date +%H:%M) boost workers finished"
while pgrep -f "bash pieces/_logs/ice-song3-chain\.sh" > /dev/null; do sleep 30; done
L=pieces/_logs/ice-song3-chain.log
if grep -q "ice song3 done" $L; then echo "$(date +%H:%M) song3 stitched by the farm"; exit 0; fi
grep -qE "chunk [0-9]+ missing" $L || { echo "$(date +%H:%M) song3 farm ended otherwise: see $L"; exit 1; }
echo "$(date +%H:%M) farm ended before the boost's last chunk: re-run to stitch" | tee -a $L
TAG=song3 bash $F >> $L 2>&1
tail -1 $L
