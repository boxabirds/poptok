"""Join a long render made in overlapping chunks (ice_world.py --range) into one frame sequence (2026-10-27, Ice Walk whole
song). Every chunk builds the same world, so the chunks agree everywhere but in the snow, which each simulates for itself;
over each overlap the frames crossfade from the earlier chunk to the later one, so the flakes dissolve rather than jump.

    python3 pipeline/world/stitch_chunks.py OUT_DIR N CHUNK OVERLAP [--start S] CHUNK_DIR_0 CHUNK_DIR_1 ...
--start S: the chunks hold song frames S..S+N-1 (files named by song frame); OUT_DIR still gets 0001.png .. N.png.
Chunk k covers frames k*CHUNK+1 .. min(N, (k+1)*CHUNK+OVERLAP); CHUNK_DIR_k/NNNN.png. OUT_DIR gets 0001.png .. N.png
(hard links where a frame comes from one chunk, blends over the overlaps).
"""
from __future__ import annotations

import os, subprocess, sys

out, N, CH, OV = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dirs = sys.argv[5:]
S0 = 1
if dirs[0] == "--start":
    S0, dirs = int(dirs[1]), dirs[2:]
os.makedirs(out, exist_ok=True)
for f in range(1, N + 1):
    k = min((f - 1) // CH, len(dirs) - 1)
    dst = os.path.join(out, f"{f:04d}.png")
    if os.path.exists(dst):
        os.remove(dst)
    cur = os.path.join(dirs[k], f"{f + S0 - 1:04d}.png")
    into = f - (k * CH + 1)                      # frames since this chunk took over
    if k > 0 and into < OV:
        w = (into + 0.5) / OV
        prev = os.path.join(dirs[k - 1], f"{f + S0 - 1:04d}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", prev, "-i", cur, "-filter_complex",
                        f"[0][1]blend=all_expr='A*{1 - w:.4f}+B*{w:.4f}'", dst], check=True)
    else:
        os.link(cur, dst)
print(out, N, "frames from", len(dirs), "chunks")
