"""Face-pass benchmark (D45), step 1, on a Spark (morph-spark): draw each case (cases.json) with MOSS, seed 1, a stranger
with the performer's description in the case's treatment and position. OUT/<id>.png.
    python3 lean/facebench/draw.py OUT"""
import json, os, sys

sys.path.insert(0, "pipeline")
from keyframe_model import load

C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json"))); out = sys.argv[1]
os.makedirs(out, exist_ok=True)
r = load("moss", offload=False)
for c in C["cases"]:
    if not os.path.exists(f"{out}/{c['id']}.png"):
        st = C["styles"][c["style"]]
        r.render(None, [(f"{st[0].upper()}{st[1:]}: {c['scene'].format(woman=C['woman'])}, {c['expression']}", 1.0)],
                 1.0, seed=1, size=(1024, 576))[0].save(f"{out}/{c['id']}.png")
        print(f"{c['id']}: drawn", flush=True)
