"""Face-pass benchmark (D45), step 3, on a Spark: each case through each face-pass region (head = the current one,
whole, nohair), with the case's expression named and the treatment as the style; every try saved,
OUT/<id>/<region>-<try>.png, scores in OUT/<id>/tries.json.
    python3 lean/facebench/run.py OUT PHOTO [ID ...]"""
import json, os, sys

sys.path.insert(0, "pipeline")
import face_pass

C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json")))
out, photo, only = sys.argv[1], sys.argv[2], set(sys.argv[3:])
fix = face_pass.load()
for c in C["cases"]:
    if only and c["id"] not in only:
        continue
    d = f"{out}/{c['id']}"; os.makedirs(d, exist_ok=True)
    res = json.load(open(f"{d}/tries.json")) if os.path.exists(f"{d}/tries.json") else {}
    for region in ("head", "whole", "nohair"):
        if region in res:
            continue
        fix(f"{out}/{c['id']}.png", photo, seed=1, tries=3, angles=os.path.splitext(photo)[0] + "-angles",
            region=region, style=C["styles"][c["style"]], expression=c["expression"], hair=f"{out}/{c['id']}-hair.png")
        res[region] = []
        for t, x in enumerate(fix.tries):
            if x["image"] is not None:
                x["image"].save(f"{d}/{region}-{t + 1}.png")
            res[region].append({"try": t + 1, "sim": x["sim"], "ok": x["ok"]})
        json.dump(res, open(f"{d}/tries.json", "w"), indent=1)
        print(f"{c['id']} {region}: " + ", ".join(f"{x['sim']:.2f}" if x["sim"] is not None else "-" for x in res[region]),
              flush=True)
