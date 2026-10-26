"""Is this her? AdaFace IR101 (WebFace12M) face embeddings against her real photos (2026-10-26). SFace
(eval_identity.py) disagreed with the user's own verdicts on the universe stills: it rated Chronophobia's openings
"same person" (0.49-0.57) where the user saw "her sisters", and Noble Hunt's "different" (0.26-0.38) where the user said
yes. This is the stronger recogniser, calibrated on those verdicts (identity_judge.py --calibrate).

    python3 pipeline/identity_judge.py IMAGE... [--refs DIR_OR_IMAGES...]       # one score per image
    python3 pipeline/identity_judge.py --calibrate LABELS.json                  # [{"image", "her": true|false}]
Faces: YuNet detection and the standard 112x112 alignment (eval_identity.py); refs default to the photos in
pieces/artists/ (r.amber*.jp*g). Score: the best cosine against any ref. Weights: ~/models/face/adaface-ir101
(huggingface.co/minchul/cvlface_adaface_ir101_webface12m; code MIT, weights follow WebFace12M's research licence, so
this is an evaluation tool, never part of what makes the video). CPU is fine.
"""
from __future__ import annotations

import argparse, glob, json, os, sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_identity import M, rec
from face_pass import face

ADA = os.environ.get("ADAFACE") or os.path.join(M, "adaface-ir101")
_model = None


def model():
    global _model
    if _model is None:
        import types
        from transformers import AutoModel
        if "fvcore" not in sys.modules:     # the model file imports fvcore only to count FLOPs; not needed to run it
            fv = types.ModuleType("fvcore"); fv.nn = types.ModuleType("fvcore.nn"); fv.nn.flop_count = None
            sys.modules["fvcore"], sys.modules["fvcore.nn"] = fv, fv.nn
        cwd = os.getcwd(); os.chdir(ADA); sys.path.insert(0, ADA)
        try:
            _model = AutoModel.from_pretrained(ADA, trust_remote_code=True).eval()
        finally:
            os.chdir(cwd); sys.path.pop(0)
    return _model


def embed(img):                 # BGR image -> unit 512-vector of its largest face, or None
    f = face(img)
    if f is None:
        return None
    crop = cv2.cvtColor(rec.alignCrop(img, f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    x = torch.from_numpy((crop - 0.5) / 0.5).permute(2, 0, 1)[None]
    with torch.no_grad():
        e = model()(x)
    e = e[0] if isinstance(e, (tuple, list)) else e
    return torch.nn.functional.normalize(e.reshape(1, -1), dim=1)[0].numpy()


def refs(paths=None):
    paths = paths or sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "pieces", "artists", "r.amber*.jp*g")))
    out = []
    for p in paths:
        for q in (sorted(glob.glob(f"{p}/*.png") + glob.glob(f"{p}/*.jp*g")) if os.path.isdir(p) else [p]):
            e = embed(cv2.imread(q))
            if e is not None:
                out.append(e)
    return out


def score(img, R):              # best cosine against any ref; None if no face
    e = embed(img if not isinstance(img, str) else cv2.imread(img))
    return None if e is None else float(max(e @ r for r in R))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--refs", nargs="*")
    ap.add_argument("--calibrate")
    a = ap.parse_args()
    R = refs(a.refs)
    if a.calibrate:
        rows = json.load(open(a.calibrate))
        for r in rows:
            r["score"] = score(r["image"], R)
            print(f"{'her ' if r['her'] else 'NOT '} {r['score'] if r['score'] is None else round(r['score'], 3)!s:>6}  {r['image']}")
        yes = [r["score"] for r in rows if r["her"] and r["score"] is not None]
        no = [r["score"] for r in rows if not r["her"] and r["score"] is not None]
        if yes and no:
            pairs = sum((y > n) + 0.5 * (y == n) for y in yes for n in no) / (len(yes) * len(no))
            print(f"her: {len(yes)}, min {min(yes):.3f}, mean {np.mean(yes):.3f} | not her: {len(no)}, max {max(no):.3f}, "
                  f"mean {np.mean(no):.3f} | ranks her above not-her {pairs:.0%} of the time")
    for p in a.images:
        s = score(p, R)
        print(f"{p}: {'no face' if s is None else f'{s:.3f}'}")
