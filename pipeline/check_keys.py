"""Still check, between the keyframes and the expensive video step: find stills worth re-rolling, and mark them so the
next keyframes / amber_keys pass draws them again with a new seed (OUT/reroll.json: shot -> attempts).

Flags (Slippery Road v1 and Chronophobia v3 review, 2026-10-24):
  blank      near-white or near-flat still (MOSS sometimes returns one; it shows as a white flare in the video)
  stranger   a performer shot (scene / sing) whose lit, frontal face has dark hair around it. SFace face recognition
             can't separate her from a dark-haired stranger (it ignores hair: 0.40 for a stranger against 0.14-0.17 for
             her own shots), but her long blond hair is the giveaway: lightness around the face 18-53 on her shots
             (night scenes included), 1-3 on the stranger
  face       a "part" shot (a hand, her back, a reflection) that shows a large full face
  repeat     the third and later of near-identical consecutive stills (reported; the merge step handles scenes)

    python3 pipeline/check_keys.py OUT --photo PHOTO [--reroll]     (--reroll: delete flagged stills, bump reroll.json)
Exit code 0; prints one line per flag and a summary. Face detection: YuNet (see eval_identity.py).
"""
from __future__ import annotations

import argparse, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--photo", help="unused for now (kept for a future identity model)")
ap.add_argument("--keys", default="keys")
ap.add_argument("--dark-hair", type=float, default=12, help="hair lightness (L*, 0-100) below this around a lit face = "
                "not her (her photo: 51)")
ap.add_argument("--max-attempts", type=int, default=2)
ap.add_argument("--reroll", action="store_true")
a = ap.parse_args()

shots = json.load(open(os.path.join(a.out, "shots.json")))["shots"]
rr_path = os.path.join(a.out, "reroll.json")
rr = json.load(open(rr_path)) if os.path.exists(rr_path) else {}
M = os.path.expanduser("~/models/face")
det = cv2.FaceDetectorYN.create(f"{M}/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.85, 0.3, 5000)


def face(img):
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, f = det.detect(img)
    if f is None or len(f) == 0:
        return None
    f = max(f, key=lambda r: r[2] * r[3])
    eyes = abs(f[4] - f[6])                                    # eye distance vs face width: frontal-ish if large
    return f, (f[2] * f[3]) / (w * h), eyes / max(f[2], 1)


def lightness(img, f):
    """L* (0-100) of the hair around a face (the lightest of a band above it and strips at its sides) and of the face."""
    H, W = img.shape[:2]
    x, y, fw, fh = (int(v) for v in f[:4])
    x, y = max(0, x), max(0, y)                               # a face at the frame edge: keep the regions inside it
    regs = [img[max(0, y - fh // 3):y, x:min(W, x + fw)], img[y:min(H, y + fh), max(0, x - fw // 3):x],
            img[y:min(H, y + fh), min(W, x + fw):min(W, x + fw + fw // 3)]]
    regs = [r for r in regs if r.size] or [img[y:min(H, y + fh), x:min(W, x + fw)]]
    L = lambda q: float(np.median(cv2.cvtColor(q.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)[:, 0, 0])) * 100 / 255
    # the lightest region: against a dark wall the band above her head is background, but her hair is lit somewhere
    return max(L(r.reshape(-1, 3)) for r in regs), L(img[y:min(H, y + fh), x:min(W, x + fw)].reshape(-1, 3))

flags, prev, run = [], None, 0
for s in shots:
    p = os.path.join(a.out, a.keys, f"shot_{s['i']:03d}.png")
    if not os.path.exists(p):
        continue
    img = cv2.imread(p)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    why = None
    if g.mean() > 215 or g.std() < 12:
        why = f"blank (mean {g.mean():.0f}, contrast {g.std():.0f})"
    elif s.get("cast") in ("scene", "sing", "part"):
        fr = face(img)
        if fr is not None:
            f, area, frontal = fr
            if s["cast"] == "part" and area > 0.04 and frontal > 0.3:
                why = f"face in a part shot (face {area:.0%} of frame)"
            elif s["cast"] in ("scene", "sing") and area > 0.01 and frontal > 0.35:
                hair_l, face_l = lightness(img, f)
                if hair_l < a.dark_hair and face_l > 25:
                    why = f"stranger (dark hair around a lit face: hair L {hair_l:.0f}, face L {face_l:.0f})"
    small = cv2.resize(g, (64, 36)).astype(np.float32)
    if prev is not None and np.corrcoef(small.ravel(), prev.ravel())[0, 1] > 0.97:
        run += 1
        if run >= 2 and why is None:
            print(f"shot {s['i']:3d} repeat (near-identical to the {run} before)")
    else:
        run = 0
    prev = small
    if why:
        n = rr.get(str(s["i"]), 0)
        flags.append((s["i"], why, n))
        print(f"shot {s['i']:3d} {s.get('cast', '-'):5s} {why}" + ("" if n < a.max_attempts else "  (kept: out of attempts)"))

todo = [i for i, _, n in flags if n < a.max_attempts]
if a.reroll and todo:
    for i in todo:
        rr[str(i)] = rr.get(str(i), 0) + 1
        os.remove(os.path.join(a.out, a.keys, f"shot_{i:03d}.png"))
    json.dump(rr, open(rr_path, "w"), indent=1)
print(f"{len(flags)} flagged, {len(todo)} to re-roll" + (" (deleted; reroll.json updated)" if a.reroll and todo else ""))
