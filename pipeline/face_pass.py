"""Make the performer's face hers in a finished still, changing nothing else (2026-10-26). Qwen-Image-Edit given the
whole still and her photo also copies the photo's clothes (her black dress replaced Noble Hunt's uniform in half the
universe stills, whatever the prompt said) and drifts her face towards someone else. So the edit is confined:

  1. find the largest face (YuNet, eval_identity.py) and crop a square around it, wide enough for context and light;
  2. Qwen-Image-Edit that crop with her photo as picture 2, at 1024x1024;
  3. paste back only a feathered oval over the face, so clothes, hair, hands and the world are untouched pixels;
  4. score each try against her photo (SFace cosine) and keep the best of N seeds.

Any step that shows her (universe stills, the performer keyframes) can call it.

    from face_pass import load
    fix = load()                                            # loads Qwen-Image-Edit (a Spark: ~60 GB, no offload)
    image, score = fix("still.png", "photo.jpg", seed=1)    # PIL image; score None if no face was found (unchanged)
    fix("still.png", "photo.jpg", angles="pieces/artists/r.amber-angles")   # with her_angles.py's set: the reference
                                                            # nearest the still's head turn goes in as picture 3

    python3 pipeline/face_pass.py PHOTO IN.png... --out DIR [--tries 3] [--angles DIR]   (writes DIR/<name>, prints the scores)
"""
from __future__ import annotations

import argparse, json, os, sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_identity import det, rec

PROMPT = ("Make the woman's face in picture 1 the face of the woman in picture 2: her eyes, eyebrows, nose, mouth, jaw "
          "and face shape. Keep picture 1's head angle, expression, gaze, framing, skin light and shadow, and any paint "
          "or marks on her face exactly. Nothing else changes. No text.")
ANGLED = ("Make the woman's face in picture 1 the face of the woman in pictures 2 and 3 (the same person; picture 3 shows "
          "her head at the same angle as in picture 1): her eyes, eyebrows, nose, mouth, jaw and face shape. Keep "
          "picture 1's head angle, expression, gaze, framing, skin light and shadow, and any paint or marks on her face "
          "exactly. Nothing else changes. No text.")
CROP = 2.6          # crop side, in face widths: the face plus enough around it for the light and the head's angle
OVAL = (0.62, 0.78)  # pasted oval's half-axes, in face widths (cheeks to cheeks, hairline to chin)


def face(img):      # largest face as (x, y, w, h) in pixels, or None
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    return None if faces is None or len(faces) == 0 else max(faces, key=lambda r: r[2] * r[3])


def turn(f):        # head turn from YuNet's landmarks: the nose's offset from between the eyes, in eye distances
    eye_x, eye_d = (f[4] + f[6]) / 2, max(abs(f[6] - f[4]), 1.0)
    return float(np.clip((f[8] - eye_x) / eye_d, -2, 2))


def angle_ref(angles, f):   # her_angles.py's reference nearest this face's turn (None: no angle set)
    if not angles or not os.path.exists(f"{angles}/angles.json"):
        return None
    meta = json.load(open(f"{angles}/angles.json"))
    name = min(meta, key=lambda n: abs(meta[n]["turn"] - turn(f)))
    return f"{angles}/{name}.png"


def similarity(img, ref):
    f = face(img)
    return None if f is None else float(rec.match(ref, rec.feature(rec.alignCrop(img, f)), cv2.FaceRecognizerSF_FR_COSINE))


def load(edit=None):
    if edit is None:
        from qwen_edit import load as load_edit
        edit = load_edit()
    refs = {}

    def fix(still, photo, seed=0, tries=3, angles=None):
        src = np.array(Image.open(still).convert("RGB") if isinstance(still, str) else still.convert("RGB"))
        bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        if photo not in refs:
            p = cv2.imread(photo)
            refs[photo] = rec.feature(rec.alignCrop(p, face(p)))
        f = face(bgr)
        if f is None:
            return Image.fromarray(src), None
        H, W = src.shape[:2]
        cx, cy, fw = f[0] + f[2] / 2, f[1] + f[3] / 2, max(f[2], f[3])
        side = int(min(CROP * fw, W, H))
        x0, y0 = int(np.clip(cx - side / 2, 0, W - side)), int(np.clip(cy - side / 2, 0, H - side))
        crop = Image.fromarray(src[y0:y0 + side, x0:x0 + side])
        mask = np.zeros((H, W), np.float32)            # the oval, feathered by ~a tenth of the face width
        cv2.ellipse(mask, (int(cx), int(cy)), (int(OVAL[0] * fw), int(OVAL[1] * fw)), 0, 0, 360, 1.0, -1)
        k = int(fw * 0.2) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
        best = (None, -2.0)
        aref = angle_ref(angles, f)       # her at this head angle: a frontal photo alone turns profiles into strangers
        for t in range(tries):
            out = edit([crop, photo, aref] if aref else [crop, photo], ANGLED if aref else PROMPT,
                       seed=seed + 1000 * t, size=(1024, 1024))
            big = src.astype(np.float32).copy()
            big[y0:y0 + side, x0:x0 + side] = np.array(out.convert("RGB").resize((side, side), Image.LANCZOS))
            comp = (src * (1 - mask) + big * mask).astype(np.uint8)
            s = similarity(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR), refs[photo])
            s = -1.0 if s is None else s
            if s > best[1]:
                best = (comp, s)
        return Image.fromarray(best[0]), best[1]

    return fix


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--angles", help="her_angles.py output dir")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fix = load()
    for p in a.inputs:
        before = similarity(cv2.imread(p), rec.feature(rec.alignCrop(cv2.imread(a.photo), face(cv2.imread(a.photo)))))
        im, s = fix(p, a.photo, tries=a.tries, angles=a.angles)
        im.save(os.path.join(a.out, os.path.basename(p)))
        print(f"{os.path.basename(p)}: " + ("no face, unchanged" if s is None else
              f"similarity {before:.2f} -> {s:.2f}"), flush=True)
