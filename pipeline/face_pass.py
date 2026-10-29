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
    fix("still.png", "photo.jpg", angles="pieces/artists/r.amber-angles")   # with her_angles.py's set: for a turned
                                                            # head, her reference at the nearest turn replaces the photo

    python3 pipeline/face_pass.py PHOTO IN.png... --out DIR [--tries 3] [--angles DIR] [--min-sim 0.5 [--min-face 80]]
(writes DIR/<name>, prints the scores; with --min-sim, exits 2 if any output scores below it, so a stand-in face
never goes on to the next step: Shot B's first frame scored 0.42 and the user saw "someone, but it's not Amber").
--min-face: a face narrower than this (pixels) is too small to judge (SFace on shot A's 38 px face): reported, not gated)
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
HEAD_PROMPT = ("Make the woman's head in picture 1 the head of the woman in picture 2: her face shape, jaw line, chin, "
               "cheekbones, forehead and hairline as well as her eyes, eyebrows, nose and mouth. Keep picture 1's head "
               "angle, expression, gaze, framing, hair colour and length, skin light and shadow, and any paint or marks "
               "on her face exactly. Nothing else changes. No text.")
WHOLE_PROMPT = ("Make the woman in picture 1 the woman in picture 2: her face, its shape and her features. Keep everything "
                "else in picture 1 exactly: her pose, head angle, gaze, her hair (its colour, length, texture and how "
                "it falls), her clothes, the background, the framing, the light. Do not use picture 2's hair, clothes "
                "or background. No text.")
CROP = 2.6          # crop side, in face widths: the face plus enough around it for the light and the head's angle
OVAL = (0.62, 0.78)  # pasted oval's half-axes, in face widths (cheeks to cheeks, hairline to chin)
HEAD = (0.85, 1.12, -0.08)  # region="head": half-axes and centre shift (up) in face widths: jaw, face outline and
# hairline too. With only the inner face replaced, the user saw "her sisters" (Chronophobia, 2026-10-26): her features
# on the image model's head shape


def face(img):      # largest face as (x, y, w, h) in pixels, or None
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    return None if faces is None or len(faces) == 0 else max(faces, key=lambda r: r[2] * r[3])


def turn(f):        # head turn from YuNet's landmarks: the nose's offset from between the eyes, in eye distances
    eye_x, eye_d = (f[4] + f[6]) / 2, max(abs(f[6] - f[4]), 1.0)
    return float(np.clip((f[8] - eye_x) / eye_d, -2, 2))


def angle_ref(angles, f):   # her_angles.py's reference nearest this face's turn, if it is a turned one (else None:
    if not angles or not os.path.exists(f"{angles}/angles.json"):          # the photo itself is frontal)
        return None
    meta = json.load(open(f"{angles}/angles.json"))
    name = min(meta, key=lambda n: abs(meta[n]["turn"] - turn(f)))
    return None if abs(meta[name]["turn"]) < 0.25 else f"{angles}/{name}.png"


def similarity(img, ref):
    f = face(img)
    return None if f is None else float(rec.match(ref, rec.feature(rec.alignCrop(img, f)), cv2.FaceRecognizerSF_FR_COSINE))


def load(edit=None):
    if edit is None:
        from qwen_edit import load as load_edit
        edit = load_edit()
    refs = {}

    def fix(still, photo, seed=0, tries=3, angles=None, region="head", style=None, expression=None, hair=None):
        # region head: user-approved 2026-10-26. style: picture 1's medium (e.g. charcoal), so her face is redrawn in it
        # instead of pasted in as a photograph. expression: picture 1's, named (D43: "keep the expression" alone gave
        # her the photo's calm look over a drawn forlorn one). D45, being tested: region "whole" edits the whole still
        # and pastes nothing (the head oval stamped photo-like hair into drawn hair); region "nohair" is the head oval
        # minus hair (a full-frame mask, True = hair), so every hair pixel stays the still's.
        # Every try is kept in fix.tries ({image, sim, ok}) for a person to choose from (D44).
        prompt = {"head": HEAD_PROMPT, "nohair": HEAD_PROMPT, "whole": WHOLE_PROMPT}.get(region, PROMPT) + \
            (f" Her face stays drawn the way picture 1 is drawn ({style}), not photographed." if style else "") + \
            (f" Her expression stays exactly picture 1's, not picture 2's: {expression}." if expression else "")
        src = np.array(Image.open(still).convert("RGB") if isinstance(still, str) else still.convert("RGB"))
        bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        if photo not in refs:
            p = cv2.imread(photo)
            refs[photo] = rec.feature(rec.alignCrop(p, face(p)))
        f = face(bgr)
        fix.tries = []
        if f is None:
            return Image.fromarray(src), None
        H, W = src.shape[:2]
        cx, cy, fw = f[0] + f[2] / 2, f[1] + f[3] / 2, max(f[2], f[3])
        if region == "whole":
            ref2 = angle_ref(angles, f) or photo
            sc = 1024 / max(W, H); size = (int(W * sc) // 16 * 16, int(H * sc) // 16 * 16)
            best = (None, -2.0)
            for t in range(tries):
                o = np.array(edit([Image.fromarray(src), ref2], prompt, seed=seed + 1000 * t, size=size)
                             .convert("RGB").resize((W, H), Image.LANCZOS))
                g = face(cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
                ok = g is not None and np.hypot(g[0] + g[2] / 2 - cx, g[1] + g[3] / 2 - cy) <= 0.15 * fw \
                    and 0.8 < max(g[2], g[3]) / fw < 1.25      # the framing kept: she is where she was, the same size
                sim = similarity(cv2.cvtColor(o, cv2.COLOR_RGB2BGR), refs[photo]) if ok else None
                fix.tries.append({"image": Image.fromarray(o), "sim": sim, "ok": ok})
                if ok and sim is not None and sim > best[1]:
                    best = (o, sim)
            if best[0] is None:
                return Image.fromarray(src), similarity(bgr, refs[photo])
            return Image.fromarray(best[0]), best[1]
        side = int(min(CROP * fw, W, H))
        x0, y0 = int(np.clip(cx - side / 2, 0, W - side)), int(np.clip(cy - side / 2, 0, H - side))
        crop = Image.fromarray(src[y0:y0 + side, x0:x0 + side])
        mask = np.zeros((H, W), np.float32)            # the oval, feathered by ~a tenth of the face width
        ax, ay, up = HEAD if region in ("head", "nohair") else (*OVAL, 0.0)
        cv2.ellipse(mask, (int(cx), int(cy + up * fw)), (int(ax * fw), int(ay * fw)), 0, 0, 360, 1.0, -1)
        k = int(fw * (0.4 if region in ("head", "nohair") else 0.2)) | 1     # head: a softer edge (a hard one drew a line at the neck)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        if region == "nohair" and hair is not None:      # the oval's hair stays the still's: cut out after feathering,
            hm = np.array(Image.open(hair).convert("L")) > 127 if isinstance(hair, str) else np.asarray(hair, bool)
            inner = np.zeros((H, W), np.uint8)            # the inner face is never hair (SAM's "hair" often is the head)
            cv2.ellipse(inner, (int(cx), int(cy)), (int(OVAL[0] * fw), int(OVAL[1] * fw)), 0, 0, 360, 1, -1)
            hm = hm & (inner == 0)
            kh = int(fw * 0.04) | 1                       # with a narrow soft edge (the oval's would leak into the hair)
            mask *= 1 - cv2.GaussianBlur(hm.astype(np.float32), (kh, kh), 0)
        mask = mask[..., None]
        best = (None, -2.0)
        # her at this head angle replaces the photo as picture 2 (a frontal photo turns profiles into strangers); never
        # a third picture: with three, Qwen moved and rescaled the head in the crop and the oval pasted a mismatched face
        ref2 = angle_ref(angles, f) or photo
        fc = (cx - x0, cy - y0)           # the face in crop coordinates, to check the edit kept it in place
        for t in range(tries):
            out = edit([crop, ref2], prompt, seed=seed + 1000 * t, size=(1024, 1024))
            o = np.array(out.convert("RGB").resize((side, side), Image.LANCZOS))
            g = face(cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
            if g is None or np.hypot(g[0] + g[2] / 2 - fc[0], g[1] + g[3] / 2 - fc[1]) > 0.15 * fw \
                    or not 0.8 < max(g[2], g[3]) / fw < 1.25:
                fix.tries.append({"image": None, "sim": None, "ok": False})
                continue                  # the face moved or changed size: pasting it would leave a seam
            big = src.astype(np.float32).copy()
            big[y0:y0 + side, x0:x0 + side] = o
            comp = (src * (1 - mask) + big * mask).astype(np.uint8)
            s = similarity(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR), refs[photo])
            s = -1.0 if s is None else s
            fix.tries.append({"image": Image.fromarray(comp), "sim": s, "ok": True})
            if s > best[1]:
                best = (comp, s)
        if best[0] is None:               # every try moved the face: keep the still as it was
            return Image.fromarray(src), similarity(bgr, refs[photo])
        return Image.fromarray(best[0]), best[1]

    return fix


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--angles", help="her_angles.py output dir")
    ap.add_argument("--region", choices=["face", "head"], default="head")
    ap.add_argument("--min-sim", type=float, help="gate: exit 2 if any output's similarity is below this")
    ap.add_argument("--min-face", type=int, default=0, help="gate only faces at least this wide (pixels)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fix = load()
    low = []
    for p in a.inputs:
        before = similarity(cv2.imread(p), rec.feature(rec.alignCrop(cv2.imread(a.photo), face(cv2.imread(a.photo)))))
        im, s = fix(p, a.photo, tries=a.tries, angles=a.angles, region=a.region)
        im.save(os.path.join(a.out, os.path.basename(p)))
        print(f"{os.path.basename(p)}: " + ("no face, unchanged" if s is None else
              f"similarity {before:.2f} -> {s:.2f}"), flush=True)
        f = face(cv2.imread(p))
        if f is not None and max(f[2], f[3]) < a.min_face:
            print(f"  {os.path.basename(p)}: face {int(max(f[2], f[3]))} px, too small to judge (not gated)", flush=True)
        elif a.min_sim is not None and (s is None or s < a.min_sim):
            low.append(os.path.basename(p))
    if low:
        print(f"below --min-sim {a.min_sim}: {' '.join(low)}"); sys.exit(2)
