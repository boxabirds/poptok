"""Self-check: is the performer in these images the same person as in her photo? Face detection (YuNet) and face
recognition (SFace), both from OpenCV's model zoo (MIT / Apache-2.0), on the CPU.

    python3 pipeline/eval_identity.py PHOTO IMAGE_OR_DIR_OR_VIDEO... [--every N] [--json OUT.json]
For each image (or every Nth frame of a video): the largest face's cosine similarity to the photo's face. SFace's own
threshold for "same person" is 0.363; real photos of one person usually score 0.5-0.8. No face found: "-".
Models in ~/models/face, or /models/face in the containers, or $FACE_MODELS (face_detection_yunet_2023mar.onnx, face_recognition_sface_2021dec.onnx).
"""
from __future__ import annotations

import argparse, glob, json, os, sys

import cv2
import numpy as np

SAME = 0.363
M = os.environ.get("FACE_MODELS") or next((d for d in (os.path.expanduser("~/models/face"), "/models/face")
                                           if os.path.isdir(d)), "/models/face")   # /models in the containers
det = cv2.FaceDetectorYN.create(f"{M}/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.8, 0.3, 5000)
rec = cv2.FaceRecognizerSF.create(f"{M}/face_recognition_sface_2021dec.onnx", "")


def feature(img):
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None, 0
    f = max(faces, key=lambda r: r[2] * r[3])
    return rec.feature(rec.alignCrop(img, f)), len(faces)


def frames(path, every):
    if os.path.isdir(path):
        for p in sorted(glob.glob(f"{path}/*.png") + glob.glob(f"{path}/*.jpg")):
            yield os.path.basename(p), cv2.imread(p)
    elif path.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
        cap, i = cv2.VideoCapture(path), 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield f"{os.path.basename(path)}#{i}", img
            i += 1
    else:
        yield os.path.basename(path), cv2.imread(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--every", type=int, default=12, help="video: score every Nth frame")
    ap.add_argument("--json")
    a = ap.parse_args()
    ref, _ = feature(cv2.imread(a.photo))
    if ref is None:
        sys.exit("no face in the photo")
    rows = []
    for inp in a.inputs:
        for name, img in frames(inp, a.every):
            f, n = feature(img)
            score = None if f is None else float(rec.match(ref, f, cv2.FaceRecognizerSF_FR_COSINE))
            rows.append({"image": name, "faces": n, "similarity": score})
            print(f"{name:40s} faces {n}  " + ("-" if score is None else f"{score:.2f}{'' if score >= SAME else '  DIFFERENT?'}"))
    s = [r["similarity"] for r in rows if r["similarity"] is not None]
    if s:
        print(f"\n{len(s)}/{len(rows)} with a face; similarity mean {np.mean(s):.2f}, min {min(s):.2f}; "
              f"below {SAME} (different person?): {sum(x < SAME for x in s)}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
