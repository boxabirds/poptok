"""Self-check: does the singer's mouth follow the vocal? For each frame, the face detector (YuNet) finds the mouth corners;
mouth openness = how dark the region between and below them is. That series is correlated with the vocal's loudness
at lags of -320..+320 ms. Singers open the mouth a little before the sound, so the peak should sit at -80..-200 ms,
well above the shuffled baseline. POC 19: InfiniteTalk scored r = 0.31 at -120..-160 ms (baseline 0.08).

    python3 pipeline/eval_lipsync.py VIDEO.mp4 VOCALS.wav [--window S]
--window: the lags searched, +-S seconds (default 0.32). D38 test 2's peaks all sat at the +320 ms edge; at +-1 s
they were at +360..+400 ms in every clip, so a clip-wide offset needs the wider window to be seen.
"""
from __future__ import annotations

import argparse, os, subprocess, wave

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("vocals"); ap.add_argument("--window", type=float, default=0.32)
A = ap.parse_args(); video, vocals = A.video, A.vocals
M = os.path.expanduser("~/models/face")
det = cv2.FaceDetectorYN.create(f"{M}/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.7, 0.3, 5000)
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
open_, found = [], 0
while True:
    ok, img = cap.read()
    if not ok:
        break
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        open_.append(np.nan)
        continue
    f = max(faces, key=lambda r: r[2] * r[3])
    found += 1
    (rx, ry), (lx, ly) = f[10:12], f[12:14]                     # mouth corners
    mw = max(4, int(abs(lx - rx)))
    cx, cy = int((rx + lx) / 2), int((ry + ly) / 2)
    box = cv2.cvtColor(img[max(0, cy - mw // 3):cy + mw // 2, max(0, cx - mw // 2):cx + mw // 2], cv2.COLOR_BGR2GRAY)
    open_.append(float((box < np.percentile(box, 30) * 0.8 + 10).mean()) if box.size else np.nan)
o = np.array(open_)
o = np.where(np.isnan(o), np.nanmean(o), o)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vocals, "-ac", "1", "-ar", "16000", "/tmp/_lipsync_v.wav"])
wv = wave.open("/tmp/_lipsync_v.wav")
a = np.frombuffer(wv.readframes(wv.getnframes()), np.int16).astype(np.float32)
hop = int(16000 / fps)
n = min(len(o), len(a) // hop)
rms = np.array([np.sqrt((a[i * hop:(i + 1) * hop] ** 2).mean()) for i in range(n)])
o = o[:n]
z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
best = (0, -1.0)
for lag in range(-int(A.window * fps), int(A.window * fps) + 1):
    c = np.corrcoef(z(o[lag:]), z(rms[:n - lag]))[0, 1] if lag >= 0 else np.corrcoef(z(o[:lag]), z(rms[-lag:]))[0, 1]
    if c > best[1]:
        best = (lag, c)
rng = np.random.default_rng(0)
base = np.mean([abs(np.corrcoef(o, np.roll(rms, rng.integers(25, n - 25)))[0, 1]) for _ in range(50)])
print(f"{os.path.basename(video)}: face in {found}/{len(open_)} frames; best r = {best[1]:.2f} at {best[0] * 1000 / fps:+.0f} ms "
      f"(negative = mouth leads the sound); shuffled baseline {base:.2f}")
