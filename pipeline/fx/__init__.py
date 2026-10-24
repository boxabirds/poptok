"""Pluggable per-piece effects for the finish step (finish.py --fx CONFIG.json).

A piece lists the effects it wants in a JSON config (e.g. pieces/NAME/inputs/fx.json, set as FX= in piece.env):
    [{"effect": "bursts", "events": {...}, ...params}]
Each entry names a module in this package (pipeline/fx/<effect>.py) exposing make(cfg, ctx) -> effect, where effect is
a callable (img, base, t) -> img applied to every output frame after grading (img/base: float32 HxW or HxWx3, 0-255;
t: seconds). ctx gives W, H, fps, xs, ys (pixel grids), analysis (the piece's analysis dir), seed.
Event selection shared by effects is in events.py. Effects are deliberately bespoke: add a module here when a piece
needs one, and reuse it from other pieces' configs when it fits.
"""
from __future__ import annotations

import importlib, json, subprocess

import numpy as np


def load(config_path: str, ctx: dict) -> list:
    effects = []
    for cfg in json.load(open(config_path)):
        mod = importlib.import_module(f"fx.{cfg['effect']}")
        effects.append(mod.make(cfg, ctx))
    return effects


def apply_to_video(src: str, dst: str, config_path: str, analysis: str, ranges: list, seed: int = 7) -> None:
    """Apply the effects to the frames of SRC inside RANGES [(start, end) s] and write DST (audio copied). For footage
    cut in after the finish step (the lip-sync inserts), which would otherwise lose the effects that fall inside it."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height,r_frame_rate",
                            "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip().split(",")
    W, H, fps = int(probe[0]), int(probe[1]), eval(probe[2])
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    effects = load(config_path, {"W": W, "H": H, "fps": fps, "xs": xs, "ys": ys, "seed": seed, "analysis": analysis})
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", probe[2],
                            "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-c:a", "copy", dst], stdin=subprocess.PIPE)
    f, size = 0, W * H * 3
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        t = f / fps
        if any(a <= t < b for a, b in ranges):
            img = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32)
            base = img.copy()
            for effect in effects:
                img = effect(img, base, t)
            buf = np.clip(img, 0, 255).astype(np.uint8).tobytes()
        enc.stdin.write(buf)
        f += 1
    enc.stdin.close(); enc.wait(); dec.wait()
