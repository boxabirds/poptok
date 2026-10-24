"""Bursts: on each event (e.g. the drum triplets after "the noble hunt"), a warm bloom of light from the frame's
brightest point, a small global lift, a spray of sparks under gravity, and a short knock of the viewpoint (a damped
shake, as if the viewer were jolted by the blast). Made for noble-hunt (2026-10-24).

Config: {"effect": "bursts", "events": {events.select spec},
         "amp": 1.0,                    light strength
         "color": [255, 222, 175],      light/spark colour (RGB)
         "sparks": 46,                  sparks per burst
         "shake_px": 14,                peak shake in pixels at 1280 px wide (scaled with width); 0 = none
         "shake_hz": 16, "shake_decay": 0.12}
"""
from __future__ import annotations

import cv2
import numpy as np

from fx.events import select


def make(cfg: dict, ctx: dict):
    W, H, fps, xs, ys = ctx["W"], ctx["H"], ctx["fps"], ctx["xs"], ctx["ys"]
    ev = select(cfg["events"], ctx["analysis"])
    amp, col = float(cfg.get("amp", 1.0)), np.array(cfg.get("color", [255, 222, 175]), np.float32)
    n_sp = int(cfg.get("sparks", 46))
    sh_px, sh_hz, sh_dec = float(cfg.get("shake_px", 14)) * W / 1280, float(cfg.get("shake_hz", 16)), float(cfg.get("shake_decay", 0.12))
    anchors = {}
    print(f"fx bursts: {len(ev)} events" + (f" from {ev[0][0]:.1f} s to {ev[-1][0]:.1f} s" if ev else ""))

    def add(img, layer):
        return img + (layer[..., None] * col / 255 if img.ndim == 3 else layer)

    def apply(img, base, t):
        dx = dy = 0.0
        for k, (h, st) in enumerate(ev):
            age = t - h
            if not 0 <= age < 0.45:
                continue
            if k not in anchors:                          # anchor: the brightest region of the frame at the hit
                g = cv2.GaussianBlur(base if base.ndim == 2 else base.mean(axis=2), (0, 0), 25)
                y0, x0 = np.unravel_index(int(np.argmax(g)), g.shape)
                r = np.random.default_rng(ctx.get("seed", 7) * 1000 + k)
                anchors[k] = (x0, y0, r.uniform(0, 2 * np.pi, n_sp), r.uniform(0.35, 1.3, n_sp) * W,
                              r.uniform(0.6, 1.0, n_sp), r.uniform(0, 2 * np.pi))
            x0, y0, ang, spd, br, sdir = anchors[k]
            s = amp * (0.55 + 0.45 * st)
            e = s * np.exp(-age / 0.07)
            bloom = np.exp(-(((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * (0.16 * W * (1 + 2 * age)) ** 2)))
            img = add(img, bloom * 210 * e + 28 * e)
            sp = np.zeros((H, W), np.float32)
            for a1, v, b in zip(ang, spd, br):
                def pos(tt):
                    return x0 + np.cos(a1) * v * tt, y0 + np.sin(a1) * v * tt + 0.9 * H * tt * tt
                p0, p1 = pos(max(age - 1 / fps, 0)), pos(age)
                cv2.line(sp, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), float(b), 2, cv2.LINE_AA)
            img = add(img, cv2.GaussianBlur(sp, (0, 0), 1.2) * 255 * s * np.exp(-age / 0.18))
            if sh_px > 0:                                 # the knock: a damped oscillation in a random direction
                m = sh_px * (0.6 + 0.4 * st) * np.exp(-age / sh_dec) * np.sin(2 * np.pi * sh_hz * age + np.pi / 2)
                dx += m * np.cos(sdir); dy += m * np.sin(sdir)
        if dx or dy:
            z = 1 + 2.2 * (abs(dx) + abs(dy)) / W        # a touch of zoom so the shifted edge never shows
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
            M[:, 2] += (dx, dy)
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return img

    return apply
