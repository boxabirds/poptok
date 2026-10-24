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

import importlib, json


def load(config_path: str, ctx: dict) -> list:
    effects = []
    for cfg in json.load(open(config_path)):
        mod = importlib.import_module(f"fx.{cfg['effect']}")
        effects.append(mod.make(cfg, ctx))
    return effects
