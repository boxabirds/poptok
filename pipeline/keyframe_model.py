"""Keyframe image model: MOSS-Image (OpenMOSS-Team/MOSS-Image-SFT, custom diffusers pipeline from its repo),
CFG 3 with block-22 skip (their default). Text-to-image, or img2img from an init image at noise level s0.
Weighted prompts [(text, weight)] are blended on the prediction (each is its own pass; the Euler step is linear).
Adapted from morphexpts/flow_renderer.py.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image


def _norm(ws):
    s = sum(ws)
    return [w / s for w in ws] if s > 0 else [1.0 / len(ws)] * len(ws)


class MOSSImage:
    repo = "/models/MOSS-Image-SFT" if os.path.isdir("/models/MOSS-Image-SFT") else "OpenMOSS-Team/MOSS-Image-SFT"

    def __init__(self, steps=30, cfg=3.0, negative="", offload=True):
        from diffusers import DiffusionPipeline
        custom = os.path.join(self.repo, "pipeline_moss_image.py") if os.path.isdir(self.repo) else self.repo
        self.pipe = DiffusionPipeline.from_pretrained(self.repo, custom_pipeline=custom, torch_dtype=torch.bfloat16,
                                                      trust_remote_code=True)
        self.pipe.vae.to(torch.float32)
        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.cfg, self.negative = steps, cfg, negative
        self._emb = {}

    def _embed(self, text):
        if text not in self._emb:
            self._emb[text] = self.pipe.encode_prompt(text)
        return self._emb[text]

    @torch.no_grad()
    def encode(self, img: Image.Image):
        p, vae = self.pipe, self.pipe.vae
        x = p.image_processor.preprocess(img.convert("RGB"), img.size[1], img.size[0]).to(p._execution_device, vae.dtype)
        z = vae.encode(x).latent_dist.mode()                                         # [1, c4, 2h, 2w]
        b, c4, H2, W2 = z.shape
        z = z.reshape(b, c4, H2 // 2, 2, W2 // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c4 * 4, H2 // 2, W2 // 2)
        bn, eps = vae.bn, vae.config.batch_norm_eps
        return (z - bn.running_mean.view(1, -1, 1, 1)) / (bn.running_var + eps).sqrt().view(1, -1, 1, 1)

    @torch.no_grad()
    def render(self, init: Image.Image | None, prompts, s0=1.0, seed=0, size=None):
        W, H = size or (init.size if init is not None else (1024, 1024))
        p, dev = self.pipe, self.pipe._execution_device
        texts, ws = [t for t, _ in prompts], _norm([w for _, w in prompts])
        conds = [self._embed(t) for t in texts]
        neg = self._embed(self.negative)
        sch = p.scheduler
        sch.set_timesteps(self.steps, device=dev)
        sig = sch.sigmas[:-1].float().cpu().numpy()
        k = int(np.argmax(sig <= s0 + 1e-6)) if (init is not None and s0 < 1) else 0
        g = torch.Generator("cpu").manual_seed(seed)
        shape = (1, p.transformer.config.latent_channels, H // 16, W // 16)
        noise = torch.randn(shape, generator=g, dtype=torch.float32).to(dev)
        s_start = float(sig[k])
        lat = noise if k == 0 and (init is None or s0 >= 1) else \
            (1 - s_start) * self.encode(init.resize((W, H), Image.LANCZOS)).float() + s_start * noise
        return self._denoise(lat, k, conds, ws, neg), s_start

    def _pred(self, lat, t, conds, ws, neg, cfg=None):
        p, dev = self.pipe, self.pipe._execution_device
        cfg = self.cfg if cfg is None else cfg
        pred = sum(w * p.transformer(lat, t, e.to(dev), m.to(dev)).sample.float() for (e, m), w in zip(conds, ws))
        if cfg != 1.0:
            un = p.transformer(lat, t, neg[0].to(dev), neg[1].to(dev), skip_layers=[22]).sample.float()
            pred = un + cfg * (pred - un)
        return pred

    @torch.no_grad()
    def _denoise(self, lat, k, conds, ws, neg):
        p, sch = self.pipe, self.pipe.scheduler
        sch.set_timesteps(self.steps, device=p._execution_device)
        sch.set_begin_index(k)
        for t in sch.timesteps[k:]:
            lat = sch.step(self._pred(lat, t, conds, ws, neg), t, lat, return_dict=False)[0]
        return self._decode(lat)

    @torch.no_grad()
    def _decode(self, lat):
        p = self.pipe
        vae = p.vae
        z = lat.to(vae.dtype)
        bn, eps = vae.bn, vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
        img = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
        return p.image_processor.postprocess(img, output_type="pil")[0]


def load(name="moss", **kw):
    return {"moss": MOSSImage}[name](**kw)
