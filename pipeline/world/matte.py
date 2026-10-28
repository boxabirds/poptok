"""Cut a person out of a photo: RGB in, RGBA out (alpha 0 where the background was), for TRELLIS.2 (her_mesh.py), which
uses the image's alpha as is (its own background remover, RMBG-2.0, is CC BY-NC and not loaded). BiRefNet (MIT; weights
ZhengPeng7/BiRefNet, MIT). The earlier cut-out of front_2 left the corners opaque (18% transparent), and TRELLIS built
the studio backdrop as a flat card behind her (2026-10-28).

    python3 pipeline/world/matte.py IN.png OUT_RGBA.png
Runs in the trellis2 image (transformers, timm, kornia, einops), HF_HOME=/models/hf.
"""
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation

src, out = sys.argv[1], sys.argv[2]
net = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).cuda().eval().float()
im = Image.open(src).convert("RGB")
x = np.asarray(im.resize((1024, 1024), Image.BICUBIC), dtype=np.float32) / 255.0
x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).cuda()
with torch.no_grad():
    m = net(x)[-1].sigmoid()[0, 0].cpu().numpy()
a = Image.fromarray((m * 255).clip(0, 255).astype(np.uint8)).resize(im.size, Image.BICUBIC)
rgba = im.copy(); rgba.putalpha(a); rgba.save(out)
al = np.asarray(a)
print("matte ok", out, "transparent %.2f opaque %.2f corners %.0f" % ((al < 8).mean(), (al > 247).mean(),
      np.mean([al[:16, :16].mean(), al[:16, -16:].mean(), al[-16:, :16].mean(), al[-16:, -16:].mean()])))
