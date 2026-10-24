"""Patch wan/modules/vae.py so CausalConv3d runs as a sum of 2D convolutions (one per temporal kernel tap, frames folded
into the batch). Exact same math as nn.Conv3d; on gfx1151 MIOpen's 3D convolutions are very slow (Wan VAE encode of an
81-frame 480p clip: ~104 s warm, ~450 s cold) while its 2D ones are well tuned.
    python3 conv3d_as_conv2d.py /opt/InfiniteTalk/wan/modules/vae.py
Set WAN_VAE_CONV3D=1 at run time to use the stock nn.Conv3d path instead."""
import sys

HELPER = '''

_CONV3D_NATIVE = os.environ.get("WAN_VAE_CONV3D", "0") == "1"


def _conv3d_via_conv2d(m, x):
    """nn.Conv3d forward (padding already applied, so m.padding is 0) as sum_t conv2d(x[:, :, t::st], w[:, :, t])."""
    kt = m.kernel_size[0]
    st, sh, sw = m.stride
    dt, dh, dw = m.dilation
    b, c, t, h, w = x.shape
    t_out = (t - dt * (kt - 1) - 1) // st + 1
    out = None
    for i in range(kt):
        s = i * dt
        xs = x[:, :, s:s + st * (t_out - 1) + 1:st]                      # B, C, T_out, H, W
        xs = xs.transpose(1, 2).reshape(b * t_out, c, h, w)
        y = F.conv2d(xs, m.weight[:, :, i], None, (sh, sw), 0, (dh, dw), m.groups)
        out = y if out is None else out.add_(y)
    if m.bias is not None:
        out = out + m.bias.view(1, -1, 1, 1)
    return out.view(b, t_out, *out.shape[1:]).transpose(1, 2).contiguous()

'''

path = sys.argv[1]
s = open(path).read()
assert "import torch.nn.functional as F" in s
if "import os\n" not in s:
    s = "import os\n" + s
anchor = "\nclass CausalConv3d(nn.Conv3d):"
s = s.replace(anchor, HELPER + anchor, 1)
old = "        x = F.pad(x, padding)\n\n        return super().forward(x)\n"
assert s.count(old) == 1, "CausalConv3d.forward not as expected"
s = s.replace(old, "        x = F.pad(x, padding)\n\n        if _CONV3D_NATIVE or tuple(self.padding) != (0, 0, 0):\n"
                   "            return super().forward(x)\n        return _conv3d_via_conv2d(self, x)\n")
open(path, "w").write(s)
print("patched", path)
