"""Check the xformers shim against an explicit softmax reference (dense and block-diagonal), on the GPU.
    python /opt/test_shim.py"""
import torch
import xformers.ops as xops
torch.manual_seed(0)
d = "cuda"

def ref(q, k, v, mask=None):   # [B, M, H, K], fp32 math
    q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))
    s = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return (s.softmax(-1) @ v).transpose(1, 2)

# InfiniteTalk single-person shape: (B*N_t) x 1560 visual tokens, 32 audio tokens per frame, 40 heads of 128
q = torch.randn(21, 1560, 40, 128, device=d, dtype=torch.bfloat16)
k = torch.randn(21, 32, 40, 128, device=d, dtype=torch.bfloat16)
v = torch.randn(21, 32, 40, 128, device=d, dtype=torch.bfloat16)
out = xops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
e = (out.float() - ref(q, k, v)).abs().max().item()
print("dense max err", e, out.shape); assert out.shape == q.shape and e < 2e-2

# block-diagonal (sequence-parallel path): 3 visual blocks, 3 audio blocks, concatenated in batch 1
qs, ks = [100, 70, 130], [32, 32, 32]
q = torch.randn(1, sum(qs), 8, 64, device=d, dtype=torch.bfloat16)
k = torch.randn(1, sum(ks), 8, 64, device=d, dtype=torch.bfloat16)
v = torch.randn(1, sum(ks), 8, 64, device=d, dtype=torch.bfloat16)
bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(qs, ks)
out = xops.memory_efficient_attention(q, k, v, attn_bias=bias)
# reference: each block separately
parts, qo, ko = [], 0, 0
for a, b in zip(qs, ks):
    parts.append(ref(q[:, qo:qo + a], k[:, ko:ko + b], v[:, ko:ko + b])); qo += a; ko += b
e = (out.float() - torch.cat(parts, 1)).abs().max().item()
print("block-diagonal max err", e); assert e < 2e-2
print("shim OK")
