"""memory_efficient_attention on torch SDPA. xformers layout: q [B, Mq, H, K], k/v [B, Mk, H, K] -> [B, Mq, H, K]."""
import torch
import torch.nn.functional as F
from . import attn_bias  # noqa: F401
from .attn_bias import AttentionBias, BlockDiagonalMask, LowerTriangularMask


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, op=None, output_dtype=None):
    if query.ndim == 3:   # [B, M, K]: single head
        return memory_efficient_attention(query[:, :, None], key[:, :, None], value[:, :, None], attn_bias, p, scale,
                                          op, output_dtype)[:, :, 0]
    q, k, v = (t.transpose(1, 2).contiguous() for t in (query, key, value))   # -> [B, H, M, K]; contiguous is
    # much faster with ROCm's AOTriton kernels than the transposed view
    if k.shape[1] != q.shape[1]:   # grouped/multi-query keys
        k, v = (t.repeat_interleave(q.shape[1] // t.shape[1], dim=1) for t in (k, v))
    mask, causal = None, False
    if isinstance(attn_bias, LowerTriangularMask):
        causal = True
    elif isinstance(attn_bias, BlockDiagonalMask):
        mask = attn_bias.to_bool(q.shape[2], k.shape[2], q.device)
    elif isinstance(attn_bias, torch.Tensor):
        mask = attn_bias.to(q.dtype) if attn_bias.dtype != torch.bool else attn_bias
    elif attn_bias is not None:
        raise NotImplementedError(f"xformers shim: attn_bias {type(attn_bias).__name__}")
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p, is_causal=causal, scale=scale)
    x = x.transpose(1, 2)
    return x if output_dtype is None else x.to(output_dtype)
