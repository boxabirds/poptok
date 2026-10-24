"""Patch wan/modules/attention.py so flash_attention() works without the flash-attn package (ROCm copy of
docker/infinitetalk/sdpa_fallback.py, plus contiguous q/k/v: see _sdpa_attention).
Wan's own fallback, attention(), drops the q_lens/k_lens padding mask; this one keeps it as an SDPA key mask, so text
cross-attention still ignores T5's padding tokens. With no padding it passes no mask, so SDPA keeps its fast kernels.
    python3 sdpa_fallback.py /opt/InfiniteTalk/wan/modules/attention.py
"""
import sys

FALLBACK = '''

def _sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                    causal=False, dtype=torch.bfloat16):
    """flash_attention() on torch SDPA: q [B, Lq, Nq, C], k/v [B, Lk, Nk, C]; padding from q_lens/k_lens is masked."""
    out_dtype = q.dtype
    half = lambda x: x if x.dtype in (torch.float16, torch.bfloat16) else x.to(dtype)
    b, lq, nq, _ = q.shape
    lk, nk = k.shape[1], k.shape[2]
    if q_scale is not None:
        q = q * q_scale
    # contiguous [B, N, L, C]: on gfx1151 AOTriton's kernel is ~5.6x slower on the transposed view (6.6 s vs 1.17 s
    # for Wan's 32760-token self-attention); the copies cost milliseconds.
    q, k, v = (half(t).transpose(1, 2).contiguous() for t in (q, k, v))
    if nk != nq:
        k, v = (t.repeat_interleave(nq // nk, dim=1) for t in (k, v))
    mask = None
    if k_lens is not None and bool((k_lens.to(q.device) < lk).any()):
        mask = (torch.arange(lk, device=q.device)[None, :] < k_lens.to(q.device)[:, None])[:, None, None, :]
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p,
                                                         is_causal=causal and mask is None, scale=softmax_scale)
    x = x.transpose(1, 2)
    if q_lens is not None and bool((q_lens.to(q.device) < lq).any()):
        x = x * (torch.arange(lq, device=q.device)[None, :] < q_lens.to(q.device)[:, None])[:, :, None, None]
    return x.type(out_dtype)

'''

GUARD = '''    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return _sdpa_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, dtype)
'''

path = sys.argv[1]
s = open(path).read()
head = "\ndef flash_attention("
doc_end = '    """\n'
i = s.index(head)
s = s[:i] + FALLBACK + s[i:]
i = s.index(head)
j = s.index(doc_end, s.index('"""', i) + 3) + len(doc_end)   # end of flash_attention's docstring
s = s[:j] + GUARD + s[j:]
open(path, "w").write(s)
print("patched", path)
