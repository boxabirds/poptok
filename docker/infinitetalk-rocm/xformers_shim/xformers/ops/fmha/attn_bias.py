"""Attention biases with xformers' semantics, materialised as dense boolean SDPA masks (True = attend)."""
from typing import Optional, Sequence
import torch


class AttentionBias:
    def materialize(self, shape, dtype=torch.float32, device="cpu"):
        raise NotImplementedError


class LowerTriangularMask(AttentionBias):
    """Causal mask (query i sees keys <= i), aligned top-left like xformers."""
    def to_bool(self, lq, lk, device):
        return torch.ones(lq, lk, dtype=torch.bool, device=device).tril()


class BlockDiagonalMask(AttentionBias):
    """xformers.ops.fmha.attn_bias.BlockDiagonalMask: the batch dimension is 1 and the queries/keys of several
    sequences are concatenated along M; query block i attends only to key block i."""
    def __init__(self, q_seqlen: Sequence[int], kv_seqlen: Sequence[int]):
        self.q_seqlen = [int(x) for x in q_seqlen]
        self.kv_seqlen = [int(x) for x in kv_seqlen]
        assert len(self.q_seqlen) == len(self.kv_seqlen), "q and kv need the same number of blocks"

    @classmethod
    def from_seqlens(cls, q_seqlen: Sequence[int], kv_seqlen: Optional[Sequence[int]] = None):
        return cls(q_seqlen, q_seqlen if kv_seqlen is None else kv_seqlen)

    def to_bool(self, lq, lk, device):
        assert lq == sum(self.q_seqlen) and lk == sum(self.kv_seqlen), \
            f"BlockDiagonalMask covers {sum(self.q_seqlen)}x{sum(self.kv_seqlen)}, inputs are {lq}x{lk}"
        qid = torch.repeat_interleave(torch.arange(len(self.q_seqlen), device=device),
                                      torch.tensor(self.q_seqlen, device=device))
        kid = torch.repeat_interleave(torch.arange(len(self.kv_seqlen), device=device),
                                      torch.tensor(self.kv_seqlen, device=device))
        return qid[:, None] == kid[None, :]

    def materialize(self, shape, dtype=torch.float32, device="cpu"):
        m = self.to_bool(shape[-2], shape[-1], device)
        out = torch.zeros(m.shape, dtype=dtype, device=device).masked_fill_(~m, float("-inf"))
        return out.expand(shape)
