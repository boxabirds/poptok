"""Stand-in for xformers on ROCm (gfx1151), where xformers has no working build. InfiniteTalk only calls
xformers.ops.memory_efficient_attention (audio cross-attention), optionally with a BlockDiagonalMask (sequence-parallel
path). Both are reimplemented on torch SDPA in ops/fmha/__init__.py, with the same [B, M, H, K] layout and semantics."""
__version__ = "0.0.0+sdpa-shim"
from . import ops  # noqa: F401
