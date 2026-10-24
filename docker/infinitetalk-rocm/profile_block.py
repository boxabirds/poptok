"""Profile one InfiniteTalk DiT block (random bf16 weights) at the real 480p/81-frame shapes on this GPU.
    python /opt/profile_block.py"""
import sys, time, torch
sys.path.insert(0, "/opt/InfiniteTalk")
from wan.modules.multitalk_model import WanAttentionBlock, rope_params
torch.manual_seed(0)
d, dt = "cuda", torch.bfloat16
blk = WanAttentionBlock("i2v_cross_attn", 5120, 13824, 40, (-1, -1), True, True, 1e-6, output_dim=768).to(d, dt)
F_, H_, W_ = 21, 30, 52
L = F_ * H_ * W_
x = torch.randn(1, L, 5120, device=d, dtype=dt)
e = torch.randn(1, 6, 5120, device=d, dtype=torch.float32)
dd = 128
freqs = torch.cat([rope_params(1024, dd - 4 * (dd // 6)), rope_params(1024, 2 * (dd // 6)), rope_params(1024, 2 * (dd // 6))], dim=1).to(d)
kw = dict(e=e, seq_lens=torch.tensor([L], device=d), grid_sizes=torch.tensor([[F_, H_, W_]], device=d), freqs=freqs,
          context=torch.randn(1, 257 + 512, 5120, device=d, dtype=dt), context_lens=None,
          audio_embedding=torch.randn(1, F_, 32, 768, device=d, dtype=dt),
          ref_target_masks=torch.ones(1, H_ * W_, device=d, dtype=dt), human_num=1)
with torch.no_grad():
    for i in range(3):
        torch.cuda.synchronize(); t = time.time(); blk(x, **kw); torch.cuda.synchronize()
        print("block forward", round(time.time() - t, 2), "s", flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        blk(x, **kw); torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
print(p.key_averages().table(sort_by="cpu_time_total", row_limit=15, max_name_column_width=70))
