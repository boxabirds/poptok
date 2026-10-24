"""Patch wan/multitalk.py so the quantized InfiniteTalk fits a 24 GB card with everything on the GPU (no CPU offload).
Two weights are dropped once they have done their job instead of being parked on the CPU:
  * T5 (fp8, ~6.7 GB) encodes the prompt once, then is deleted when --offload_model False (stock code keeps it on
    the GPU for the whole run, or moves it to the CPU when offloading).
  * CLIP's text tower (XLM-RoBERTa, ~1.1 GB fp16) is never used; InfiniteTalk only calls clip.visual().
  * With a still image as cond_video every streaming clip feeds CLIP the same frame, so its embedding is kept and the
    rest of CLIP freed after the first clip (a video as cond_video keeps CLIP).
And one activation peak is cut without changing the maths: the audio reference-attention map
(utils/multitalk_utils.get_attn_map_with_target) materialises softmax(q k_ref^T) for half the heads at a time (~2 GB
per copy at 896x448x81, several copies alive); it is computed one head at a time instead (the head mean is identical).
Likewise RoPE (modules/multitalk_model.rope_apply) rotates q and k in complex128 over the whole sequence at once
(~1.3 GB per temporary) and returns float32 q/k; it is applied in 4096-token slices, still in float64, into an
output of the input's dtype (bf16): both consumers (flash_attention and the reference-attention map) cast q/k to bf16
first anyway, so the values are the same and q/k take half the memory.
The quantized DiT's per-channel weight scales are stored in float32, which makes quanto return float32 from every
linear layer, so every activation (q/k/v, the FFN's 13824-wide hidden) is twice the size of the bf16 model's; the
scales are cast to bf16 at load, so the quantized model computes in bf16 like the unquantized one (the bf16 model's
weights carry the same 8-bit-mantissa rounding).
The VAE decoder grows its output with torch.cat once per latent frame (two full float32 copies of the clip alive at
the end); it writes into a preallocated output instead (same values). The VAE's convolutions also run under bf16
autocast instead of float32 (its latent mean/std and the returned frames stay float32): with the 19 GB DiT resident,
a float32 decode of an 81-frame 896x448 clip does not fit in what is left. Even in bf16 the decoder's causal
feature cache (2 frames of every conv input at full resolution, ~3.4 GB at 896x448) does not fit next to the DiT, so
the decode is tiled across the width: 48-latent-wide (384 px) tiles overlapping by 16 latents (128 px), each decoded
on its own and cross-faded linearly over the overlap. From the second streaming clip on, the VAE also encodes the
81-frame conditioning clip with the DiT resident, so encode is tiled the same way (384 px tiles, 128 px overlap,
cross-faded in latent space). bf16 VAE plus the two tilings are the numerical changes.
Logs GPU memory once the DiT is on the GPU.
    python3 vram_fit.py wan/multitalk.py wan/utils/multitalk_utils.py wan/modules/multitalk_model.py wan/modules/vae.py
"""
import sys

path = sys.argv[1]
s = open(path).read()
old_t5 = """            if offload_model:
                self.text_encoder.model.cpu()
"""
new_t5 = """            if offload_model:
                self.text_encoder.model.cpu()
            else:   # vram_fit: the prompt is encoded once; free T5 for the DiT instead of keeping it resident
                del self.text_encoder.model
                self.text_encoder.model = None
"""
assert s.count(old_t5) == 1, "T5 offload block not found"
s = s.replace(old_t5, new_t5)
old_clip = """            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))
"""
new_clip = old_clip + """        self.clip.model.textual = None   # vram_fit: text tower unused (only clip.visual is called)
        torch.cuda.empty_cache()
"""
assert s.count(old_clip) == 1, "CLIP init not found"
s = s.replace(old_clip, new_clip)
old_clipcall = """                self.clip.model.to(self.device)
                clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).to(self.param_dtype) 
                if offload_model:
                    self.clip.model.cpu()
"""
new_clipcall = """                if self.clip is None:   # vram_fit: still image, CLIP freed after the first clip
                    clip_context = self._clip_context
                else:
                    self.clip.model.to(self.device)
                    clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).to(self.param_dtype)
                    if offload_model:
                        self.clip.model.cpu()
                    elif os.path.splitext(cond_file_path)[1].lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
                        self._clip_context = clip_context
                        self.clip = None
"""
assert s.count(old_clipcall) == 1, "CLIP call not found"
s = s.replace(old_clipcall, new_clipcall)
old_sd = """            model_state_dict = load_file(quant_dir)
"""
new_sd = old_sd + """            model_state_dict = {k: v.to(self.param_dtype) if k.endswith('weight._scale') else v   # vram_fit
                                for k, v in model_state_dict.items()}
"""
assert s.count(old_sd) == 1, "quant load_file not found"
s = s.replace(old_sd, new_sd)
old_dit = """                if not self.vram_management:
                    self.model.to(self.device)
"""
new_dit = old_dit + """                    logging.info(f"vram_fit: DiT on GPU, allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB")
"""
assert s.count(old_dit) == 1, "DiT .to(device) not found"
s = s.replace(old_dit, new_dit)
open(path, "w").write(s)
print("patched", path)

path = sys.argv[2]
u = open(path).read()
old_split = "def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=2, enable_sp=False):"
assert u.count(old_split) == 1, "get_attn_map_with_target not found"
u = u.replace(old_split, old_split.replace("split_num=2", "split_num=40"))   # one head at a time (40 heads)
open(path, "w").write(u)
print("patched", path)

path = sys.argv[3]
m = open(path).read()
old_rope = """        freqs_i = freqs_i.to(device=x_i.device)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output).float()
"""
new_rope = """        freqs_i = freqs_i.to(device=x_i.device)
        out_i = x[i].clone()   # vram_fit: rotate in float64 slices; keep x's dtype (consumers cast to bf16 anyway)
        for a in range(0, seq_len, 4096):
            b = min(a + 4096, seq_len)
            xc = torch.view_as_complex(x_i[a:b].to(torch.float64).contiguous())
            out_i[a:b] = torch.view_as_real(xc * freqs_i[a:b]).flatten(2).to(out_i.dtype)
        output.append(out_i)
    return torch.stack(output) if len(output) > 1 else output[0][None]
"""
old_xi = """        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
"""
new_xi = """        x_i = x[i, :seq_len].reshape(seq_len, n, -1, 2)   # vram_fit: cast to complex128 per slice below
"""
assert m.count(old_rope) == 1 and m.count(old_xi) == 1, "rope_apply not found"
m = m.replace(old_rope, new_rope).replace(old_xi, new_xi)
open(path, "w").write(m)
print("patched", path)

path = sys.argv[4]
v = open(path).read()
old_dec = """        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out
"""
new_dec = """        iter_ = z.shape[2]
        x = self.conv2(z)
        out, t0 = None, 0   # vram_fit: preallocated output instead of a growing torch.cat
        for i in range(iter_):
            self._conv_idx = [0]
            out_ = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx)
            if out is None:
                out = out_.new_empty(out_.shape[0], out_.shape[1], 1 + 4 * (iter_ - 1), *out_.shape[3:])
            out[:, :, t0:t0 + out_.shape[2]] = out_
            t0 += out_.shape[2]
            del out_
        self.clear_cache()
        return out[:, :, :t0]
"""
assert v.count(old_dec) == 1, "VAE decode loop not found"
v = v.replace(old_dec, new_dec)
old_wrap = """    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
"""
new_wrap = """    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self._decode_tiled(u.unsqueeze(0)).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]

    def _decode_tiled(self, z, tile=48, overlap=16):   # vram_fit: width tiles, in latent px (x8 in pixels)
        W = z.shape[-1]
        if W <= tile:
            return self.model.decode(z, self.scale)
        starts = list(range(0, W - tile, tile - overlap)) + [W - tile]
        out = wsum = None
        for s0 in starts:
            o = self.model.decode(z[..., s0:s0 + tile].contiguous(), self.scale).float()
            f = o.shape[-1] // tile   # pixels per latent px
            w = torch.ones(o.shape[-1], device=o.device)
            ramp = torch.linspace(0, 1, overlap * f + 2, device=o.device)[1:-1]
            if s0 > 0:
                w[:overlap * f] = ramp
            if s0 + tile < W:
                w[-overlap * f:] = ramp.flip(0)
            if out is None:
                out = o.new_zeros(*o.shape[:-1], W * f)
                wsum = torch.zeros(W * f, device=o.device)
            out[..., s0 * f:(s0 + tile) * f] += o * w
            wsum[s0 * f:(s0 + tile) * f] += w
            del o
        return out / wsum
"""
old_enc = """    def encode(self, videos):
        \"\"\"
        videos: A list of videos each with shape [C, T, H, W].
        \"\"\"
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]
"""
new_enc = """    def encode(self, videos):
        \"\"\"
        videos: A list of videos each with shape [C, T, H, W].
        \"\"\"
        with amp.autocast(dtype=self.dtype):
            return [
                self._encode_tiled(u.unsqueeze(0)).float().squeeze(0)
                for u in videos
            ]

    def _encode_tiled(self, x, tile=48, overlap=16):   # vram_fit: width tiles, in latent px (x8 in pixels)
        f = 8
        W = x.shape[-1] // f
        if W <= tile:
            return self.model.encode(x, self.scale)
        starts = list(range(0, W - tile, tile - overlap)) + [W - tile]
        out = wsum = None
        for s0 in starts:
            o = self.model.encode(x[..., s0 * f:(s0 + tile) * f].contiguous(), self.scale).float()
            w = torch.ones(tile, device=o.device)
            ramp = torch.linspace(0, 1, overlap + 2, device=o.device)[1:-1]
            if s0 > 0:
                w[:overlap] = ramp
            if s0 + tile < W:
                w[-overlap:] = ramp.flip(0)
            if out is None:
                out = o.new_zeros(*o.shape[:-1], W)
                wsum = torch.zeros(W, device=o.device)
            out[..., s0:s0 + tile] += o * w
            wsum[s0:s0 + tile] += w
            del o
        return out / wsum
"""
assert v.count(old_enc) == 1, "WanVAE.encode not found"
v = v.replace(old_enc, new_enc)
assert v.count(old_wrap) == 1, "WanVAE.decode not found"
v = v.replace(old_wrap, new_wrap)
assert v.count("        with amp.autocast(dtype=self.dtype):") == 2, "VAE encode/decode autocast not found"
v = v.replace("        with amp.autocast(dtype=self.dtype):", "        with amp.autocast(dtype=torch.bfloat16):   # vram_fit")
open(path, "w").write(v)
print("patched", path)
