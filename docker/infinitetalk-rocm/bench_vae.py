"""Time Wan2.1 VAE encode/decode of an 81-frame 480x832 clip on this GPU, with MIOpen on or off.
    python /opt/bench_vae.py [miopen|nomiopen]"""
import sys, time, torch
sys.path.insert(0, "/opt/InfiniteTalk")
from wan.modules.vae import WanVAE
torch.backends.cudnn.enabled = (sys.argv[1] != "nomiopen") if len(sys.argv) > 1 else True
vae = WanVAE(vae_pth="/weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth", device="cuda")
x = torch.randn(3, 81, 480, 832, device="cuda")
for rep in range(2):
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        z = vae.encode([x])
    torch.cuda.synchronize(); te = time.time() - t; t = time.time()
    with torch.no_grad():
        y = vae.decode(z)
    torch.cuda.synchronize(); print(sys.argv[1:], "rep", rep, "encode", round(te, 1), "s decode", round(time.time() - t, 1), "s", flush=True)
