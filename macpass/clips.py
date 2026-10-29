"""poptok/macpass step 3: one clip per shot, macvideo's render_clips.py with poptok's LTX-2.5 (distilled, on a Spark)
in place of mlx-video: the keyframe animated from frame 0 by its camera intent and beat, no transition and no LoRA (the
shots meet in plain cuts, as in macvideo), 1920x1088 (macvideo's "final" size). A clip already made is kept.
    python macpass/clips.py PIECE [--part 0/2]      (--part i/n: every n-th shot from i, to share the shots across boxes)
-> PIECE/macpass/clips/clip_NNN.mp4"""
import argparse, json, os, subprocess, sys

import torch

import ltx_pipelines.distilled as D
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode

W, H = 1920, 1088
# macvideo's camera vocabulary (render_clips.CAMERA_LANGUAGE) and prompt (build_prompt), the style as a drawing that
# moves: LTX turned cuckoo's charcoal into a photograph within 2.5 s unless the clip is named a hand-drawn animation
# (probe ltx_style.py, 2026-10-29)
CAMERA = {
    "static hold": "static camera, subject held steady in frame",
    "slow push in": "slow continuous forward dolly push-in, camera moves toward subject",
    "slow pull back": "slow continuous dolly pull-back, camera retreats from subject",
    "pan left": "slow horizontal pan to the left, smooth camera motion",
    "pan right": "slow horizontal pan to the right, smooth camera motion",
    "tilt up": "slow tilt upward, smooth camera motion",
    "tilt down": "slow tilt downward, smooth camera motion",
    "orbit left": "slow orbital motion around subject to the left",
    "orbit right": "slow orbital motion around subject to the right",
    "handheld drift": "gentle handheld drift, soft natural camera motion",
    "held on detail": "camera held on fine detail, very slow drift",
}

ap = argparse.ArgumentParser()
ap.add_argument("piece")
ap.add_argument("--part", default="0/1")
ap.add_argument("--only", default="", help="shot indices, e.g. 1,2 (a dry run)")
a = ap.parse_args()
O = os.path.join(a.piece.rstrip("/"), "macpass"); C = os.path.join(O, "clips"); os.makedirs(C, exist_ok=True)
plan = json.load(open(f"{O}/shots.json")); fps = plan["fps"]
sb = {s["index"]: s for s in json.load(open(f"{O}/storyboard.json"))["shots"]}
filt = json.load(open(f"{O}/brief.json"))["filter"]
i, n = (int(x) for x in a.part.split("/"))
only = {int(x) for x in a.only.split(",") if x}
todo = [s for s in plan["shots"] if (s["index"] - 1) % n == i and (not only or s["index"] in only)
        and not os.path.exists(f"{C}/clip_{s['index']:03d}.mp4")]
if not todo:
    sys.exit(0)

MODELS = os.environ.get("LTX_MODELS", "/models"); M = f"{MODELS}/ltx-2.5"
paths = ModelPaths.from_split(transformer_path=f"{M}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                              text_encoder_path=f"{M}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                              video_vae_path=f"{M}/vae/ltx-2.5-video-vae-bf16.safetensors",
                              audio_vae_path=f"{M}/vae/ltx-2.5-audio-vae-bf16.safetensors")
pipe = D.DistilledPipeline(model_paths=paths,
                           spatial_upsampler_path=f"{M}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                           loras=(), quantization=None, offload_mode=OffloadMode.NONE)
for s in todo:
    b = sb.get(s["index"], {})
    prompt = " ".join(x for x in [
        f"A 2D hand-drawn animation, {filt}: a drawing that moves, drawn frames, not live action.",
        CAMERA.get(b.get("camera_intent"), b.get("camera_intent") or CAMERA["static hold"]) + ".",
        f"Subject: {b['subject_focus']}." if b.get("subject_focus") else "",
        b.get("beat", ""),
        f"Rendered in {filt} style, consistent throughout the clip; it stays a hand-drawn drawing, never live action."]
        if x)
    key = f"{O}/keyframes/keyframe_{s['index']:03d}.png"
    with torch.inference_mode():
        out = pipe(prompt=prompt, seed=42 + s["index"], height=H, width=W, frame_rate=fps, num_frames=s["num_frames"],
                   images=[ImageConditioningInput.at(path=key, frame_idx=0, strength=1.0)])
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                               f"{C}/clip_{s['index']:03d}.mp4"], stdin=subprocess.PIPE)
        for seg in out.video:
            seg = seg.float().cpu()
            seg = seg * 255 if seg.max() <= 1.5 else seg
            for f in seg.clamp(0, 255).to(torch.uint8).numpy():
                ff.stdin.write(f.tobytes())
        ff.stdin.close(); ff.wait()
    print(f"clip {s['index']}: {s['num_frames']} frames, {b.get('camera_intent')}: {s['target_text']}", flush=True)
