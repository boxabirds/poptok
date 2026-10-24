"""Finish: rendered frames (or stills) -> graded video with the song: occasional pulse on real bass hits, grade, mux.

    python3 pipeline/finish.py OUT_DIR --audio SONG.wav [--analysis DIR]           # stills from OUT/keys, cuts on downbeats
    python3 pipeline/finish.py OUT_DIR --audio ... --frames-dir DIR --src-fps 24 # use rendered frames instead of stills

Per output frame (24 fps): shot = last downbeat <= t; zoom = product of beat_zoom.py's per-frame zoom since the shot
started (a punch on every beat, reset at each cut); a 2-frame flash on each cut; grade: black and white, contrast
S-curve, crushed shadows, per-frame film grain, vignette. Writes OUT/<name>.mp4 with the song.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import analysis_dir

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--audio", required=True)
ap.add_argument("--analysis", default=None)
ap.add_argument("--fps", type=float, default=24)
ap.add_argument("--size", default="1280x720")
ap.add_argument("--frames-dir", default=None)
ap.add_argument("--src-fps", type=float, default=24)
ap.add_argument("--zoom-file", default=None, help="beat_zoom.py output at --fps (default: ANALYSIS/beatzoom_24fps.txt)")
ap.add_argument("--no-cuts", action="store_true", help="frames-dir mode: don't reset zoom / flash at downbeats")
ap.add_argument("--no-zoom", action="store_true", help="don't add the beat zoom (the source already has it)")
ap.add_argument("--no-flash", action="store_true", help="no white flash on cuts")
ap.add_argument("--scene-cuts", action="store_true", help="cuts only where the shot prompt changes (not every downbeat)")
ap.add_argument("--zoom-mode", default="punch", choices=["punch", "cumulative"],
                help="punch: scale = 1 + G*(z-1.002) per frame (beat_zoom's values are Deforum per-frame feedback rates; "
                     "compounding them on finished frames gives ~3x per bar and a snap back at each cut)")
ap.add_argument("--punch-gain", type=float, default=1.5)
ap.add_argument("--punch-every", type=int, default=0, help="N>0: punch only on every Nth downbeat (N=2: the parity "
                "with the heavier low-end hit in the drums+bass stems), instead of beat_zoom's every-beat punches")
ap.add_argument("--punch-lowhits", action="store_true", help="punch on REAL low-frequency hits: per bar (downbeat to "
                "downbeat), the strongest onset below 120 Hz in drums+bass, kept only if it is in the top half of those; at "
                "most one per bar. (The beat grid's downbeats match a real bass hit only ~56%% of the time on one test song.)")
ap.add_argument("--punch-min-gap", type=float, default=1.5, help="lowhits: seconds between punches at least")
ap.add_argument("--no-pulse", action="store_true", help="no pulse at all (slow songs: user review 2026-10-24, the pulse "
                "felt like an accident on Slippery Road)")
ap.add_argument("--punch-snap", type=float, default=0.2, help="lowhits: move each hit onto the nearest beat if it is "
                "within this many seconds, drop it otherwise (0: no snapping). Detected hits on soft or slow songs land "
                "late or between beats and the pulse reads as random (Slippery Road, user review 2026-10-24)")
ap.add_argument("--punch-amp", type=float, default=0.05, help="downbeat punch size (scale 1+AMP at the hit)")
ap.add_argument("--punch-decay", type=float, default=0.35, help="downbeat punch decay, seconds")
ap.add_argument("--drift", type=float, default=0.0, help="slow camera move per shot: push from 1.0 to 1+DRIFT "
                "over the shot plus a pan of the same size, direction varying per shot (Ken Burns)")
ap.add_argument("--deflare", type=float, default=0.0, help="frames-dir mode: cap each frame's mean brightness at "
                "(local 2 s median + DEFLARE) by darkening it, to remove the video model's white light bursts")
ap.add_argument("--plan", default="shots.json")
ap.add_argument("--grade", default="bw", choices=["bw", "color", "treatment"], help="bw: black and white, S-curve, crushed "
                "shadows, grain. treatment: the plan's treatment.grade (from treatment.py): colour with "
                "saturation / warmth / fade / grain moving from its start to its end values over the song (or bw)")
ap.add_argument("--step", type=int, default=1, help="hold each source frame N output frames (2 = stop-motion on twos)")
ap.add_argument("--name", default="video")
ap.add_argument("--fx", default=None, help="a piece's effects config (JSON list; pipeline/fx/): bespoke per-frame effects "
                "applied after grading, e.g. bursts of light + a shake on chosen drum hits")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
W, H = (int(x) for x in a.size.split("x"))
plan = json.load(open(os.path.join(a.out, a.plan)))
dur = plan["duration"]
n = int(round(dur * a.fps))
song = a.audio
ANA = str(analysis_dir(song, a.analysis))
zf = a.zoom_file or os.path.join(ANA, "beatzoom_24fps.txt")
z = np.array([float(x) for x in re.findall(r"\(([-\d.eE]+)\)", open(zf).read())]) if os.path.exists(zf) else np.ones(1)
z = np.pad(z, (0, max(0, n - len(z))), constant_values=1.0)[:n]
sh = plan["shots"]
cut_bar = [i for i, s in enumerate(sh) if not a.scene_cuts or i == 0 or s["prompt"] != sh[i - 1]["prompt"]]
cuts = np.array([sh[i]["start"] for i in cut_bar])
def low_hits(db):
    """(time, strength) of the strongest <120 Hz onset in each bar of the drums+bass stems; weak bars dropped."""
    ana = ANA
    stem = os.path.splitext(os.path.basename(song))[0]
    x = 0
    for st in ("drums", "bass"):
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", os.path.join(ana, f"{stem}-{st}.mp3"), "-ac", "1",
                              "-ar", "4000", "-af", "lowpass=120", "-f", "f32le", "-"], capture_output=True).stdout
        x = x + np.frombuffer(raw, np.float32)
    hop = 40                                                            # 10 ms frames
    env = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same")[::hop] + 1e-12) + 1e-6)
    flux = np.maximum(np.diff(env, prepend=env[0]), 0)
    edges = list(db) + [len(flux) / 100]
    picks = []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        i0, i1 = int(b0 * 100), int(b1 * 100)
        if i1 - i0 > 2:
            k = i0 + int(np.argmax(flux[i0:i1])); picks.append((k / 100, flux[k]))
    cut = np.median([v for _, v in picks])
    kept = []
    for t, v in sorted(((t, v) for t, v in picks if v >= cut), key=lambda p: -p[1]):   # strongest first
        if all(abs(t - k) >= a.punch_min_gap for k, _ in kept):                         # adjacent-bar edge hits
            kept.append((t, v))
    return sorted(kept)


if a.no_pulse:
    z = np.ones(n)
    print("no pulse")
elif a.punch_lowhits:
    hits = low_hits(np.array(plan["downbeats"]))
    if a.punch_snap > 0 and plan.get("beats"):
        grid = np.array(plan["beats"])
        snapped = []
        for h, v in hits:
            b = float(grid[np.argmin(np.abs(grid - h))])
            if abs(b - h) <= a.punch_snap and all(abs(b - k) >= a.punch_min_gap for k, _ in snapped):
                snapped.append((b, v))
        print(f"pulse: {len(snapped)} of {len(hits)} low hits on the beat grid (within {a.punch_snap} s)")
        hits = snapped
    tt = np.arange(n) / a.fps
    z = np.ones(n)
    for h, _ in hits:
        m = tt >= h - 0.5 / a.fps                                       # the frame nearest the hit gets the peak
        z[m] = np.maximum(z[m], 1 + a.punch_amp * np.exp(-np.maximum(tt[m] - h, 0) / a.punch_decay))
    json.dump([round(h, 3) for h, _ in hits], open(os.path.join(a.out, f"{a.name}_punches.json"), "w"))
    print(f"punch on real low hits: {len(hits)} (max one per bar, {len(plan['downbeats'])} bars)")
elif a.punch_every > 0:                                                 # downbeat-only punch, replaces z
    db = np.array(plan["downbeats"])
    ana = ANA
    stem = os.path.splitext(os.path.basename(song))[0]
    low = None
    for st in ("drums", "bass"):                                        # low-end (<150 Hz) energy just after each downbeat
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", os.path.join(ana, f"{stem}-{st}.mp3"), "-ac", "1",
                              "-ar", "4000", "-af", "lowpass=150", "-f", "f32le", "-"], capture_output=True).stdout
        x = np.frombuffer(raw, np.float32)
        e = np.array([np.sqrt(np.mean(x[int(t * 4000):int((t + 0.15) * 4000)] ** 2) + 1e-12) for t in db])
        low = e if low is None else low + e
    ph = int(np.argmax([low[p::a.punch_every].mean() for p in range(a.punch_every)]))
    hits = db[ph::a.punch_every]
    tt = np.arange(n) / a.fps
    z = np.ones(n)
    for h in hits:
        m = tt >= h - 1e-6
        z[m] = np.maximum(z[m], 1 + a.punch_amp * np.exp(-(tt[m] - h) / a.punch_decay))
    print(f"punch on every {a.punch_every} downbeat(s), phase {ph}: {len(hits)} hits")
rng = np.random.default_rng(a.seed)
G = (plan.get("treatment", {}).get("grade", {}) if a.grade == "treatment" else
     {"mode": "color", "start": {"saturation": 1.0, "warmth": 0.0, "fade": 0.0, "grain": 0.3}} if a.grade == "color" else
     {"mode": "bw"})                # color: the keyframes' own colours, plus grain and vignette
COLOR = G.get("mode", "bw") == "color"
CREAM = np.array([236, 226, 206], np.float32)


def grade_at(u):
    """treatment grade parameters at song fraction u (linear from start to end)"""
    s0, s1 = G.get("start", {}), G.get("end", {})
    return {k: (1 - u) * float(s0.get(k, d)) + u * float(s1.get(k, s0.get(k, d)))
            for k, d in (("saturation", 0.8), ("warmth", 0.0), ("fade", 0.0), ("grain", 0.3))}

ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
vig = 1 - 0.35 * (((xs - W / 2) / (W / 2)) ** 2 + ((ys - H / 2) / (H / 2)) ** 2) ** 1.4
vig = np.clip(vig, 0.45, 1)
lut = np.clip(255 * (1 / (1 + np.exp(-(np.arange(256) / 255 - 0.45) * 6))), 0, 255)
lut = ((lut - lut.min()) / (lut.max() - lut.min()) * 245 + 6).astype(np.float32)  # gentle S-curve, lifted floor
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def cover(im):
    """resize+centre-crop a PIL image to W x H"""
    s = max(W / im.width, H / im.height)
    im = im.resize((int(im.width * s + 0.5), int(im.height * s + 0.5)), Image.LANCZOS)
    l, t = (im.width - W) // 2, (im.height - H) // 2
    if COLOR:                                                                 # colour: no level stretch / CLAHE
        return np.asarray(im.crop((l, t, l + W, t + H)).convert("RGB"), np.float32)
    g = np.asarray(im.crop((l, t, l + W, t + H)).convert("L"), np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)                        # per-image level stretch
    g = np.clip((g - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    return clahe.apply(g).astype(np.float32)                                  # local contrast (gritty detail)


FX = []
if a.fx:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fx as fxmod
    FX = fxmod.load(a.fx, {"W": W, "H": H, "fps": a.fps, "xs": xs, "ys": ys, "seed": a.seed,
                            "analysis": a.analysis or analysis_dir(song)})


keys = {}
pipe = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24" if COLOR else "gray", "-s", f"{W}x{H}",
                         "-r", str(a.fps), "-i", "-", "-i", song, "-shortest", "-c:a", "aac", "-c:v", "libx264", "-crf", "18",
                         "-pix_fmt", "yuv420p", os.path.join(a.out, f"{a.name}.mp4")], stdin=subprocess.PIPE)
cum = 1.0
prev_shot = -1
lm = []
for f in range(n):
    t = f / a.fps
    shot = int(np.searchsorted(cuts, t + 1e-6, side="right") - 1)
    if a.frames_dir:
        src = os.path.join(a.frames_dir, f"{min(int(t * a.src_fps) // a.step * a.step, 10**6):05d}.png")
        if not os.path.exists(src):
            break
        base = cover(Image.open(src))
        if a.deflare > 0:
            lm.append(base.mean()); ref = np.median(lm[-48:])
            if base.mean() > ref + a.deflare:
                base = base * ((ref + a.deflare) / base.mean())
    else:
        if shot not in keys:
            keys = {shot: cover(Image.open(os.path.join(a.out, "keys", f"shot_{cut_bar[shot]:03d}.png")))}
        base = keys[shot]
    new_shot = shot != prev_shot and not a.no_cuts
    cum = (1.0 if new_shot else cum * z[f]) if a.zoom_mode == "cumulative" else (z[f] if a.punch_every > 0 or a.punch_lowhits or a.no_pulse else 1 + a.punch_gain * (z[f] - 1.002))
    if a.drift > 0:
        sa, sb = cuts[shot], cuts[shot + 1] if shot + 1 < len(cuts) else dur
        u = (t - sa) / max(sb - sa, 1e-3); u = u * u * (3 - 2 * u)            # eased progress through the shot
        ang = shot * 2.39996                                                  # golden-angle: a new direction per shot
        dz = 1 + a.drift * u
        cx, cy = W / 2 + np.cos(ang) * a.drift * 0.5 * W * (u - 0.5), H / 2 + np.sin(ang) * a.drift * 0.5 * H * (u - 0.5)
    else:
        dz, cx, cy = 1.0, W / 2, H / 2
    if a.no_zoom:
        cum = 1.0
    if cum * dz > 1.0001:
        m = cv2.getRotationMatrix2D((cx, cy), 0, cum * dz)
        m[:, 2] += (W / 2 - cx, H / 2 - cy)
        base = cv2.warpAffine(base, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if COLOR:
        g = grade_at(t / dur)
        lum = base @ np.array([0.299, 0.587, 0.114], np.float32)
        img = lum[..., None] + g["saturation"] * (base - lum[..., None])
        img = img * np.array([1 + 0.10 * g["warmth"], 1 + 0.02 * g["warmth"], 1 - 0.12 * g["warmth"]], np.float32)
        img = img * (1 - 0.35 * g["fade"]) + 0.35 * g["fade"] * CREAM        # bleach toward pale cream
    else:
        img = lut[np.clip(base, 0, 255).astype(np.uint8)]
    frames_since_cut = int(round((t - cuts[shot]) * a.fps))
    if not a.no_cuts and not a.no_flash and frames_since_cut < 2:
        img = img + (90 if frames_since_cut == 0 else 40)                     # flash on the cut
    for effect in FX:                                                         # the piece's bespoke effects (--fx)
        img = effect(img, base, t)
    if COLOR:                                                                 # softer vignette, monochrome grain
        img = img * (0.55 + 0.45 * vig)[..., None] + rng.normal(0, 3 + 12 * g["grain"], (H, W, 1)).astype(np.float32)
    else:
        img = img * vig + rng.normal(0, 14, (H, W)).astype(np.float32)       # vignette + per-frame grain
    pipe.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
    prev_shot = shot
pipe.stdin.close(); pipe.wait()
print("wrote", os.path.join(a.out, f"{a.name}.mp4"), f"({f + 1} frames)")
