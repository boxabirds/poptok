"""A legend of the Ice Walk layers (D32, 2026-10-28; user: "a sheet of the visuals for each layer? e.g. 'here is what a
swift with contrails looks like'"). layer_sheet.py shows full frames around an event, where a 40 px swift or a faint halo
can't be judged. Here each layer's best frame is rendered from the cached world twice, with the layer on and with it off
(its objects hidden; for the chimes, also the crystals' glow switched off), so everything else is the same pixels. Where the two
differ is the layer: the sheet crops round the difference and enlarges it. A layer whose difference is empty or tiny
doesn't reach the screen (the "where did the birds go?" class of problem), and its row says so.

    blender -b -P pipeline/world/layer_legend.py -- render SCENE.blend OUT CURVES.json LAYERS.json [--gpu] [--snow-cache DIR] [--only swifts,..]
    python3 pipeline/world/layer_legend.py sheet OUT SHEET.jpg [--tag T]
render: OUT/<layer>_on.png, <layer>_off.png (heartbeat: 3 frames, no off), OUT/picks.json (frame, why). sheet: one row per
layer: on | off | the difference | the zoomed crop, labelled 1a..8d with the layer and the song frame.
"""
import json, os, sys

LAYERS = ["swifts", "chime", "drone", "pad", "subbass", "clicks", "mist", "heartbeat"]
NAMES = {"heartbeat": "heartbeat: camera speed (motion; 3 frames)", "swifts": "tweets: swifts + contrails",
         "subbass": "sub-bass: ground drift", "pad": "chord change: cloud shadows", "chime": "chime: crystal lit",
         "drone": "drone: sun halo", "clicks": "click: crack glint", "mist": "mist"}


def render(argv):
    import argparse, math

    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ice_range
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("out"); ap.add_argument("curves"); ap.add_argument("layers")
    ap.add_argument("--gpu", action="store_true"); ap.add_argument("--snow-cache", default="")
    ap.add_argument("--only", default=",".join(LAYERS))
    a = ap.parse_args(argv)
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.scene), load_ui=False)
    sc = bpy.context.scene
    if a.gpu:
        ice_range.use_gpu(sc)
    if a.snow_cache:
        ice_range.point_snow(a.snow_cache)
    C, LY = json.load(open(a.curves)), json.load(open(a.layers)); N, FO, cam = sc["ice_N"], sc["ice_FO"], sc.camera
    speed = list(sc["ice_speed"]); out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    obs = lambda *pre: [o for o in bpy.data.objects if o.name.startswith(pre)]
    clip = lambda f: max(1, min(N, f))                     # event frame f (0-based, as in the JSONs) is clip frame f + 1

    def swift_frame():                  # the swift drawn largest on screen: in frame, away from the edges, nearest
        best, birds = (0, None, ""), obs("swift_bird")
        for h in C["tweets"]["hits"]:
            for k in range(clip(h + 2), clip(h + 9), 2):    # while it draws (DRAW 8 frames in ice_world.py)
                sc.frame_set(FO + k)
                for b_ in birds:
                    if b_.hide_render:
                        continue
                    v_ = world_to_camera_view(sc, cam, b_.matrix_world.translation)
                    if 0.15 <= v_.x <= 0.85 and 0.15 <= v_.y <= 0.85 and v_.z > 0 and 1 / v_.z > best[0]:
                        best = (1 / v_.z, k, f"{b_.name} {v_.z:.0f} m away")
        return best[1], best[2]

    def peak(v, why):
        i = max(range(min(N, len(v))), key=lambda i: v[i]); return clip(i + 1), f"{why} {v[i]:.3f}"

    def pick(layer):
        if layer == "swifts":
            return swift_frame()
        if layer == "chime":
            n_ = max(LY["chime"], key=lambda n: n["strength"]); return clip(n_["frame"] + 2), f"strongest note {n_['strength']}"
        if layer == "clicks":
            s_, h = max(zip(LY["clicks"]["strength"], LY["clicks"]["hits"])); return clip(h + 1), f"strongest click {s_}"
        if layer == "pad":              # mid-sweep after the change with the most pad energy
            h = max(LY["pad"]["changes"], key=lambda h: LY["pad"]["energy"][min(h, len(LY["pad"]["energy"]) - 1)])
            return clip(h + 15), f"change at {h + 1}"
        if layer == "drone":
            return peak(LY["drone"]["energy"], "loudest drone")
        if layer == "subbass":
            return peak(LY["subbass"], "loudest sub-bass")
        if layer == "mist":
            return peak(C["mist"], "densest mist")
        if layer == "heartbeat":        # the biggest surge: speed rises most over half a second
            i = max(range(N - 12), key=lambda i: speed[i + 12] - speed[i])
            return clip(i + 1), f"surge {speed[i]:.1f} -> {speed[i + 12]:.1f} m/s"

    def off(layer):                     # the layer gone, everything else as it was; returns how to put it back
        glow = []
        if layer == "chime":            # the crystals glow by object colour x a MULTIPLY (chime_glow); the lights are chime*
            for m in bpy.data.materials:
                nd = m.node_tree.nodes if m.use_nodes and m.node_tree else []
                if any(n.type == "OBJECT_INFO" for n in nd):
                    glow += [(n, n.inputs[1].default_value) for n in nd if n.type == "MATH" and n.operation == "MULTIPLY"]
            for n, _ in glow:
                n.inputs[1].default_value = 0
            hid = obs("chime")
        else:
            hid = obs({"swifts": "swift", "drone": "halo", "pad": "clouds", "subbass": "drift", "clicks": "crack",
                       "mist": "mist"}[layer])
        was = [(o, o.hide_render) for o in hid]
        for o in hid:
            o.hide_render = True
        print(f"legend {layer} off: {len(hid)} objects hidden, {len(glow)} glows off", flush=True)

        def back():
            for o, h in was:
                o.hide_render = h
            for n, v in glow:
                n.inputs[1].default_value = v
        return back

    def shot(k, path):
        sc.frame_set(FO + k); sc.render.filepath = path; bpy.ops.render.render(write_still=True)

    picks = json.load(open(f"{out}/picks.json")) if os.path.exists(f"{out}/picks.json") else {}
    for layer in a.only.split(","):
        k, why = pick(layer)
        if k is None:
            print(f"legend {layer}: no frame ({why or 'none of the layer on screen'})", flush=True)
            picks[layer] = {"frame": None, "why": why or "none on screen"}; continue
        print(f"legend {layer}: clip frame {k} ({why})", flush=True)
        if layer == "heartbeat":
            for j, d in enumerate((0, 6, 12)):
                shot(clip(k + d), f"{out}/heartbeat_{j}.png")
            picks[layer] = {"frame": k, "why": why, "frames": [clip(k + d) for d in (0, 6, 12)],
                            "speed": [round(speed[clip(k + d) - 1], 1) for d in (0, 6, 12)]}
        else:
            shot(k, f"{out}/{layer}_on.png"); back = off(layer); shot(k, f"{out}/{layer}_off.png"); back()
            picks[layer] = {"frame": k, "why": why}
        json.dump(picks, open(f"{out}/picks.json", "w"), indent=1)
    print("legend ok", out, flush=True)


def sheet(argv):
    import argparse

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("sheet"); ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    P = json.load(open(f"{a.out}/picks.json"))
    F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    W, H = 640, 360

    def lab(im, y, t):
        d = ImageDraw.Draw(im); d.rectangle((0, y - 3, d.textlength(t, font=F) + 10, y + 25), fill=0)
        d.text((5, y), t, font=F, fill=(255, 255, 0))

    rows = [l for l in LAYERS if l in P]
    S = Image.new("RGB", (W * 4, H * len(rows)))
    report = []
    for r, layer in enumerate(rows):
        p, tiles = P[layer], []
        if p["frame"] is None:
            im = Image.new("RGB", (W, H)); lab(im, 3, f"{r + 1}a {NAMES[layer]}"); lab(im, 40, f"NOT ON SCREEN: {p['why']}")
            S.paste(im, (0, r * H)); report.append((layer, "not on screen")); continue
        if layer == "heartbeat":
            for j, (f, v) in enumerate(zip(p["frames"], p["speed"])):
                im = Image.open(f"{a.out}/heartbeat_{j}.png").convert("RGB").resize((W, H))
                lab(im, 3, f"{r + 1}{'abc'[j]} f{f}" + (f"  {NAMES[layer]}" if j == 0 else "")); lab(im, 330, f"{v} m/s")
                tiles.append(im)
            report.append((layer, p["why"]))
        else:
            on = np.asarray(Image.open(f"{a.out}/{layer}_on.png").convert("RGB")).astype(np.float32)
            of = np.asarray(Image.open(f"{a.out}/{layer}_off.png").convert("RGB")).astype(np.float32)
            d = np.abs(on - of).max(axis=2)
            m = d > 8                                     # changed by more than 8/255 in some channel
            share, mean = m.mean(), d.mean()
            if m.any():                                   # the box round the changed pixels (2nd-98th percentile)
                ys, xs = np.nonzero(m)
                y0, y1 = np.percentile(ys, [2, 98]).astype(int); x0, x1 = np.percentile(xs, [2, 98]).astype(int)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                w = max(x1 - x0 + 40, (y1 - y0 + 40) * 16 / 9, 160); h = w * 9 / 16
                ih, iw = d.shape
                bx0 = int(min(max(cx - w / 2, 0), max(iw - w, 0))); by0 = int(min(max(cy - h / 2, 0), max(ih - h, 0)))
                box = (bx0, by0, int(min(bx0 + w, iw)), int(min(by0 + h, ih)))
            else:
                box = None
            ion = Image.fromarray(on.astype(np.uint8)); dim = Image.fromarray(np.clip(d * 4, 0, 255).astype(np.uint8)).convert("RGB")
            t_on = ion.resize((W, H)); t_off = Image.fromarray(of.astype(np.uint8)).resize((W, H)); t_d = dim.resize((W, H))
            if box:
                sx = W / on.shape[1]
                ImageDraw.Draw(t_on).rectangle([v * sx for v in box], outline=(255, 0, 255), width=2)
                t_z = ion.crop(box).resize((W, H), Image.LANCZOS)
            else:
                t_z = Image.new("RGB", (W, H))
            zoom = f"x{on.shape[1] / (box[2] - box[0]):.1f}" if box else "nothing changed"
            lab(t_on, 3, f"{r + 1}a f{p['frame']}  {NAMES[layer]}"); lab(t_on, 330, p["why"])
            lab(t_off, 3, f"{r + 1}b layer off"); lab(t_d, 3, f"{r + 1}c difference x4")
            lab(t_d, 330, f"{share * 100:.2f}% of pixels changed > 8/255")
            lab(t_z, 3, f"{r + 1}d on, zoomed {zoom}")
            tiles = [t_on, t_off, t_d, t_z]
            report.append((layer, f"f{p['frame']}: {share * 100:.2f}% of pixels changed, mean {mean:.2f}/255, crop {box}"))
        for j, t in enumerate(tiles):
            S.paste(t, (j * W, r * H))
    S.save(a.sheet, quality=88)
    for layer, s in report:
        print(f"{layer:10s} {s}")
    print(a.sheet)


if __name__ == "__main__":
    if "--" in sys.argv:                                # inside Blender
        args = sys.argv[sys.argv.index("--") + 1:]
        assert args[0] == "render", args; render(args[1:])
    else:
        assert sys.argv[1] == "sheet", sys.argv; sheet(sys.argv[2:])
