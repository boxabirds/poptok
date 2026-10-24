"""Geometry checks for the plausibility critic (POC 28 step 1): text-prompted instance segmentation (LLMDet-large, a
Grounding-DINO-family open-vocabulary detector, boxes from text prompts -> SAM 2.1 masks; both Apache-2.0, via
transformers; plain Grounding DINO base is selectable with --det) and simple rules on the masks:

  loose hand        a hand mask that touches no (dilated) real-person mask
  detached hand     a hand inside a person's mask but not in the same connected piece as that person's body
  extra hand        more than 2 hand masks attached to one real person
  extra arm         more than 2 arms attached to one real person, counted on an upscaled crop of that person (hands
                    clasped together are often one detection, arms stay separate)
  reflection beside a face (or person) inside a puddle mask with no real person above it on roughly the same vertical
                    line (a horizontal surface reflects things straight below them, not off to the side)
  vehicle no rails  a train/tram mask with no rail-track mask under its bottom edge: ADVISORY only (written to "notes",
                    not "flags"): on the labelled set the open-vocabulary "rail track" prompt fires on wet plazas and
                    roads and a train window is taken for a train, so the rule is wrong both ways
  reflected person  person masks lying inside puddle/window masks are recorded (not flagged): they are reflections or
                    images, and their hands are not counted

Runs in the `segment` container (docker/segment), weights in ~/models/segment (mounted at /models):
    docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -v ~/models/segment:/models:ro -v $PWD:/work \
        segment python pipeline/segment_check.py IMG [...] --out DIR
Writes DIR/<name>.json (detections, masks summary, person boxes for critic.py --segment, flags) and DIR/<name>.png
(debug overlay: masks drawn and labelled, flags printed).
"""
from __future__ import annotations

import argparse, json, os, time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

# Prompt groups: one Grounding DINO pass per group. Vehicle classes share a pass so the model picks the best phrase
# for each box (a bus is then less often called a tram).
GROUPS = [
    (["person"], 0.35),
    (["face"], 0.35),
    (["hand"], 0.30),
    (["train", "tram", "bus", "car"], 0.35),
    (["rail track"], 0.30),
    (["puddle", "window"], 0.30),
]
REFLECTIVE = ("puddle", "window")
RAIL_VEHICLES = ("train", "tram")
LOOSE_HAND_MIN = 0.40  # weaker "hand" boxes are often bottles, gloves of light, faces


class Segmenter:
    def __init__(self, gd="/models/llmdet_large", sam="/models/sam2.1-hiera-large", dev="cuda"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.dev = dev
        self.gp = AutoProcessor.from_pretrained(gd)
        self.gm = AutoModelForZeroShotObjectDetection.from_pretrained(gd).to(dev).eval()
        self.sp = Sam2Processor.from_pretrained(sam)
        self.sm = Sam2Model.from_pretrained(sam).to(dev).eval()

    @torch.no_grad()
    def detect(self, img: Image.Image, groups=GROUPS, max_hand=0.15):
        dets = []
        for labels, thr in groups:
            text = " ".join(f"{l}." for l in labels)
            inp = self.gp(images=img, text=text, return_tensors="pt").to(self.dev)
            out = self.gm(**inp)
            r = self.gp.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=thr, text_threshold=0.25, target_sizes=[img.size[::-1]])[0]
            boxes, scores = r["boxes"].cpu(), r["scores"].cpu()
            names = r.get("text_labels", r.get("labels"))
            # map each phrase to one of this group's labels (DINO sometimes returns "train tram" or "")
            mapped = []
            for n in names:
                n = str(n).strip()
                hit = [l for l in labels if l == n] or [l for l in labels if l in n] or [labels[0]]
                mapped.append(hit[0])
            for l in set(mapped):
                idx = [i for i, m in enumerate(mapped) if m == l]
                if not idx:
                    continue
                b, s = boxes[idx], scores[idx]
                keep = nms(b, s, 0.5)
                for k in keep.tolist():
                    dets.append({"label": l, "score": round(float(s[k]), 3), "box": [round(float(v), 1) for v in b[k]]})
        # rail vehicles: a box called both train and tram (or bus) keeps the best label only
        veh = [d for d in dets if d["label"] in ("train", "tram", "bus", "car")]
        drop = set()
        for i, a in enumerate(veh):
            for b in veh[i + 1:]:
                if box_iou(a["box"], b["box"]) > 0.7:
                    drop.add(id(a if a["score"] < b["score"] else b))
        dets = [d for d in dets if id(d) not in drop]
        # a "hand" box covering most of a person is a whole-person detection, not a hand
        img_area = img.size[0] * img.size[1]
        dets = [d for d in dets if not (d["label"] == "hand" and box_area(d["box"]) > max_hand * img_area)]
        return dets

    @torch.no_grad()
    def masks(self, img: Image.Image, dets):
        if not dets:
            return []
        boxes = [[d["box"] for d in dets]]
        inp = self.sp(images=img, input_boxes=boxes, return_tensors="pt").to(self.dev)
        out = self.sm(**inp, multimask_output=False)
        m = self.sp.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]  # (n, 1, H, W)
        return [mm[0].numpy().astype(bool) for mm in m]


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = box_area(a) + box_area(b) - inter
    return inter / u if u else 0.0


def dilate(m, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def check(dets, masks, w, h, real_people, notes):
    """Geometry rules on the masks. Mutates dets with per-detection notes; returns flags (advisory ones go to notes)."""
    flags = []
    diag = (w * w + h * h) ** 0.5
    for d, m in zip(dets, masks):
        d["area"] = int(m.sum())
    refl = [m for d, m in zip(dets, masks) if d["label"] in REFLECTIVE]
    refl_union = np.any(refl, axis=0) if refl else np.zeros((h, w), bool)
    persons = [i for i, d in enumerate(dets) if d["label"] == "person" and d["area"] > 0]
    real, mirrored = [], []
    for i in persons:
        inside = (masks[i] & refl_union).sum() / dets[i]["area"]
        dets[i]["inside_reflective"] = round(float(inside), 2)
        (mirrored if inside > 0.7 else real).append(i)
        if inside > 0.7:
            dets[i]["note"] = "reflection/image (inside puddle/window)"
    # people's masks are drawn around the whole body, so hands touching the body mask attach to that person
    dil = {i: dilate(masks[i], max(3, int(0.012 * diag))) for i in real}
    attached = {i: [] for i in real}
    hands = [i for i, d in enumerate(dets) if d["label"] == "hand" and d["area"] > 0]
    # drop duplicate hands (same hand found twice)
    keep = []
    for i in sorted(hands, key=lambda i: -dets[i]["score"]):
        if all((masks[i] & masks[j]).sum() / min(dets[i]["area"], dets[j]["area"]) < 0.5 for j in keep):
            keep.append(i)
        else:
            dets[i]["note"] = "duplicate hand"
    for i in keep:
        hm, ha = masks[i], dets[i]["area"]
        in_mirror = max([(hm & masks[j]).sum() / ha for j in mirrored] or [0])
        if in_mirror > 0.6:
            dets[i]["note"] = "hand of a reflection/image"
            continue
        ov = {j: (hm & dil[j]).sum() / ha for j in real}
        best = max(ov, key=ov.get) if ov else None
        if best is None or ov[best] < 0.05:
            if dets[i]["score"] < LOOSE_HAND_MIN:
                dets[i]["note"] = "weak hand, touches no person (ignored)"
                continue
            dets[i]["note"] = "loose: touches no person"
            flags.append(f"loose hand at {box_str(dets[i]['box'])} (touches no person)")
        else:
            dets[i]["attached_to"] = best
            attached[best].append(i)
            # the hand must join the main body of that person's mask (SAM sometimes folds a stray hand into a
            # person's mask when the person's box spans it)
            k = max(2, int(0.004 * diag))
            n, lab = cv2.connectedComponents(dilate(masks[best] | hm, k).astype(np.uint8))
            if n > 2:
                body = np.bincount(lab[masks[best] & ~hm].ravel(), minlength=n)
                body[0] = 0
                hand_lab = np.bincount(lab[hm].ravel(), minlength=n)
                hand_lab[0] = 0
                if body.argmax() != hand_lab.argmax():
                    dets[i]["note"] = "not joined to the body of the person it touches"
                    flags.append(f"detached hand at {box_str(dets[i]['box'])}: not joined to the body of the person "
                                 f"at {box_str(dets[best]['box'])}")
    for j, hs in attached.items():
        dets[j]["hands"] = len(hs)
        if len(hs) > 2:
            flags.append(f"extra hand: {len(hs)} hands attached to the person at {box_str(dets[j]['box'])}")
    real_people[:] = real
    # reflections in puddles: a face/person inside a puddle must have its source above it on about the same vertical
    puddles = [m for d, m in zip(dets, masks) if d["label"] == "puddle"]
    pud = np.any(puddles, axis=0) if puddles else np.zeros((h, w), bool)
    for i, d in enumerate(dets):
        if d["label"] not in ("face", "person") or d["area"] == 0 or i in real:
            continue
        if (masks[i] & pud).sum() / d["area"] < 0.6:
            continue
        x0, y0, x1, y1 = d["box"]
        d["note"] = "in a puddle"
        src = [j for j in real if dets[j]["box"][1] < y0 and
               min(x1, dets[j]["box"][2]) - max(x0, dets[j]["box"][0]) > -0.5 * (x1 - x0)]
        if not src:
            d["note"] = "reflection with no source above it"
            flags.append(f"reflection beside: {d['label']} in a puddle at {box_str(d['box'])} has no person above it")
    rails = [i for i, d in enumerate(dets) if d["label"] == "rail track" and d["area"] > 0]
    for i, d in enumerate(dets):
        if d["label"] not in RAIL_VEHICLES or d["area"] == 0:
            continue
        x0, y0, x1, y1 = d["box"]
        vh = y1 - y0
        # band under the vehicle: its lowest 15% and down to 35% of its height below it
        band = np.zeros((h, w), bool)
        band[int(max(0, y1 - 0.15 * vh)):int(min(h, y1 + 0.35 * vh)), int(x0):int(x1)] = True
        band &= ~masks[i] | (np.arange(h)[:, None] > y1 - 0.15 * vh)
        cover = sum((masks[r] & band).sum() for r in rails) / max(1, band.sum())
        d["rails_under"] = round(float(cover), 3)
        if cover < 0.03:
            notes.append(f"vehicle without rails: {d['label']} at {box_str(d['box'])} (rail-track cover under it "
                         f"{cover:.0%})")
    return flags


PART_GROUPS = [(["arm"], 0.33), (["hand"], 0.30)]
MIRROR_GROUPS = [(["face"], 0.40), (["person"], 0.40)]


def puddle_reflections(seg, img, dets, masks, real):
    """A horizontal surface reflects things upside down: flip each puddle's crop vertically, upscale it and look for
    faces/people. Each one found must have a real person above it on about the same vertical line."""
    W, H = img.size
    flags = []
    for i, d in enumerate(list(dets)):
        if d["label"] != "puddle" or d.get("area", 0) < 0.003 * W * H:
            continue
        x0, y0, x1, y1 = d["box"]
        crop = img.crop((int(x0), int(y0), int(x1), int(y1))).transpose(Image.FLIP_TOP_BOTTOM)
        sc = 768 / max(crop.size)
        crop = crop.resize((max(1, round(crop.width * sc)), max(1, round(crop.height * sc))), Image.LANCZOS)
        found = seg.detect(crop, MIRROR_GROUPS)
        # the water's mask often leaves the reflected figure out as a hole: test against its convex hull
        hull = np.zeros((H, W), np.uint8)
        cnts, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cnts))], 1)
        for f in found:
            bx0, by0, bx1, by1 = f["box"]
            # back to frame coordinates (undo scale, then the vertical flip)
            fb = [x0 + bx0 / sc, y1 - by1 / sc, x0 + bx1 / sc, y1 - by0 / sc]
            if box_area(fb) > 0.8 * box_area(d["box"]):
                continue  # the whole puddle called a person: not a figure
            cx0, cy0, cx1, cy1 = (int(v) for v in fb)
            inpud = hull[cy0:cy1 + 1, cx0:cx1 + 1].mean() if cy1 > cy0 and cx1 > cx0 else 0
            if inpud < 0.5:
                continue
            rec = {"label": "reflected " + f["label"], "score": f["score"], "box": [round(v, 1) for v in fb],
                   "in_puddle": i, "area": 0}
            src = [j for j in real if dets[j]["box"][1] < fb[1] and
                   min(fb[2], dets[j]["box"][2]) - max(fb[0], dets[j]["box"][0]) > -0.5 * (fb[2] - fb[0])]
            rec["note"] = "reflection, source above" if src else "reflection with no source above it"
            dets.append(rec)
            masks.append(np.zeros((H, W), bool))
            if not src:
                flags.append(f"reflection beside: a {f['label']} reflected in the puddle at {box_str(fb)} has no "
                             f"person above it")
    return flags


def distinct(parts):
    """One box per body part: drop boxes that group two or more others, then keep the best of overlapping boxes."""
    def inside(a, b):  # fraction of a inside b
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / max(1.0, box_area(a))
    parts = [p for p in parts if sum(inside(q["box"], p["box"]) > 0.7 for q in parts if q is not p) < 2]
    keep = []
    for p in sorted(parts, key=lambda p: -p["score"]):
        if all(box_iou(p["box"], k["box"]) < 0.5 and inside(p["box"], k["box"]) < 0.7 and inside(k["box"], p["box"]) < 0.7
               for k in keep):
            keep.append(p)
    return keep


def person_parts(seg, img, dets, real, min_h=0.3):
    """Count arms and hands per real person on a crop of the person's box (+25% margin) upscaled to 1024 on the long
    side. Every real person overlapping the crop is segmented too; a part counts for the person whose mask it
    overlaps most (at least half of the part). Returns flags; writes "arms"/"hands_crop" into dets."""
    W, H = img.size
    flags = []
    for j in real:
        b = dets[j]["box"]
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bh < min_h * H:
            continue
        x0, y0 = max(0, b[0] - 0.25 * bw), max(0, b[1] - 0.25 * bh)
        x1, y1 = min(W, b[2] + 0.25 * bw), min(H, b[3] + 0.25 * bh)
        crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
        sc = 1024 / max(crop.size)
        crop = crop.resize((round(crop.width * sc), round(crop.height * sc)), Image.LANCZOS)
        tocrop = lambda bb: [(bb[0] - int(x0)) * sc, (bb[1] - int(y0)) * sc, (bb[2] - int(x0)) * sc, (bb[3] - int(y0)) * sc]
        ppl = [k for k in real if box_iou(dets[k]["box"], [x0, y0, x1, y1]) > 0 and
               (k == j or (min(b[2], dets[k]["box"][2]) > max(b[0], dets[k]["box"][0])))]
        parts = seg.detect(crop, PART_GROUPS, max_hand=0.25)
        parts = [p for p in parts if box_area(p["box"]) < 0.35 * crop.width * crop.height]
        parts = distinct([p for p in parts if p["label"] == "arm"]) + distinct([p for p in parts if p["label"] == "hand"])
        pdets = [{"box": tocrop(dets[k]["box"])} for k in ppl] + parts
        ms = seg.masks(crop, pdets)
        pm = {k: ms[n] for n, k in enumerate(ppl)}
        counts = {"arm": [], "hand": []}
        for p, m in zip(parts, ms[len(ppl):]):
            a = max(1, m.sum())
            ov = {k: (m & pm[k]).sum() / a for k in ppl}
            best = max(ov, key=ov.get)
            if best == j and ov[best] >= 0.5:
                counts[p["label"]].append({"score": p["score"], "box_full": [round(v / sc + o, 1) for v, o in
                                           zip(p["box"], (int(x0), int(y0), int(x0), int(y0)))]})
        dets[j]["arms"] = len(counts["arm"])
        dets[j]["hands_crop"] = len(counts["hand"])
        dets[j]["parts"] = counts
        if len(counts["arm"]) > 2:
            flags.append(f"extra arm: {len(counts['arm'])} arms on the person at {box_str(b)}")
        if len(counts["hand"]) > 2:
            flags.append(f"extra hand: {len(counts['hand'])} hands on the person at {box_str(b)} (crop)")
    return flags


def box_str(b):
    return "[" + ",".join(str(int(v)) for v in b) + "]"


COLORS = {"person": (60, 200, 60), "hand": (255, 60, 60), "train": (255, 160, 0), "tram": (255, 220, 0),
          "bus": (200, 120, 255), "reflected face": (255, 0, 255), "reflected person": (255, 0, 255), "car": (120, 180, 255), "rail track": (0, 255, 255), "puddle": (60, 120, 255),
          "window": (180, 180, 180), "face": (255, 120, 200)}


def overlay(img: Image.Image, dets, masks, flags, path):
    a = np.array(img.convert("RGB")).astype(np.float32)
    order = sorted(range(len(dets)), key=lambda i: -dets[i].get("area", 0))  # big masks first, small on top
    for i in order:
        c = np.array(COLORS.get(dets[i]["label"], (255, 255, 255)), np.float32)
        a[masks[i]] = a[masks[i]] * 0.55 + c * 0.45
    a = a.astype(np.uint8)
    for i in order:
        d = dets[i]
        c = COLORS.get(d["label"], (255, 255, 255))
        x0, y0, x1, y1 = map(int, d["box"])
        cv2.rectangle(a, (x0, y0), (x1, y1), c, 1)
        cnts, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(a, cnts, -1, c, 1)
        t = f"{i}:{d['label']} {d['score']:.2f}" + (f" h={d['hands']}" if "hands" in d else "") + (f" arms={d['arms']}" if "arms" in d else "") + \
            (" [refl]" if "reflection" in d.get("note", "") else "") + (" LOOSE" if "loose" in d.get("note", "") else "")
        cv2.putText(a, t, (x0 + 2, max(10, y0 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(a, t, (x0 + 2, max(10, y0 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    y = a.shape[0] - 8 - 16 * max(0, len(flags) - 1)
    for f in flags or ["no geometry flags"]:
        cv2.putText(a, f, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(a, f, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 80) if flags else (120, 255, 120), 1,
                    cv2.LINE_AA)
        y += 16
    Image.fromarray(a).save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--det", default="/models/llmdet_large", help="open-vocabulary detector (transformers checkpoint)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    seg = Segmenter(gd=a.det)
    torch.cuda.synchronize()
    print(f"models loaded in {time.time() - t0:.1f} s", flush=True)
    for p in a.images:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        t1 = time.time()
        dets = seg.detect(img)
        t2 = time.time()
        masks = seg.masks(img, dets)
        torch.cuda.synchronize()
        t3 = time.time()
        real, notes = [], []
        flags = check(dets, masks, w, h, real, notes)
        flags += person_parts(seg, img, dets, real)
        flags += puddle_reflections(seg, img, dets, masks, real)
        torch.cuda.synchronize()
        t4 = time.time()
        name = os.path.splitext(os.path.basename(p))[0]
        overlay(img, dets, masks, flags, os.path.join(a.out, name + ".png"))
        people = [{"idx": i, "box": d["box"], "score": d["score"], "hands": d.get("hands"), "arms": d.get("arms"),
                   "hands_crop": d.get("hands_crop"),
                   "reflection": "reflection" in d.get("note", "")} for i, d in enumerate(dets) if d["label"] == "person"]
        rec = {"image": p, "size": [w, h], "ok": not flags, "flags": flags, "notes": notes, "people": people, "detections": dets,
               "seconds": {"detect": round(t2 - t1, 3), "masks": round(t3 - t2, 3), "person_crops": round(t4 - t3, 3),
                           "total": round(t4 - t1, 3)},
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        json.dump(rec, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
        print(f"{p}: {'OK' if not flags else 'FLAG'} ({t4 - t1:.2f} s) " + "; ".join(flags) + (f"  [notes: {'; '.join(notes)}]" if notes else ""), flush=True)
