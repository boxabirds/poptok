"""Plausibility critic (POC 28): the local Qwen3.8-27B with its vision projector looks at stills or video frames and
reports physically impossible things (extra or disconnected hands, the same person twice, vehicles without rails or
road, reflections that don't fit the geometry, floating or merged objects). It asks countable questions (people, hands,
vehicles and what they stand on, reflections and where they are) rather than "is anything wrong?", and decides with
simple rules on the answers, so a miss or a flag can be traced to a specific answer.

    python3 pipeline/critic.py IMAGE[::INTENT] [...] [--json OUT.json] [--llm URL] [--crops]
INTENT (optional, after "::") is what the scene was meant to show (its prompt): intended surreal elements are then
not flagged.
Needs llama-server started with --mmproj (the vision projector), e.g.
    llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf --mmproj mmproj-F16.gguf -ngl 99 -c 16384 --reasoning-budget 0 --port 8091
--crops also asks about the left and right 60% of the image at 1.6x, and unions the flags (small hands are easy to miss).
--segment DIR adds the geometry checks of pipeline/segment_check.py (run it first, in the `segment` container; it writes
DIR/<name>.json) and asks one narrow question per real person it found: the person's box +25% margin, upscaled to 1024 on
the long side, "how many hands and arms does this person have; does each hand connect by an arm to this body?". Flags
from all three (whole frame, geometry, person crops) are unioned.
"""
from __future__ import annotations

import argparse, base64, io, json, os, re, sys, time, urllib.request

from PIL import Image

ASK = """Look at this frame from a photographic music video very carefully, and answer with facts, not impressions.

1. people: list every person (or part of a person) you can see, including reflections and faces on screens/billboards.
   For each: "where" (short), "is_reflection_or_image" (true if it is a reflection, a photo, a screen or a billboard),
   "hands_visible" (count every hand you can see that belongs to this person), "hands_with_visible_arm" (how many of
   those hands connect to an arm that leads to this person's body).
2. loose_hands: the number of hands in the image that do not clearly belong to any listed person's body.
3. same_person_twice: true if the same real person appears twice as two separate bodies (a correct mirror reflection
   does not count).
4. vehicles: every train, tram, bus or car. For each: "type", "on" (what it is standing or running on: rails, road,
   pavement, plaza, water, air...), "rails_visible_under_it" (true/false, for trains and trams).
5. reflections: every reflection of a person or object. For each: "of", "where", "geometry_ok" (true if its position,
   orientation and size are where a real reflection would be, given the surface and the camera angle).
6. impossible_objects: anything floating without support, merged into something else, cut off unnaturally, or
   appearing with no plausible source. Short descriptions; empty list if none.

Return JSON only, with exactly these keys: people, loose_hands, same_person_twice, vehicles, reflections,
impossible_objects."""


def b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


INTENT = """
The scene was meant to show: "{ctx}"
Things this description asks for (however surreal) are intended: do not list them as impossible. Report only what
the description does not ask for."""


PERSON = """This is a crop around one person from a frame of a photographic music video. Look only at the person in the
centre of the crop (the main figure), and count carefully, as facts:
1. hands: every hand that belongs to this person or lies next to them (on their body, face, lap, or on the ground or a
   surface beside them). For each: "where" (short), "arm_connects_to_body" (true if an arm runs from this hand to this
   person's shoulder, or the arm leaves the crop at the edge in a direction consistent with this person's body; false
   if the hand has no arm, or its arm cannot be this person's).
2. arms: how many arms this person has in total (count each separately visible arm/forearm, even if it looks wrong).
Return JSON only: {"hands": [{"where": "...", "arm_connects_to_body": true}], "arms": 2}"""


def person_crops(img: Image.Image, seg: dict, min_h=0.2):
    """Crops of each real (not reflected) person from segment_check's JSON: box + 25% margin, 1024 on the long side."""
    W, H = img.size
    for p in seg.get("people", []):
        if p.get("reflection"):
            continue
        x0, y0, x1, y1 = p["box"]
        bw, bh = x1 - x0, y1 - y0
        if bh < min_h * H:
            continue
        c = img.crop((int(max(0, x0 - .25 * bw)), int(max(0, y0 - .25 * bh)), int(min(W, x1 + .25 * bw)),
                      int(min(H, y1 + .25 * bh))))
        sc = 1024 / max(c.size)
        yield p, c.resize((round(c.width * sc), round(c.height * sc)), Image.LANCZOS)


def judge_person(a: dict, where: str) -> list[str]:
    flags = []
    hands = a.get("hands") or []
    if len(hands) > 2:
        flags.append(f"person crop: {len(hands)} hands on the person at {where}")
    try:
        arms = int(a.get("arms") or 0)
    except (TypeError, ValueError):
        arms = 0
    if arms > 2:
        flags.append(f"person crop: {arms} arms on the person at {where}")
    for h in hands:
        if isinstance(h, dict) and h.get("arm_connects_to_body") is False:
            flags.append(f"person crop: hand ({h.get('where', '?')}) not connected by an arm (person at {where})")
    return flags


def ask(img: Image.Image, url: str, ctx: str = "", question: str | None = None) -> dict:
    q = question or (ASK + (INTENT.format(ctx=ctx) if ctx else ""))
    body = json.dumps({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(img)}},
        {"type": "text", "text": q}]}], "temperature": 0.1, "max_tokens": 1200}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {"_unparsed": txt[:500]}


def judge(a: dict) -> list[str]:
    """Flags from the answers (rules, so each flag points at the answer behind it)."""
    flags = []
    for p in a.get("people") or []:
        if p.get("is_reflection_or_image"):
            continue
        hv, ha = int(p.get("hands_visible") or 0), int(p.get("hands_with_visible_arm") or 0)
        if hv > 2:
            flags.append(f"hands: {hv} hands on one person ({p.get('where', '?')})")
        if hv > ha:
            flags.append(f"hands: {hv - ha} hand(s) without a visible arm ({p.get('where', '?')})")
    if int(a.get("loose_hands") or 0) > 0:
        flags.append(f"hands: {a['loose_hands']} hand(s) belonging to nobody")
    if a.get("same_person_twice"):
        flags.append("duplicate: the same person twice")
    for v in a.get("vehicles") or []:
        t, on = str(v.get("type", "")).lower(), str(v.get("on", "")).lower()
        if t in ("train", "tram") and "rail" not in on and "track" not in on and "platform" not in on \
                and v.get("rails_visible_under_it") is False:
            flags.append(f"vehicle: {t} on {on or '?'} without rails")
        elif any(w in on for w in ("plaza", "pavement", "water", "air", "sidewalk", "square")):
            flags.append(f"vehicle: {t} on {on}")
    for r in a.get("reflections") or []:
        if r.get("geometry_ok") is False:
            flags.append(f"reflection: {r.get('of', '?')} at {r.get('where', '?')} doesn't fit the geometry")
    for o in a.get("impossible_objects") or []:
        flags.append(f"object: {o}")
    return flags


def crops(img: Image.Image):
    w, h = img.size
    for x0, x1 in ((0, int(w * 0.6)), (int(w * 0.4), w)):
        c = img.crop((x0, 0, x1, h))
        yield c.resize((int(c.width * 1.6), int(c.height * 1.6)), Image.LANCZOS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    ap.add_argument("--crops", action="store_true")
    ap.add_argument("--segment", metavar="DIR", help="segment_check.py output dir: add geometry flags + person crops")
    a = ap.parse_args()
    out = {}
    for arg in a.images:
        p, _, ctx = arg.partition("::")
        img = Image.open(p).convert("RGB")
        views = [img] + (list(crops(img)) if a.crops else [])
        t0 = time.time()
        answers = [ask(v, a.llm, ctx) for v in views]
        frame_flags = sorted({f for x in answers for f in judge(x)})
        rec = {"frame_flags": frame_flags, "answers": answers, "seconds": {"frame": round(time.time() - t0, 2)}}
        flags = list(frame_flags)
        if a.segment:
            seg = json.load(open(os.path.join(a.segment, os.path.splitext(os.path.basename(p))[0] + ".json")))
            geo = ["geometry: " + f for f in seg.get("flags", [])]
            t1, crop_ans, crop_flags = time.time(), [], []
            for person, c in person_crops(img, seg):
                where = "[" + ",".join(str(int(v)) for v in person["box"]) + "]"
                ans = ask(c, a.llm, question=PERSON)
                crop_ans.append({"box": person["box"], "answer": ans})
                crop_flags += judge_person(ans, where)
            rec.update(geometry_flags=geo, geometry_notes=seg.get("notes", []), crop_flags=crop_flags,
                       crop_answers=crop_ans)
            rec["seconds"].update(crops=round(time.time() - t1, 2), n_crops=len(crop_ans),
                                  segmentation=seg.get("seconds", {}).get("total"))
            flags += geo + crop_flags
        out[p] = {"ok": not flags, "flags": flags, **rec}
        print(f"{p}: {'OK' if not flags else 'FLAG'} " + "; ".join(flags), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
