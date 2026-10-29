"""poptok/lean step 3: the visualiser. music2vid's question, asked of each of the directors' beats (never of a raw
lyric): the best picture for this beat, in the context of this song and its story. It adds visual detail only (light,
textures, composition, what is in the frame), never changes who, what or where, and never names a medium or style:
the piece's STYLE is put first in every prompt, word for word. Where Amber is in the keyframe, the picture describes
her as PERFORMER does (her photo is matched later by the face pass).

    WORKSHOP_PY=lean/visualiser.py bin/workshop.sh PIECE       (starts the local LLM; reads PIECE/lean/storyboard.json)
Writes PIECE/lean/shots.json: the storyboard's keyframes, each with "picture" and the full "prompt". A keyframe whose
picture has her in it (PERFORMER's words) is hers, whatever the directors marked: amber "none" becomes "seen", so the
face pass makes her Amber (D42: cuckoo's last shot was a stranger). --only 9,11: rewrite just those keyframes' pictures
and keep the rest as they are in shots.json.
"""
from __future__ import annotations

import argparse, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from workshop import as_json, llm, read_env

PROMPT = """You are the visualiser of a music video. The directors have planned it as a storyboard of keyframes; each
keyframe is one picture that a video model animates. For each keyframe, write the best visual representation of its
BEAT in the context of this song and its story: one picture, 30-60 words, concrete and visual (what is in the frame,
where, light, textures, composition, how the camera sees it).

Rules:
- Keep the beat's who, what, where and camera exactly; you only add visual detail.
- Never name a medium, art style or colour palette (no "drawing", "charcoal", "photo", "cinematic", "black and white",
  colours): the style is added separately. No motion blur or camera moves: it is a still.
- No text or words in the picture.
- Where the woman is in the keyframe, describe her with these words exactly: "{performer}"; {look_clothes}
- Not the lyrics as words; the lyrics are context for meaning.
- Any named animal, plant or thing a picture model may not know well: say what it looks like (shape, size, colouring,
  markings), not only its name (a picture model drew "a cuckoo" as a crow).{looks}

THE BRIEF:
---
{brief}
---
THE STORY: {story}

THE LYRICS:
{lyrics}

THE KEYFRAMES:
{keyframes}

Return ONLY JSON: {{"pictures": ["keyframe 1's picture", "keyframe 2's picture", ...]}} (one per keyframe, in order)"""

EXPR = """A music video's keyframes show a young woman. For each one below, write her facial expression as a picture
editor must keep it: 8-20 words naming the feeling, where her eyes look, her brows and her mouth (D43: a face edit told
only "keep the expression" gave her a calm look). Fit the beat and the song; the brief: {brief}

{shots}

Return ONLY JSON: {{"expressions": {{"keyframe number": "her expression", ...}}}}"""

BANNED = r"\b(drawing|drawn|charcoal|sketch|photo\w*|cinematic|black and white|monochrome|colou?rs?|blur\w*|bokeh|" \
         r"zoom\w*|pans?|pulls? back|dolly)\b"


def check(pics, kf, performer, looks):
    p = []
    if len(pics) != len(kf):
        return [f"{len(pics)} pictures for {len(kf)} keyframes"]
    for i, (x, k) in enumerate(zip(pics, kf)):
        if m := re.search(BANNED, x, re.I):
            p.append(f"picture {i + 1} says \"{m.group(0)}\" (no medium, colour or camera move)")
        if (k["amber"] != "none" or re.search(r"\b(woman|girl|she|her|person|figure|someone)\b", x, re.I)) and performer.lower() not in x.lower():
            p.append(f"picture {i + 1} has the woman but not her description word for word (\"{performer}\")")
        for name, desc in looks.items():
            if re.search(rf"\b{re.escape(name)}s?\b", x, re.I) and desc.lower() not in x.lower():
                p.append(f"picture {i + 1} has the {name} but not its description word for word (\"{desc}\")")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    ap.add_argument("--only", help="keyframe numbers to rewrite, e.g. 9,11 (the rest kept from shots.json)")
    a = ap.parse_args()
    P = a.piece.rstrip("/"); e = read_env(f"{P}/piece.env")
    sb = json.load(open(f"{P}/lean/storyboard.json")); kf = sb["keyframes"]
    style = open(f"{P}/{e['STYLE']}").read().strip()
    performer = e.get("PERFORMER", "the singer")
    # inputs/looks.txt: "name: description" lines, what named things look like (D42: the LLM described a cuckoo as a
    # crow, so the facts come from the piece, not from its memory)
    lf = f"{P}/inputs/looks.txt"
    looks = dict(l.split(":", 1) for l in open(lf).read().splitlines() if ":" in l and not l.startswith("#")) \
        if os.path.exists(lf) else {}
    looks = {k.strip().lower(): v.strip() for k, v in looks.items()}
    # the directors' look, minus her hair (PERFORMER, from her photo, decides that): her clothes
    clothes = re.sub(r"\band (has|with) [^,.]*hair[^,.]*", "", sb.get("performer_look", "")).strip()
    keyframes = "\n".join(f"  {i + 1}. {k['start']:.1f}-{k['end']:.1f} s, the woman: {k['amber']}. BEAT: {k['beat']}"
                          for i, k in enumerate(kf))
    prompt = PROMPT.format(looks="".join(f'\n- Wherever the {k} is in a picture, describe it with these words exactly: "{v}"'
                                         for k, v in looks.items()), performer=performer, look_clothes=clothes, brief=open(f"{P}/inputs/brief.txt").read().strip(),
                           story=sb.get("story", ""), lyrics=open(f"{P}/{e['LYRICS']}").read().strip(),
                           keyframes=keyframes)
    pics, probs = [], ["no answer"]
    for attempt in range(2):
        pics = as_json(llm(prompt, a.llm, max_tokens=6000, temperature=0.7)).get("pictures") or []
        probs = check(pics, kf, performer, looks)
        print(f"visualiser: attempt {attempt + 1}: {len(pics)} pictures, {len(probs)} problems"
              + "".join(f"\n  - {x}" for x in probs), flush=True)
        if not probs:
            break
        prompt += ("\n\nYour previous answer:\n" + json.dumps({"pictures": pics}) + "\nIt has these problems; fix them "
                   "and return all the pictures again:\n" + "\n".join(f"- {x}" for x in probs))
    if a.only:
        old = json.load(open(f"{P}/lean/shots.json"))["shots"]; keep = {int(n) for n in a.only.split(",")}
        pics = [p if i + 1 in keep else o["picture"] for i, (p, o) in enumerate(zip(pics, old))]
    S = style[0].upper() + style[1:]
    shots = []
    for k, p in zip(kf, pics):
        x = dict(k, picture=p, prompt=f"{S}: {p}")
        if x["amber"] == "none" and performer.lower() in p.lower():
            x["amber"], x["amber_by"] = "seen", "visualiser: her description is in the picture"
        shots.append(x)
    # her expression where she is in a keyframe, kept from shots.json while the picture is unchanged
    old = {o["picture"]: o.get("expression") for o in json.load(open(f"{P}/lean/shots.json"))["shots"]} \
        if os.path.exists(f"{P}/lean/shots.json") else {}
    for x in shots:
        if x["amber"] != "none" and old.get(x["picture"]):
            x["expression"] = old[x["picture"]]
    need = [(i, x) for i, x in enumerate(shots) if x["amber"] != "none" and not x.get("expression")]
    if need:
        ex = as_json(llm(EXPR.format(brief=open(f"{P}/inputs/brief.txt").read().strip(), shots="\n".join(
            f"  {i + 1}. beat: {x['beat']} picture: {x['picture']}" for i, x in need)), a.llm, max_tokens=2000,
            temperature=0.5)).get("expressions") or {}
        for i, x in need:
            if ex.get(str(i + 1)):
                x["expression"] = ex[str(i + 1)].strip().rstrip(".")
        print("visualiser: her expressions:" + "".join(f"\n  {i + 1}: {x.get('expression')}" for i, x in need), flush=True)
    json.dump({"style": style, "checks": probs, "shots": shots}, open(f"{P}/lean/shots.json", "w"), indent=1,
              ensure_ascii=False)
    print(f"visualiser: wrote {P}/lean/shots.json")


if __name__ == "__main__":
    main()
