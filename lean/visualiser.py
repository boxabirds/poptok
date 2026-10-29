"""poptok/lean step 3: the visualiser. music2vid's question, asked of each of the directors' beats (never of a raw
lyric): the best picture for this beat, in the context of this song and its story. It adds visual detail only (light,
textures, composition, what is in the frame), never changes who, what or where, and never names a medium or style:
the piece's STYLE is put first in every prompt, word for word. Where Amber is in the keyframe, the picture describes
her as PERFORMER does (her photo is matched later by the face pass).

    WORKSHOP_PY=lean/visualiser.py bin/workshop.sh PIECE       (starts the local LLM; reads PIECE/lean/storyboard.json)
Writes PIECE/lean/shots.json: the storyboard's keyframes, each with "picture" and the full "prompt".
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

BANNED = r"\b(drawing|drawn|charcoal|sketch|photo\w*|cinematic|black and white|monochrome|colou?rs?|motion blur|" \
         r"zoom\w*|pans?|pulls? back|dolly)\b"


def check(pics, kf, performer):
    p = []
    if len(pics) != len(kf):
        return [f"{len(pics)} pictures for {len(kf)} keyframes"]
    for i, (x, k) in enumerate(zip(pics, kf)):
        if m := re.search(BANNED, x, re.I):
            p.append(f"picture {i + 1} says \"{m.group(0)}\" (no medium, colour or camera move)")
        if k["amber"] != "none" and performer.lower() not in x.lower():
            p.append(f"picture {i + 1} has the woman but not her description word for word (\"{performer}\")")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    a = ap.parse_args()
    P = a.piece.rstrip("/"); e = read_env(f"{P}/piece.env")
    sb = json.load(open(f"{P}/lean/storyboard.json")); kf = sb["keyframes"]
    style = open(f"{P}/{e['STYLE']}").read().strip()
    performer = e.get("PERFORMER", "the singer")
    # the directors' look, minus her hair (PERFORMER, from her photo, decides that): her clothes
    clothes = re.sub(r"\band (has|with) [^,.]*hair[^,.]*", "", sb.get("performer_look", "")).strip()
    keyframes = "\n".join(f"  {i + 1}. {k['start']:.1f}-{k['end']:.1f} s, the woman: {k['amber']}. BEAT: {k['beat']}"
                          for i, k in enumerate(kf))
    prompt = PROMPT.format(performer=performer, look_clothes=clothes, brief=open(f"{P}/inputs/brief.txt").read().strip(),
                           story=sb.get("story", ""), lyrics=open(f"{P}/{e['LYRICS']}").read().strip(),
                           keyframes=keyframes)
    pics, probs = [], ["no answer"]
    for attempt in range(2):
        pics = as_json(llm(prompt, a.llm, max_tokens=6000, temperature=0.7)).get("pictures") or []
        probs = check(pics, kf, performer)
        print(f"visualiser: attempt {attempt + 1}: {len(pics)} pictures, {len(probs)} problems"
              + "".join(f"\n  - {x}" for x in probs), flush=True)
        if not probs:
            break
        prompt += ("\n\nYour previous answer:\n" + json.dumps({"pictures": pics}) + "\nIt has these problems; fix them "
                   "and return all the pictures again:\n" + "\n".join(f"- {x}" for x in probs))
    S = style[0].upper() + style[1:]
    shots = [dict(k, picture=p, prompt=f"{S}: {p}") for k, p in zip(kf, pics)]
    json.dump({"style": style, "checks": probs, "shots": shots}, open(f"{P}/lean/shots.json", "w"), indent=1,
              ensure_ascii=False)
    print(f"visualiser: wrote {P}/lean/shots.json")


if __name__ == "__main__":
    main()
