"""poptok/lean step 4b: the storyboard as poptok's LTX chain input (pipeline/ltx_chain.py renders it), PIECE/lean/render/:
  keys/shot_NNN.png   the keyframes (lean/keyframes.py; the user's picks);
  shots.json          one shot per keyframe: its span and its picture (the still's words), and the style;
  scenes.json         per shot: motion = the directors' beat (it says how the camera and people move), and
                      transition_in = how the previous keyframe becomes this one, written by the LLM with poptok's
                      TRANSITION prompt (pipeline/scenes.py): a physical EVENT in action words, since a quiet A->B
                      cross-fades with or without the Transition LoRA (POC 27). Kept while both pictures are unchanged.

    WORKSHOP_PY=lean/render_prep.py bin/workshop.sh PIECE        (starts the local LLM)
"""
from __future__ import annotations

import argparse, json, os, shutil, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from scenes import TRANSITION, llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    a = ap.parse_args()
    P = a.piece.rstrip("/"); L = f"{P}/lean"; O = f"{L}/render"; os.makedirs(f"{O}/keys", exist_ok=True)
    sh = json.load(open(f"{L}/shots.json")); sb = json.load(open(f"{L}/storyboard.json"))
    segs_f = [f for f in os.listdir(f"{P}/analysis") if f.endswith("-true-segments.json")][0]
    segs = json.load(open(f"{P}/analysis/{segs_f}"))["segments"]
    shots = []
    for j, s in enumerate(sh["shots"]):
        shutil.copy(f"{L}/keyframes/{j + 1:02d}.png", f"{O}/keys/shot_{j:03d}.png")
        # the style first (LTX, like MOSS, follows the start of a prompt: with the style at the end, cuckoo's ink
        # keyframe became a colour photograph of a forest within 2 s, 2026-10-29); ltx_chain appends the plan's "style"
        # and as a kind of video, not a description: "2D hand-drawn animation ... not live action" kept cuckoo a drawing
        # for the whole clip, where the style words alone turned photographic by 2.5 s (probe ltx_style.py, 2026-10-29)
        anim = f"A 2D hand-drawn animation in this style: {sh['style']}; a drawing that moves, drawn frames, not live action."
        shots.append({"i": j, "start": s["start"], "end": s["end"], "prompt": f"{anim} {s['picture']}", "amber": s["amber"],
                      "lyric": " / ".join(segs[n - 1]["text"] for n in s.get("lines", []) if 0 < n <= len(segs))})
    guard = ("The whole video stays a hand-drawn drawing in that style, never a photograph; drawn on paper that stays as "
             "light as it began, the light even, never darker")
    json.dump({"style": guard, "shots": shots}, open(f"{O}/shots.json", "w"), indent=1, ensure_ascii=False)
    old = {(x.get("a"), x.get("b")): x["transition_in"] for x in json.load(open(f"{O}/scenes.json"))["scenes"]} \
        if os.path.exists(f"{O}/scenes.json") else {}
    story = f"The story of the video: {sb.get('story', '')}\n\n"
    scenes = []
    for j, s in enumerate(shots):
        x = {"id": f"k{j + 1:02d}", "bars": [j, j], "motion": sh["shots"][j]["beat"], "continuous": False,
             "a": sh["shots"][j - 1]["picture"] if j else None, "b": sh["shots"][j]["picture"], "transition_in": ""}
        if j:
            x["transition_in"] = old.get((x["a"], x["b"])) or llm(TRANSITION.format(
                story=story, lyric=s["lyric"] or "(instrumental)", a=x["a"], b=x["b"]), a.llm)
            print(f"into {j + 1}: {x['transition_in']}", flush=True)
        scenes.append(x)
    json.dump({"scenes": scenes}, open(f"{O}/scenes.json", "w"), indent=1, ensure_ascii=False)
    print(f"render_prep: wrote {O}/keys, shots.json, scenes.json")


if __name__ == "__main__":
    main()
