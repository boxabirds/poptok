"""Merge consecutive lyric prompts that describe the same scene, so a scene change only happens when the picture
should really change. Asks the local LLM (music2vid's prompt model) about each consecutive pair of distinct prompts.

    python3 pipeline/merge_scenes.py OUT [--llm http://127.0.0.1:11435] [--model gemma4-31b]

Writes OUT/shots_merged.json (same format as shots.json; a merged bar takes the previous scene's prompt).
"""
from __future__ import annotations

import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts_per_line
from prompts_per_line import complete

Q = ("Two image prompts for consecutive shots of a music video follow. Answer REPEAT only if the main subject AND the "
     "setting are essentially the same, so the second shot would look like a re-take of the first (e.g. a command center "
     "in panic, then a command center in chaos). If the main subject, setting or framing changes (a person vs a room, a "
     "gear vs a server room, a city vs a vault, a close-up vs a wide shot), answer NEW. Answer with one word.\n\n"
     "A: {a}\n\nB: {b}\n")
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--llm", default="http://127.0.0.1:11435")
ap.add_argument("--model", default="gemma4-31b")
ap.add_argument("--api", default="ollama", choices=["ollama", "openai"])
ap.add_argument("--plan", default="shots.json")
ap.add_argument("--max-bars", type=int, default=3, help="never merge a scene beyond this many bars")
a = ap.parse_args()
prompts_per_line.API = a.api
plan = json.load(open(os.path.join(a.out, a.plan)))
look = plan.get("treatment", {}).get("look", "")
strip = lambda p: p[len(look) + 2:] if look and p.startswith(look) else p   # compare scenes, not the shared look
cur, merged = None, 0
for s in plan["shots"]:
    if cur is not None and s["prompt"] != cur["orig"]:
        ans = complete(a.llm, a.model, Q.format(a=strip(cur["prompt"]), b=strip(s["prompt"])), temperature=0, max_tokens=8).upper()
        same = "REPEAT" in ans and cur["bars"] < a.max_bars
        print(f"bar {s['i']:2d}: {'merge' if same else 'NEW  '} {ans[:8]:8s} {s['prompt'][:80]}", flush=True)
        if same:
            merged += 1
            cur["orig"] = s["prompt"]; s["prompt"] = cur["prompt"]; cur["bars"] += 1
            s.pop("word", None); s.update({"word": cur["word"]} if cur["word"] else {}); continue
    if cur is not None and s["prompt"] == cur["orig"]:
        s["prompt"] = cur["prompt"]; cur["bars"] += 1
        s.pop("word", None); s.update({"word": cur["word"]} if cur["word"] else {}); continue
    cur = {"prompt": s["prompt"], "orig": s["prompt"], "bars": 1, "word": s.get("word", "")}
json.dump(plan, open(os.path.join(a.out, "shots_merged.json"), "w"), indent=1)
n = 1 + sum(plan["shots"][i]["prompt"] != plan["shots"][i - 1]["prompt"] for i in range(1, len(plan["shots"])))
print(f"merged {merged} near-duplicate changes -> {n} scenes")
