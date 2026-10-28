"""The directors pick the next R.Amber song for the procedural world (ice_world.py, the Ice Walk engine) and its look
(decision D5, 2026-10-28). The engine is an abstract walk through a made landscape that moves to the music, with no
performer; the LLM reads each candidate's lyrics and brief and chooses the song it suits best, and the settings,
within what the engine can do. A choice the user makes later overrides this one (directors-decide).

    python3 pipeline/world/pick_song.py OUT_DIR [--url http://127.0.0.1:8091]
Candidates: pieces/*/inputs with a song (.wav) and lyrics. Writes OUT_DIR/director.json and director.md.
"""
from __future__ import annotations

import argparse, glob, json, os, re, subprocess, urllib.request


def llm(prompt, temperature, url):      # the directors' LLM (llama-server, port 8091), as universe.py's llm(); not
    # imported from it: universe.py parses its own command line at import
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                       "max_tokens": 4096}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8091")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

songs = []
for d in sorted(glob.glob("pieces/*/inputs")):
    piece = d.split("/")[1]
    wav = ([w for w in glob.glob(f"{d}/*.wav") if os.path.basename(w) == "song.wav"] or sorted(glob.glob(f"{d}/*.wav")))
    lyr = [f for f in (f"{d}/lyrics.txt", f"{d}/{piece}-lyrics.txt", f"{d}/offline-lyrics.txt") if os.path.exists(f)]
    if not wav or not lyr or "-" in piece and piece not in ("new-parade", "noble-hunt", "slippery-road"):
        continue                        # variants and tests (chronophobia-oil, ...-test10) are not songs of their own
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav[0]],
                               capture_output=True, text=True).stdout)
    brief = open(f"{d}/brief.txt").read()[:1200] if os.path.exists(f"{d}/brief.txt") else ""
    songs.append({"piece": piece, "audio": wav[0], "seconds": round(dur, 1), "lyrics": open(lyr[0]).read()[:2500], "brief": brief})
assert songs, "no candidate songs"

ENGINE = """The engine renders one continuous walk, at eye height, through a made landscape at dusk: fields of crystal
formations (ice, polished chrome, or a mix of both), a low sun in the sky, drifting snow, a low mist, and a camera that
surges forward on the song's low swells. The music's separate sounds each drive their own visual (e.g. high chirps
send swifts tearing across the sky, chord changes sweep cloud shadows over the ground, clicks glint as cracks). There is
no performer, no story and no text: it is a mood piece. Settings:
  theme: "ice" | "chrome" | "icechrome"
  sun_el: degrees above the horizon, 3 (dusk, long shadows) .. 30 (bright)
  sky: three numbers 0..1.5 (air, dust, ozone; more dust = warmer, hazier sky)
  sun_color, mist_color: RGB 0..1 each
  snow: 0 (none) .. 30000 (blizzard); mist: 0 .. 0.05 (density)"""

PROMPT = f"""You are the director of a music video series for the singer R.Amber. Choose which ONE of these songs should
be made next with the engine below, and how it should look. Choose the song whose music and words this wordless,
moving landscape would serve best (not the one with the most literal imagery), and settings that fit its mood.

{ENGINE}

Songs:
""" + "\n".join(f"--- {s['piece']} ({s['seconds']} s)\nBrief: {s['brief'] or '(none)'}\nLyrics:\n{s['lyrics']}\n" for s in songs) + """
Return ONLY JSON: {"piece": "<one of the names above>", "why": "<2-3 sentences>", "world": "<one sentence: the place>",
"theme": "...", "sun_el": N, "sky": [a, b, c], "sun_color": [r, g, b], "mist_color": [r, g, b], "snow": N, "mist": N}"""

names = {s["piece"] for s in songs}
for _ in range(3):
    m = re.search(r"\{.*\}", llm(PROMPT, 0.7, a.url), re.S)
    try:
        d = json.loads(m.group(0))
        if d["piece"] in names and d["theme"] in ("ice", "chrome", "icechrome"):
            break
    except (AttributeError, json.JSONDecodeError, KeyError, TypeError):
        pass
    print("  unusable answer, again", flush=True)
else:
    raise SystemExit("pick_song: no usable answer from the LLM")
cl = lambda v, lo, hi: max(lo, min(hi, float(v)))
d["sun_el"] = cl(d["sun_el"], 3, 30); d["snow"] = int(cl(d["snow"], 0, 30000)); d["mist"] = cl(d["mist"], 0, 0.05)
d["sky"] = [cl(v, 0, 1.5) for v in d["sky"]][:3]
d["sun_color"] = [cl(v, 0, 1) for v in d["sun_color"]][:3]; d["mist_color"] = [cl(v, 0, 1) for v in d["mist_color"]][:3]
d["audio"] = next(s["audio"] for s in songs if s["piece"] == d["piece"])
d["seconds"] = next(s["seconds"] for s in songs if s["piece"] == d["piece"])
d["candidates"] = sorted(names)
json.dump(d, open(f"{a.out}/director.json", "w"), indent=1)
open(f"{a.out}/director.md", "w").write(
    f"# Next procedural piece: {d['piece']}\n\n{d['why']}\n\nWorld: {d['world']}\n\nSettings: theme {d['theme']}, sun "
    f"{d['sun_el']:.0f} deg, sky {d['sky']}, sun colour {d['sun_color']}, mist {d['mist']} {d['mist_color']}, snow "
    f"{d['snow']}\n\nCandidates: {', '.join(d['candidates'])}. Override: name another song.\n")
print("pick ok", d["piece"], "|", d["why"])
