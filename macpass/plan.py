"""poptok/macpass step 1: macvideo's plan (POC 29, ~/sambashare/expts/macvideo/pocs/29-full-song), with poptok's local
LLM in place of Gemini. Tried after poptok/lean's cuckoo v2 (2026-10-29: the user rated it 1/10: no continuity, a
spectacle at every cut, 1024x576). macvideo keeps a world together by chaining: one world brief for every shot, a
storyboard whose shots link to the previous and next, and each keyframe drawn from the ones before it (keys.py).
  - shots: one per lyric line, instrumental gaps their own shots (macvideo's make_shots.build_shots, imported, at 24 fps);
  - Pass A: the world brief; Pass C: the storyboard; Pass B: one image prompt per shot (the prompts are macvideo's,
    copied: gen_keyframes.py cannot be imported without Gemini). poptok adds the artist's brief and the performer, and
    keeps the medium out of the world (cuckoo's charcoal brief said "white paper background", and lean drew the paper),
    and the narrator's pose out of the brief (cuckoo's brief gave her "arms extended upward in a fixed, reaching
    gesture", so every shot had her arms up and LTX moved them like a robot's, 2026-10-29).
Writes PIECE/macpass/{shots,brief,storyboard,image_prompts}.json; each is kept if it exists (delete it to redo).
    WORKSHOP_PY=macpass/plan.py bin/workshop.sh PIECE        (starts the local LLM)"""
import json, os, re, sys, urllib.request

sys.path.insert(0, os.path.expanduser("~/sambashare/expts/macvideo/pocs/29-full-song/scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
import make_shots
from workshop import read_env

make_shots.FPS = 24                       # the first cut is 24 fps (the artist's brief)
URL = "http://127.0.0.1:8091"
ABSTRACTION = 25                          # macvideo's default (pipeline.sh)
ABSTRACTION_DESCRIPTOR = ("loosely expressive — brushwork and line quality given primacy over accuracy; subjects still "
                          "clearly legible but simplified; distortion and gesture honoured")
CAMERA_INTENTS = ["static hold", "slow push in", "slow pull back", "pan left", "pan right", "tilt up", "tilt down",
                  "orbit left", "orbit right", "handheld drift", "held on detail"]

PASS_A = """You are a music video director. Read the song carefully.

Song lyrics:
---
{lyrics}
---

The artist's brief (what the video must say and show; it overrides your own ideas):
---
{brief}
---

The narrator is {performer}.

The video will be rendered in the "{filter_word}" style.
Abstraction level: {abstraction}/100. Apply throughout: {abstraction_descriptor}

Write a short "character & world brief" (5-8 sentences) that every shot of
this song must honour. Describe:
 - The narrator: appearance and clothing only (concrete visual cues). No pose,
   posture or gesture: each shot's beat decides how she stands and moves
 - The central metaphorical subject of this song as a visible entity
 - The primary setting / world (domestic? exterior? atmospheric qualities)
 - How the "{filter_word}" style manifests on all of the above (materials,
   textures, line quality, palette, lighting)

Use concrete visual language only — no emotional labels. Describe what IS
present, never what is absent.

The style is how every picture is made, never a thing inside the world: never
mention paper, canvas, a page, a sketchbook or any surface it is drawn on.

Return ONLY the brief, as a single paragraph. No preamble, no quotes."""

PASS_C = """You are a music video director AND a storyboard artist.

World brief (applies to every shot):
---
{brief}
---

The artist's brief (must be honoured, above all its ending):
---
{artist}
---

You are given {n} shots to storyboard as a SEQUENCE for a full music video.
Each shot's target text is below in order; some are lyric lines, some are
instrumental fillers ([instrumental intro], [instrumental bridge], etc.).

CRITICAL:
  - Plan the sequence so adjacent shots progress narratively.
  - Instrumental fillers should evoke the song's atmosphere, NOT narrate.
  - Camera direction should not reverse on adjacent shots unless punctuated.
  - Use ONLY these exact values for camera_intent:
    {camera_intents}
  - Never let two adjacent shots produce identical beats, even if identical
    target text appears (establish → develop → culminate progression).

For each shot, produce:
  - index              (1-based)
  - target_text        (echo input)
  - beat               (one concrete sentence: the narrative moment this shot captures)
  - camera_intent      (from the allowed vocabulary)
  - subject_focus      (what the frame centres on — a concrete element)
  - prev_link          (one sentence connecting back; null for shot 1)
  - next_link          (one sentence setting up next; null for last shot)

Also produce "sequence_arc" — one sentence describing the full video arc.

Target shots (in order):
{targets_block}

Return a SINGLE JSON object:
{{
  "sequence_arc": "...",
  "shots": [
    {{"index": 1, "target_text": "...", "beat": "...", "camera_intent": "...",
      "subject_focus": "...", "prev_link": null, "next_link": "..."}},
    ...
  ]
}}

Return ONLY the JSON, no preamble, no code fences."""

PASS_B = """You are a music video director generating ONE image prompt for a keyframe.

World brief:
---
{brief}
---

Abstraction level: {abstraction}/100. Apply as visual instruction: {abstraction_descriptor}

Storyboard for this specific shot:
  - target text:       "{target_text}"
  - narrative beat:    {beat}
  - subject focus:     {subject_focus}
  - camera intent:     {camera_intent}
  - connects from prior shot: {prev_link}
  - sets up next shot:        {next_link}

Write ONE image-generation prompt (3-4 sentences) for the keyframe. The image must:
  - Depict the beat (not just the literal surface of the target line)
  - Centre on the subject_focus
  - Honour the world brief (same character, same setting, same filter)
  - Describe only what IS present in the frame — materials, objects, atmosphere, motion
  - Use concrete visual cues only — no emotional labels, no negations
  - Never mention paper, canvas, a page or any surface the picture is drawn on

Return ONLY the final image prompt as a single paragraph. No preamble, no quotes."""


def llm(prompt, max_tokens=1200):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0.6,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def main():
    P = sys.argv[1].rstrip("/"); e = read_env(f"{P}/piece.env"); O = f"{P}/macpass"; os.makedirs(O, exist_ok=True)
    have = lambda n: os.path.exists(f"{O}/{n}.json")
    put = lambda n, x: json.dump(x, open(f"{O}/{n}.json", "w"), indent=1, ensure_ascii=False)
    get = lambda n: json.load(open(f"{O}/{n}.json"))
    lyrics = open(f"{P}/{e['LYRICS']}").read(); artist = open(f"{P}/inputs/brief.txt").read()
    filt = e.get("MAC_FILTER") or open(f"{P}/{e['STYLE']}").read().strip()
    if not have("shots"):
        # the lyric lines as timed by poptok's alignment (macvideo uses WhisperX the same way)
        segs = json.load(open([f"{P}/analysis/{f}" for f in os.listdir(f"{P}/analysis") if f.endswith("-true-segments.json")][0]))
        spans = [{"start_s": s["start"], "end_s": s["end"], "text": s["text"], "line_idx": i}
                 for i, s in enumerate(segs["segments"])]
        dur = json.load(open(f"{P}/analysis/metadata.json")).get("duration") or float(os.popen(
            f"ffprobe -v error -show_entries format=duration -of csv=p=0 '{P}/{e['SONG']}'").read())
        put("shots", {"fps": make_shots.FPS, "duration_s": dur, "shots": make_shots.build_shots(spans, dur)})
    shots = get("shots")["shots"]; print(f"plan: {len(shots)} shots", flush=True)
    if not have("brief"):
        put("brief", {"brief": llm(PASS_A.format(lyrics=lyrics, brief=artist, performer=e["PERFORMER"], filter_word=filt,
                                                 abstraction=ABSTRACTION, abstraction_descriptor=ABSTRACTION_DESCRIPTOR)),
                      "filter": filt})
    brief = get("brief")["brief"]; print(f"Pass A: {brief}", flush=True)
    if not have("storyboard"):
        targets = "\n".join(f"  Shot {s['index']}: ({s['kind']}) {s['target_text']!r}" for s in shots)
        raw = llm(PASS_C.format(brief=brief, artist=artist, n=len(shots), camera_intents=", ".join(CAMERA_INTENTS),
                                targets_block=targets), max_tokens=8000)
        raw = re.sub(r"^```(json)?|```$", "", raw.strip()).strip()
        put("storyboard", json.loads(raw[raw.find("{"):raw.rfind("}") + 1]))
    sb = {s["index"]: s for s in get("storyboard")["shots"]}
    print(f"Pass C: {get('storyboard').get('sequence_arc')}", flush=True)
    ip = get("image_prompts") if have("image_prompts") else {}
    for s in shots:
        k = f"shot_{s['index']:03d}"
        if k in ip:
            continue
        b = sb[s["index"]]
        ip[k] = llm(PASS_B.format(brief=brief, abstraction=ABSTRACTION, abstraction_descriptor=ABSTRACTION_DESCRIPTOR,
                                  target_text=s["target_text"], beat=b.get("beat", ""),
                                  subject_focus=b.get("subject_focus", ""),
                                  camera_intent=b.get("camera_intent", "static hold"),
                                  prev_link=b.get("prev_link") or "null", next_link=b.get("next_link") or "null")
                    ).strip('"').strip()
        put("image_prompts", ip)
        print(f"Pass B {s['index']}: {ip[k]}", flush=True)
    print(f"plan: wrote {O}", flush=True)


if __name__ == "__main__":
    main()
