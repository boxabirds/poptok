"""Director's treatment for the whole video, then one scene prompt per lyric line written FROM that treatment.
Replaces gen_prompts_local.py's independent per-line prompts (no arc, no recurring cast, style as a weak tail suffix).

    python3 pipeline/treatment.py SONG LYRICS VISUAL_TREATMENT.txt --out OUT [--analysis DIR] [--llm http://127.0.0.1:8091]

Inputs: the master lyrics with their [Section] tags, the lyric lines on the song's clock (<song>-true-segments.json
from align_lyrics.py), loudness per section (from the wav), and the visual treatment text (any length; for a plain
style such as "gritty lo-fi black and white" the LLM expands it).
Step 1 - one LLM call writes OUT/treatment.json (also OUT/treatment.md to read and edit): logline, story arc, a small
fixed cast / places / motifs with exact wording, the look that leads every prompt, how things move and transition,
a text policy, the colour grade (start -> end) and one entry per section (palette stage, shot scale, story beat).
Step 2 - per lyric line, the LLM writes the scene from the treatment, reusing the cast/place wording verbatim.
Re-running step 2 only: delete ANALYSIS/full-metadata<tag>.json and keep treatment.json (edits are kept).
Writes ANALYSIS/full-metadata<tag>.json in the usual format (segments[*].prompt, fully composed: look first,
then the scene, then the section's palette; "composed": true tells keyframes.py not to append a style), plus
"treatment" (motion, transition, grade) for ltx_chain.py and beatcut.py.
"""
from __future__ import annotations

import argparse, json, os, re, sys, urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import analysis_dir

BIBLE = """You are the director of a music video. Write the treatment for the WHOLE video as one JSON object.

VISUAL TREATMENT FROM THE ARTIST (follow it closely; if it is short, expand it in its own spirit):
{visual}

SONG: {dur:.0f} seconds. Sections with their times and loudness (0 = quietest section, 1 = loudest):
{sections}

FULL LYRICS:
{lyrics}

Return JSON with exactly these keys:
- "logline": one sentence: what the video is about, start to finish.
- "story": 3-5 sentences: the visual story arc across the sections (what changes, what returns at the end).
- "cast": 1-3 recurring people, each {{"name": short id, "look": a fixed visual description of 12-25 words that every
  scene will reuse word for word (age, face, hair, clothes)}}. Use no people only if the treatment demands it.
- "places": 1-3 recurring places, each {{"name", "look": 12-25 words reused word for word}}.
- "motifs": 3-6 recurring objects or images taken from the lyrics, each {{"name", "look": under 15 words}}.
- "look": 25-45 words that lead EVERY image prompt: medium, materials, texture, lighting, grain. Each idea once, no
  repeated words. No colours that change over the song (those go in the sections).
- "motion": one sentence for the video model: how things move within a shot (camera and subject).
- "frame_step": 1 for smooth motion, 2 if the treatment asks for stop-motion / animated-photo jerkiness (frames held
  in pairs).
- "transition": one sentence: how one scene becomes the next. The shapes of one picture physically reshape into the
  next (a cable becomes a root, a window becomes a page). Never a cut, fade, dissolve, flash, burst of light, static,
  glitch or overlay.
- "text_policy": one sentence: whether any written words appear; at most one short word or number per scene, copied
  exactly from the lyrics, and only where it matters.
- "grade": {{"mode": "bw" or "color", "start": {{"saturation": 0-1, "warmth": -1 cool to 1 warm, "fade": 0-1 lifted
  faded blacks, "grain": 0-1}}, "end": same keys}} - the finishing colour treatment from the first to the last second.
- "sections": one entry per section above, in order, each {{"name", "palette": colour and light for this section (if
  the grade is "bw": light and shadow only, no colours; never flashes or bursts of light),
  "shot_scale": e.g. extreme close-up / close-up / medium / wide, "beat": what happens in this section's pictures}}.
Return only the JSON."""

SCENE = """You are writing one shot of a music video from the director's treatment below.

TREATMENT:
{bible}

SECTION: {section} - palette: {palette}; shot scale: {scale}; what happens: {beat}
PREVIOUS LINE: {prev}
THIS LYRIC LINE: {line}
NEXT LINE: {next}
SHOTS ALREADY WRITTEN FOR THIS SECTION (vary framing and subject; do not repeat them): {done}

Write the image for THIS LINE as JSON {{"prompt": ..., "text": ...}}:
- "prompt": 35-70 words, one concrete frame: who or what, where, doing what, framing and light. When a cast member,
  place or motif appears, copy its "look" words exactly. Show the line's meaning through the treatment's world; do not
  illustrate it literally if the treatment's story says otherwise. No colour or film-stock words (added separately).
- "text": "" unless the text policy allows a word here AND the line names something whose exact written word matters
  (a day, a number, a name); then that ONE word or number exactly as in the lyric, and the prompt shows it written on
  a physical surface.
Return only the JSON."""


def complete(llm, prompt, max_tokens, temperature=0.6):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                       "max_tokens": max_tokens, "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request(f"{llm}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])


def sections_of(lyrics_path, segs, dur, wav):
    """[(name, start, end, loudness 0-1, [line indices])] from the lyric file's [Section] tags and the aligned lines"""
    secs, cur, k = [], None, 0
    for raw in open(lyrics_path):
        l = raw.strip()
        if not l:
            continue
        m = re.match(r"^\[(.*)\]$", l)
        if m:
            cur = {"name": m.group(1).strip(), "lines": []}; secs.append(cur); continue
        if cur is None:
            cur = {"name": "Verse", "lines": []}; secs.append(cur)
        cur["lines"].append(k); k += 1
    secs = [s for s in secs if s["lines"]]
    for i, s in enumerate(secs):
        s["start"] = 0.0 if i == 0 else segs[s["lines"][0]]["start"]
        s["end"] = dur if i == len(secs) - 1 else segs[secs[i + 1]["lines"][0]]["start"]
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = x.mean(1)
    rms = [20 * np.log10(np.sqrt(np.mean(x[int(s["start"] * sr):max(int(s["end"] * sr), int(s["start"] * sr) + 1)] ** 2)) + 1e-9)
           for s in secs]
    lo, hi = min(rms), max(rms)
    for s, r in zip(secs, rms):
        s["loud"] = round(float((r - lo) / max(hi - lo, 1e-6)), 2)
    return secs


def dedupe(look, max_words=50):
    """LLMs sometimes loop ("gritty, raw, gritty, raw, ..."): keep each comma-separated phrase once, cap the length"""
    seen, out = set(), []
    for ph in (p.strip() for p in look.replace(".", ",").split(",")):
        if ph and ph.lower() not in seen:
            seen.add(ph.lower()); out.append(ph)
    kept, n = [], 0
    for ph in out:                                                   # whole phrases only, up to max_words
        if n and n + len(ph.split()) > max_words:
            break
        kept.append(ph); n += len(ph.split())
    return ", ".join(kept)


def to_md(b, secs):
    out = [f"# Treatment\n\n**Logline:** {b['logline']}\n\n**Story:** {b['story']}\n\n**Look (leads every prompt):** "
           f"{b['look']}\n\n**Motion:** {b['motion']}\n\n**Transition:** {b['transition']}\n\n**Text:** {b['text_policy']}\n"
           f"\n**Grade:** {json.dumps(b['grade'])}\n\n**Frame step:** {b.get('frame_step', 1)}\n"]
    for key in ("cast", "places", "motifs"):
        out.append(f"\n## {key.title()}\n" + "".join(f"- **{c['name']}**: {c['look']}\n" for c in b.get(key, [])))
    out.append("\n## Sections\n")
    for s, d in zip(secs, b["sections"]):
        out.append(f"- **{s['name']}** ({s['start']:.0f}-{s['end']:.0f}s, loudness {s['loud']}): {d.get('palette')}; "
                   f"{d.get('shot_scale')}; {d.get('beat')}\n")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song"); ap.add_argument("lyrics"); ap.add_argument("visual")
    ap.add_argument("--out", required=True)
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", default="-true")
    ap.add_argument("--bible-only", action="store_true")
    a = ap.parse_args()
    song = Path(a.song); ana = analysis_dir(song, a.analysis)
    segs = json.load(open(ana / f"{song.stem}{a.tag}-segments.json"))["segments"]
    dur = sf.info(str(song)).duration
    secs = sections_of(a.lyrics, segs, dur, str(song))
    lyrics = "\n".join((f"[{s['name']}]\n" + "\n".join(segs[i]["text"] for i in s["lines"])) for s in secs)
    bp = Path(a.out) / "treatment.json"; os.makedirs(a.out, exist_ok=True)
    if bp.exists():
        b = json.load(open(bp)); print("using existing", bp)
    else:
        sec_txt = "\n".join(f"- {s['name']}: {s['start']:.0f}-{s['end']:.0f}s, loudness {s['loud']}" for s in secs)
        for attempt in range(3):
            b = complete(a.llm, BIBLE.format(visual=open(a.visual).read().strip(), dur=dur, sections=sec_txt,
                                             lyrics=lyrics), 4000)
            if len(b.get("sections", [])) == len(secs) and all(k in b for k in ("look", "motion", "grade", "cast", "frame_step")):
                break
            print("treatment incomplete, retrying", file=sys.stderr)
        else:
            sys.exit("the LLM did not return a complete treatment")
        b["look"] = dedupe(b["look"])
        for s, d in zip(secs, b["sections"]):
            d.update({"name": s["name"], "start": round(s["start"], 2), "end": round(s["end"], 2), "loud": s["loud"]})
        json.dump(b, open(bp, "w"), indent=1)
    open(Path(a.out) / "treatment.md", "w").write(to_md(b, secs))
    print(open(Path(a.out) / "treatment.md").read(), flush=True)
    if a.bible_only:
        return

    world = {k: b[k] for k in ("logline", "story", "cast", "places", "motifs", "look", "text_policy")}
    rows = []
    for si, (s, d) in enumerate(zip(secs, b["sections"])):
        done = []
        for i in s["lines"]:
            sc = complete(a.llm, SCENE.format(bible=json.dumps(world, indent=1), section=s["name"],
                                              palette=d.get("palette", ""), scale=d.get("shot_scale", ""),
                                              beat=d.get("beat", ""), prev=segs[i - 1]["text"] if i else "(start)",
                                              line=segs[i]["text"], next=segs[i + 1]["text"] if i + 1 < len(segs) else "(end)",
                                              done=json.dumps(done) if done else "(none)"), 600)
            word = (sc.get("text") or "").strip()
            if word and word.lower() not in segs[i]["text"].lower():           # only words copied from the lyric line
                word = ""
            prompt = f"{b['look']}. {sc['prompt'].strip()} {d.get('palette', '')}."
            done.append(sc["prompt"][:90])
            rows.append({"start": segs[i]["start"], "end": segs[i]["end"], "text": segs[i]["text"], "prompt": prompt,
                         "section": s["name"], "word": word})
            print(f"{segs[i]['start']:7.2f} [{s['name']}] {segs[i]['text']!r}{' WORD=' + word if word else ''}\n"
                  f"         -> {sc['prompt']}", flush=True)
    opening = complete(a.llm, SCENE.format(bible=json.dumps(world, indent=1), section=secs[0]["name"] + " (instrumental "
                       "opening before the first sung line: establish the main place and person)",
                       palette=b["sections"][0].get("palette", ""), scale="wide", beat=b["sections"][0].get("beat", ""),
                       prev="(start)", line="(no words yet)", next=segs[0]["text"], done="(none)"), 600)
    meta = {"visual_description": f"{b['look']}. {opening['prompt'].strip()} {b['sections'][0].get('palette', '')}.",
            "style": "", "composed": True, "duration": str(dur),
            "treatment": {k: b[k] for k in ("look", "motion", "transition", "grade", "frame_step", "sections")},
            "full_transcript": {"segments": rows},
            "animation": {"keyframes_per_second": 3, "zoom_animation_times": {}}}
    json.dump(meta, open(ana / f"full-metadata{a.tag}.json", "w"), indent=2)
    print("wrote", ana / f"full-metadata{a.tag}.json")


if __name__ == "__main__":
    main()
