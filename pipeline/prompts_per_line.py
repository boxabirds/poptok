"""Per-line prompts (the C5/C6 method, superseded by treatment.py): music2vid prompt generation (generate_styles.py) with a LOCAL LLM instead of the retired OpenAI Completions API.

    python3 pipeline/prompts_per_line.py SONG.wav --style "gritty lo-fi black and white, grainy, shadowy" \
        --llm http://127.0.0.1:11435 --model gemma4-31b

Same instructions and post-processing as music2vid/generate_styles.py (generate_overall_song_prompt,
generate_visual_prompts: keep only the last line of each reply). The general style is FIXED by --style rather than
chosen from music2vid's list. Lines = Whisper segments from poc.py (ANALYSIS/<song>-medium-segments.json).
Writes ANALYSIS/<song>-visual-prompts.json (music2vid's format) and full-metadata.json (run_batches.py format:
visual_description, style, estd_bpm, duration, full_transcript.segments[*].prompt, animation from generate_animations).
"""
from __future__ import annotations

import argparse, json, urllib.request
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import analysis_dir

SONG_PROMPT = (
    "To create a uniquely visual illustrative style for the song lyrics below, "
    "please come up with a text prompt of no more than 25 words that can be used for AI image generation tools "
    "that best matches the mood and sentiment of the song lyrics below."
    "Exclude any specific actions like 'text prompt' just provide the visual imagery without quotes.\n\n"
)
LINE_PROMPT = """
        "Below is a single line from the lyrics of a song, followed by the full lyrics.  We want to create a supporting visual for this single line.
        Using all lyrics as context for meaning, generate a corresponding vivid visual prompt that
        can be used to communicate the meaning of the single line visually. Do not prefix with any text like 'visual prompt' or 'vivid visual prompt'.
         \n\n"
        """


API = "ollama"


def complete(llm, model, prompt, temperature=0.7, max_tokens=1024):
    if API == "openai":                                       # llama-server / vLLM / SGLang
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "temperature": temperature, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(f"{llm}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
    req = urllib.request.Request(f"{llm}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"].strip()


def last_line(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return (lines[-1] if lines else "").strip('"').strip()


DIRECTIONS = {
    "none": "the singer does not appear in this shot",
    "pov": "seen through the singer's eyes: she is not visible; people may look toward or away from the camera as if at her",
    "part": "show only part of the singer (a hand, her back, her silhouette or a reflection), never her face",
    "scene": "the singer is in the shot ({performer}); call her 'the woman'",
    "sing": "the singer is in the shot, singing this line ({performer}); call her 'the woman'",
}


def direction(cue, performer):
    mode, note = cue
    return (f"\n(Direction for this shot: {DIRECTIONS[mode].format(performer=performer)}."
            + (f" Staging: {note}." if note else "") + ")\n")


def cues_for(lines, path):
    """Each lyric line's (mode, note) from an annotated lyrics file, matched in order by its words; a section's
    [Singer: ...] direction is added to the notes of lines in that section that show her."""
    import re
    words = lambda t: re.findall(r"[a-z0-9']+", t.lower().replace("’", "'"))
    cued, mode, sec_note, user_dir, pending = [], "none", "", "", ""
    for raw in open(path).read().split("\n"):
        t = raw.strip()
        m = re.match(r"^\[([^\]]*)\]\s*\{amber:\s*(\w+)(?::\s*([^}]*))?\}", t)
        if m:                                   # a section: the user's direction just above it applies to it only
            mode, sec_note = m.group(2), (m.group(3) or "").strip()
            user_dir, pending = pending, ""
            continue
        if re.match(r"^\[.*\]$", t):
            inner = t[1:-1]
            if ":" in inner and inner.split(":")[0].strip().lower() in ("singer", "amber", "performer"):
                pending = inner.split(":", 1)[1].strip()
            continue
        if not t:
            continue
        c = re.search(r"\{(\w+)(?::\s*([^}]*))?\}\s*$", t)
        lm, ln = (c.group(1), (c.group(2) or "").strip()) if c else (mode, sec_note)
        if user_dir and lm in ("scene", "sing", "part"):
            ln = f"{user_dir}; {ln}" if ln else user_dir
        cued.append((words(re.sub(r"\{[^}]*\}", "", t)), (lm, ln)))
    out, j = [], 0
    for _, _, text in lines:
        w = words(text)
        k = next((i for i in range(j, len(cued)) if cued[i][0] == w), None)
        if k is None:
            out.append(None)
            continue
        out.append(cued[k][1])
        j = k + 1
    missing = sum(c is None for c in out)
    if missing:
        print(f"cues: {missing} of {len(lines)} lines had no cue (left as before)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--style", required=True)
    ap.add_argument("--llm", default="http://127.0.0.1:11435")
    ap.add_argument("--model", default="gemma4-31b")
    ap.add_argument("--api", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--segments", default=None, help="lyric lines with timings, e.g. <song>-true-segments.json from "
                    "align_lyrics.py (master lyrics on Whisper's clock); default: Whisper's own <song>-medium-segments.json")
    ap.add_argument("--no-text", action="store_true", help="ask for scenes without written words / signs / on-screen "
                    "text (video models garble lettering)")
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--lyrics-context", default=None, help="full lyrics file used as context (default: the lines being "
                    "prompted); a short test excerpt still sees the whole song's lyrics")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. -true")
    ap.add_argument("--cues", default=None, help="lyrics annotated with performer cues (cast_cues.py): each line's visual "
                    "is written for its cue (none / pov / part / scene / sing) and the cue is kept for the keyframes")
    ap.add_argument("--style-word", default="", help="a few-word style (POC 24), e.g. 'stained glass': leads every prompt")
    ap.add_argument("--style-brief", default="", help="the style brief written from --style-word (style_bench.py BRIEF): "
                    "what everything is made of and how it's lit; every line's visual is written inside it")
    ap.add_argument("--performer", default="a young woman with long wavy blond hair",
                    help="how the performer looks, for lines where she appears")
    a = ap.parse_args()
    global API
    API = a.api
    song = Path(a.song)
    ana = analysis_dir(song, a.analysis)
    segs = json.load(open(a.segments or ana / f"{song.stem}-medium-segments.json"))["segments"]
    lines = [(float(s["start"]), float(s["end"]), s["text"].strip()) for s in segs if s["text"].strip()]
    full_lyrics = "\n".join(t for _, _, t in lines) + "\n"
    if a.lyrics_context:
        import re
        full_lyrics = "\n".join(l for l in (re.sub(r"\{[^}]*\}", "", x).strip() for x in open(a.lyrics_context))
                                if l and not re.match(r"^\[.*\]$", l)) + "\n"

    cues = cues_for(lines, a.cues) if a.cues else [None] * len(lines)
    song_prompt = last_line(complete(a.llm, a.model, SONG_PROMPT + full_lyrics, max_tokens=256))
    if a.cues:              # the intro (before the first sung line) uses this prompt: with a cast, it holds no people,
        song_prompt += ", no people"            # or MOSS invents a stranger who isn't the performer (POC 22)
    print("song prompt:", song_prompt, flush=True)
    visual = []
    prev = None                     # the previous line and the visual written for it: continuity between shots
    for (st, en, text), cue in zip(lines, cues):
        extra = (f"\n(The whole video is rendered in the {a.style_word} style: {a.style_brief} Describe this shot entirely "
                 f"within that style; the line still decides what is in it.)\n" if a.style_word else "") + \
                (direction(cue, a.performer) if cue else "") + ("\n(The visual must contain no written words, letters, numbers, logos, signs or on-screen text; show "
                 "the meaning through people, places, objects and light instead.)\n" if a.no_text else "")
        ctx = (f"\n(For continuity: the previous line was \"{prev[0]}\" and its visual was: {prev[1]})\n" if prev else "")
        v = last_line(complete(a.llm, a.model, LINE_PROMPT + text + ctx + extra + full_lyrics))
        if a.style_word:
            v = f"{a.style_word}: {v}"
        prev = (text, v)
        print(f"{st:7.2f}  {text!r}\n         -> {v}", flush=True)
        visual.append((st, en, text, v, cue))

    with open(ana / f"{song.stem}-visual-prompts{a.tag}.json", "w") as f:            # music2vid's (python-ish) format
        f.write(f'"style": "{a.style}",\n0: "{song_prompt}",\n')
        for st, _, _, v, _ in visual:
            f.write(f'{st:.2f}: "{v}",\n')

    meta_path = ana / f"full-metadata{a.tag}.json"
    base = ana / "full-metadata.json"
    meta = json.load(open(meta_path if meta_path.exists() else base)) if (meta_path.exists() or base.exists()) else {}
    tempo = json.load(open(ana / "metadata.json")).get("tempo") if (ana / "metadata.json").exists() else None
    import subprocess
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(song)],
                               capture_output=True, text=True).stdout)
    meta.update({
        "visual_description": song_prompt, "style": a.style, "estd_bpm": str(tempo), "duration": str(dur),
        "full_transcript": {"segments": [{"start": st, "end": en, "text": t, "prompt": v,
                                          **({"cast": c[0], "cast_note": c[1]} if c else {})}
                                         for st, en, t, v, c in visual]},
    })
    meta.setdefault("animation", {"keyframes_per_second": 3, "zoom_animation_times": {}})
    json.dump(meta, open(meta_path, "w"), indent=2)
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
