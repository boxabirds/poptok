"""Songwriters' workshop: from a prompt ("a song about unrequited love") to lyrics and a music style, by simulating a
lyric-writing workshop between three of the world's best songwriters with the local LLM. The first step of a piece
made from a prompt alone: then ACE-Step sings it (bin/make_song.sh), the directors' workshop plans the video
(pipeline/workshop.py), and the video is made as usual.

    python3 pipeline/songwriters.py PIECE [--llm URL] [--force]
Reads PIECE/piece.env's SONG_PROMPT (the prompt itself, or a file under the piece holding it) and SONG_FORM (optional:
the structure, default FORM below). Writes, in PIECE/inputs/:
  songwriters.md     the discussion transcript and the final lyrics (for the user to read)
  lyrics.txt         the final lyrics, one line per sung line, [Section] tags (LYRICS)
  music-style.txt    genre, tempo (BPM), key/mood, instruments, vocals, production, under 500 characters (MUSIC_STYLE)
Two LLM passes: (1) the conversation (with repetition penalties; a conversation that falls into a loop, the same line
over and over, is retried once and otherwise cut where the loop starts); (2) the songwriters write the final song from
it as JSON (lyrics and music style), checked against the form (intro, 4 different verses, choruses between them, an
outro whose second chorus changes) and retried with the problem named if it doesn't fit. First test (2026-10-25,
"a song about unrequited love"): the conversation looped on one line for half its length and never reached its
final lyrics; the extraction then repeated verse 1 as verse 4 and made the outro two identical choruses.

The songwriters are fictional composites standing for traditions, not real people (as the directors are).
"""
from __future__ import annotations

import argparse, json, os, re, sys, time, urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workshop import as_json, read_env

FORM = ("an intro, 4 verses interleaved with the chorus, ending with an outro that is a hybrid of 2 choruses back to "
        "back")

SONGWRITERS = """- June Calloway: the Nashville craft tradition (the title as the hook, one concrete image per line, the twist
  in the last chorus, every word earning its place).
- Rafa Mendel: the pop-melody tradition (syllables that sing, open vowels on long notes, a chorus you know by the second
  time, rhythm before meaning in the hook).
- Noor Achterberg: the singer-songwriter tradition (the confessional detail nobody else would write, plain speech,
  restraint, the line that turns the song)."""

CONVERSATION = """Simulate a lyric-writing workshop between three of the world's best songwriters, fictional composites
who each stand for a tradition:
{songwriters}

They are writing a song from this prompt: "{prompt}"

The form: {form}.
The song will be sung by an AI singer (ACE-Step) and then turned into a music video, so:
- lines a singer can phrase: mostly 5-10 syllables, natural stress, no tongue-twisters, no names or words that are
  hard to hear when sung; a verse is 4-6 lines, the chorus 4-6 lines and the same words each time (small changes in the
  last ones are fine)
- concrete images a camera could film (places, objects, weather, light, gestures) carry the feeling, rather than
  abstract statements
- the intro is instrumental, or one or two short lines at most

Write the workshop as a conversation (names in bold, then what they say), about 1500-2000 words. They should: start
from what the prompt really means and whose voice the song is in (the angle that hasn't been written a thousand
times); agree the title and the hook; agree the music (genre, tempo, instruments, the kind of voice) because the words
must fit it; draft the chorus first, then the verses, each verse moving the story on (not repeating the first); argue,
cut cliches, and improve each other's lines; then build the outro from two choruses back to back, the second changed
so the song lands somewhere new. Keep it moving: once a line is agreed, move on (never restate the same change). End with the complete final lyrics under
the heading FINAL LYRICS, with [Section] tags ([Intro], [Verse 1], [Chorus], ..., [Outro]), and then a one-paragraph
MUSIC line describing the music."""

RESULT = """You are the three songwriters of this workshop, finishing the song you were writing. Take everything you
agreed (the angle, the title, the hook, the chorus, the best lines); where the discussion stopped short, finish it in
the same spirit.

The workshop:
{conversation}

The form it must follow: {form}.
Rules: every verse is new and moves the story on (no verse repeats another, none reuses the chorus lines); the chorus
keeps the same words each time; the outro is two choruses back to back, the second CHANGED (at least two of its lines
rewritten) so the song lands somewhere new; lines a singer can phrase (mostly 5-10 syllables), concrete images.{problem}

Return JSON only:
{{"title": "the song title",
 "music_style": "for an AI music generator, UNDER 480 characters, plain comma-separated phrases: genre, tempo as 'NN BPM', key or mood, instruments, drums and bass, vocal type and delivery, production. No section directions, no lyrics",
 "sections": [{{"tag": "Intro" | "Verse 1" | "Verse 2" | ... | "Chorus" | "Outro", "lines": ["sung line", ...]}}]}}
The sections in song order. An instrumental intro has "lines": []. Lines are only sung words: no stage directions, no
parentheses, no section names."""


def llm(prompt: str, url: str, max_tokens: int, temperature: float, **extra) -> str:
    """workshop.llm, plus sampler settings (repetition penalties) for the long conversation."""
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                       "max_tokens": max_tokens, **extra}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def loop_start(text: str) -> int:
    """The character offset where the text starts repeating one line (a line of 20+ characters seen 4+ times), or -1."""
    lines = [" ".join(x.split()).lower() for x in text.split("\n")]
    c = Counter(x for x in lines if len(x) >= 20)
    rep = {x for x, n in c.items() if n >= 4}
    if not rep:
        return -1
    pos = 0
    for raw, x in zip(text.split("\n"), lines):
        if x in rep:
            return pos
        pos += len(raw) + 1
    return -1


def check(res: dict, form: str) -> str:
    """'' if the sections fit the form; otherwise what's wrong. Only the default form is checked in detail."""
    secs = res.get("sections") or []
    if not secs or not all(isinstance(s, dict) and s.get("tag") for s in secs):
        return "no sections"
    if not str(res.get("music_style", "")).strip():
        return "no music style"
    if len(str(res["music_style"])) >= 500:
        return f"music style is {len(res['music_style'])} characters"
    if form != FORM:
        return ""
    tags = [str(s["tag"]).strip().lower() for s in secs]
    verses = [t for t in tags if t.startswith("verse")]
    if not tags[0].startswith("intro") or not tags[-1].startswith("outro"):
        return f"must start with an intro and end with an outro: {tags}"
    if len(verses) != 4:
        return f"{len(verses)} verses, not 4: {tags}"
    body = tags[1:-1]
    for a, b in zip(body, body[1:]):
        if a.startswith("verse") and b.startswith("verse"):
            return f"two verses without a chorus between them: {tags}"
    if sum(t.startswith("chorus") for t in body) < 3:
        return f"fewer than 3 choruses between the verses: {tags}"
    norm = lambda ls: [" ".join(clean(x).lower().split()) for x in ls or []]
    vs = [norm(s.get("lines")) for s in secs if str(s["tag"]).lower().startswith("verse")]
    for i in range(len(vs)):
        for j in range(i):
            if len(set(vs[i]) & set(vs[j])) * 2 > max(1, len(vs[i])):
                return f"verse {i + 1} repeats verse {j + 1}"
    ch = next((norm(s.get("lines")) for s in secs if str(s["tag"]).lower().startswith("chorus")), [])
    out = norm(secs[-1].get("lines"))
    if len(out) < 2 * max(2, len(ch) - 1):
        return "the outro is not two choruses long"
    second = out[len(out) // 2:]
    if len(set(second) - set(ch)) < 2:
        return "the outro's second chorus is not changed (rewrite at least two of its lines)"
    return ""


def clean(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\[\]{}]|\([^)]*\)", "", str(line))).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--llm", default="http://127.0.0.1:8091")
    ap.add_argument("--force", action="store_true", help="redo (the existing transcript is kept as songwriters.prev-*.md)")
    a = ap.parse_args()
    P = a.piece.rstrip("/")
    e = read_env(os.path.join(P, "piece.env"))
    prompt = e.get("SONG_PROMPT", "").strip()
    if not prompt:
        sys.exit("songwriters: set SONG_PROMPT in piece.env (the prompt, or a file under the piece holding it)")
    if os.path.isfile(os.path.join(P, prompt)):
        prompt = open(os.path.join(P, prompt)).read().strip()
    form = e.get("SONG_FORM", "").strip() or FORM
    inp = os.path.join(P, "inputs")
    os.makedirs(inp, exist_ok=True)
    out_md, out_lyr, out_style = (os.path.join(inp, n) for n in ("songwriters.md", "lyrics.txt", "music-style.txt"))
    if os.path.exists(out_md):
        if not a.force:
            print(f"{out_md} exists (use --force to redo)")
            return
        os.rename(out_md, os.path.join(inp, f"songwriters.prev-{time.strftime('%Y%m%d-%H%M%S')}.md"))
    print("songwriters: conversation", flush=True)
    for attempt in range(2):
        conv = llm(CONVERSATION.format(songwriters=SONGWRITERS, prompt=prompt, form=form), a.llm, max_tokens=5000,
                   temperature=0.8, repeat_penalty=1.1, frequency_penalty=0.3, presence_penalty=0.2)
        cut = loop_start(conv)
        if cut < 0:
            break
        print(f"  the conversation loops from character {cut} of {len(conv)}" + ("; retry" if attempt == 0 else
                                                                                 "; cut there"), flush=True)
    if cut >= 0:
        conv = conv[:cut].rstrip() + "\n\n[The transcript is cut here: the conversation began repeating itself.]"
    print("songwriters: final lyrics and music style", flush=True)
    res, why = {}, "not run"
    for attempt in range(3):
        try:
            problem = "" if why == "not run" else f"\nA previous draft was rejected: {why}. Fix that."
            res = as_json(llm(RESULT.format(conversation=conv, form=form, problem=problem), a.llm, max_tokens=3000,
                              temperature=0.4))
            why = check(res, form)
        except (ValueError, KeyError, TypeError) as ex:
            why = f"unparsable ({ex})"
        if not why:
            break
        print(f"  {why}; retry", flush=True)
    if why:
        sys.exit(f"songwriters: no usable result ({why}); the conversation is in the LLM log, re-run to try again")
    rows = []
    for s in res["sections"]:
        rows.append(("" if not rows else "\n") + f"[{clean(s['tag']) or 'Verse'}]")
        rows += [c for c in (clean(x) for x in s.get("lines") or []) if c]
    lyrics = "\n".join(rows) + "\n"
    style = " ".join(str(res["music_style"]).split())
    open(out_lyr, "w").write(lyrics)
    open(out_style, "w").write(style + "\n")
    md = [f"# Songwriters' workshop: {res.get('title', os.path.basename(P))}", "",
          "Simulated by `pipeline/songwriters.py` with the local LLM. The songwriters are fictional composites standing "
          "for traditions, not real people.", "", "**Songwriters**", SONGWRITERS, "", f"**Prompt:** {prompt}  ",
          f"**Form:** {form}", "", "## The conversation", "", conv, "", "## The final song", "",
          f"**Title:** {res.get('title', '')}  ", f"**Music style** (`inputs/music-style.txt`): {style}", "",
          "**Lyrics** (`inputs/lyrics.txt`)", "", "```", lyrics.rstrip(), "```"]
    open(out_md, "w").write("\n".join(md) + "\n")
    n = sum(1 for r in rows if r and not r.lstrip().startswith("["))
    print(f"songwriters: wrote {out_md}, {out_lyr} ({len(res['sections'])} sections, {n} lines), {out_style}")


if __name__ == "__main__":
    main()
