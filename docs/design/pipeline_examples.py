"""Real example assets for tasks 3-8 of the pipeline page (docs/design/20261107-pipeline.html), for the article.
Each figure is one PNG at most 760 px wide, laid out for a phone held upright: wide sheets are restacked, labels are
large (20 px and up at 760 px, so ~10 px+ on a phone), every panel says what it is and when.

    python3 docs/design/pipeline_examples.py            (from poptok/, on gruntus: needs ffmpeg, numpy, PIL, node, chrome)

Source: Noble Hunt v6 (pieces/noble-hunt), the one piece run end to end through the main pipeline with a sung insert:
analysis/ (stems, Whisper words, beats, drum runs), inputs/ (story, storyboard, universes), outputs/v6/ (stills,
scenes, video frames, lip-sync clip, the final video). Writes:
    docs/design/20261107-pipeline-examples.html          the figures on one page (also readable by itself)
    docs/design/20261107-pipeline-examples/src/          the frames and crops the page shows
    docs/design/20261107-pipeline-examples/NN-*.png      one image per figure (via page_cards.mjs --select figure.ex)
    docs/design/20261107-pipeline-examples/3x-*.svg      the earlier explainers (beat detection, Demucs, onsets)
"""
import html, json, os, shutil, subprocess, sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
P = "pieces/noble-hunt"
V = f"{P}/outputs/v6"
OUT = "docs/design/20261107-pipeline-examples"
SRC = f"{OUT}/src"
PAGE = "docs/design/20261107-pipeline-examples.html"
os.makedirs(SRC, exist_ok=True)
W = 712                                     # content width inside a 760 px figure (24 px padding each side)
esc = html.escape
J = lambda f: json.load(open(f))


def mmss(t):                                # 1:05.2
    t = round(t, 1); return f"{int(t // 60)}:{t % 60:04.1f}"


def mss(t):                                 # 1:05
    t = round(t); return f"{t // 60}:{t % 60:02d}"


def save(im, name, width=W):
    im = im.convert("RGB")
    if im.width != width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(f"{SRC}/{name}", quality=88)
    return f"20261107-pipeline-examples/src/{name}"


def frame(video, t, name, width=W):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", f"{SRC}/_f.png"], check=True)
    return save(Image.open(f"{SRC}/_f.png"), name, width)


def audio(path, t0, t1, sr=8000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0}", "-t", f"{t1 - t0}", "-i", path, "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def wave(y, h, colour, scale=None, W=W):
    """Min/max envelope per pixel column as one SVG path (width W, height h, centred)."""
    cols = np.array_split(y, W)
    lo = np.array([c.min() if len(c) else 0 for c in cols]); hi = np.array([c.max() if len(c) else 0 for c in cols])
    s = scale or max(1e-6, np.abs(y).max())
    top = " ".join(f"L{x},{h / 2 - hi[x] / s * h / 2 * .95:.1f}" for x in range(W))
    bot = " ".join(f"L{x},{h / 2 - lo[x] / s * h / 2 * .95:.1f}" for x in range(W - 1, -1, -1))
    return f'<path d="M0,{h / 2} {top} {bot} Z" fill="{colour}"/>'


def axis(t0, t1, h, step=1.0):
    xs = []
    t = np.ceil(t0 / step) * step
    while t <= t1 + 1e-6:
        x = (t - t0) / (t1 - t0) * W
        anc = "start" if x < 30 else "end" if x > W - 30 else "middle"
        xs.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="6" stroke="#5b6475"/>'
                  f'<text x="{x:.1f}" y="26" font-size="19" text-anchor="{anc}" fill="#5b6475">{mss(t) if step >= 1 else mmss(t)}</text>')
        t += step
    return f'<svg width="{W}" height="{h}" class="axis">{"".join(xs)}</svg>'


figs = []                                    # (id, task label, title, body html, caption)


def fig(fid, task, title, body, cap):
    figs.append(f'<figure class="ex" id="{fid}"><div class="task">{esc(task)}</div><h2>{esc(title)}</h2>{body}'
                f'<p class="cap">{cap}</p><p class="src">Noble Hunt · {esc(fid)}</p></figure>')


# ---------------------------------------------------------------- task 3: disassemble the song
T3 = "Task 3 · Disassemble the song"
t0, t1 = 58.0, 66.0
rows = [("Song", f"{P}/inputs/song.wav", "#1d2433"), ("Vocals", f"{P}/analysis/song-vocals.mp3", "#2f7d9e"),
        ("Drums", f"{P}/analysis/song-drums.mp3", "#b8642e"), ("Bass", f"{P}/analysis/song-bass.mp3", "#6b4fa0"),
        ("Other (guitars, keys…)", f"{P}/analysis/song-other.mp3", "#2f9e6b")]
ys = [audio(f, t0, t1) for _, f, _ in rows]
peak = max(np.abs(y).max() for y in ys)
body = "".join(f'<div class="lab">{esc(n)}</div><svg width="{W}" height="78">{wave(y, 78, c, peak)}</svg>'
               for (n, _, c), y in zip(rows, ys)) + axis(t0, t1, 32)
fig("3a-stems", T3, "The song split into its parts", body,
    "The first chorus coming in (0:58–1:06), split by Demucs into four stems, all on one loudness scale. "
    "The vocal stem is what the speech recogniser listens to; the drum stem gives the beats and drum hits.")

segs = J(f"{P}/analysis/song-true-whisper-segments.json")["segments"]
words = [w for s in J(f"{P}/analysis/song-medium.json")["segments"] for w in s.get("words", [])]
body = ""
for k in (0, 2):
    s = segs[k]; a, b = s["start"] - 0.5, s["end"] + 0.4
    y = audio(f"{P}/analysis/song-vocals.mp3", a, b)
    X = lambda t: (t - a) / (b - a) * W
    ws = [w for w in words if w["start"] >= s["start"] - 0.4 and w["end"] <= s["end"] + 0.1]
    marks = []
    for j, w in enumerate(ws):
        x0, x1 = X(w["start"]), X(w["end"]); yy = 110 + (j % 2) * 32
        marks.append(f'<rect x="{x0:.1f}" y="{yy - 21}" width="{max(2, x1 - x0 - 2):.1f}" height="28" rx="3" fill="#e4eef3"/>'
                     f'<text x="{x0 + 3:.1f}" y="{yy}" font-size="20" fill="#1d4f66">{esc(w["text"])}</text>')
    xs, xe = X(s["start"]), X(s["end"])
    body += (f'<div class="lab">Written lyric line, placed: <b>{esc(s["text"])}</b> ({mmss(s["start"])}–{mmss(s["end"])})</div>'
             f'<svg width="{W}" height="176"><g transform="translate(0,4)">{wave(y, 76, "#9fbcc9")}</g>'
             f'<line x1="{xs:.1f}" y1="0" x2="{xs:.1f}" y2="176" stroke="#c0392b" stroke-width="2.5"/>'
             f'<line x1="{xe:.1f}" y1="0" x2="{xe:.1f}" y2="176" stroke="#c0392b" stroke-width="2.5" stroke-dasharray="6 4"/>'
             + "".join(marks) + "</svg>" + axis(a, b, 32, 0.5))
fig("3b-lyrics-timed", T3, "The written lyrics, on Whisper's timings", body,
    "Blue boxes: the words Whisper heard in the vocal stem, with their times (it heard “While the rain keeps time”). "
    "Red lines: where the written lyric line (from the lyric sheet) is placed, start (solid) and end (dashed), by matching its words to Whisper's. The video uses the written words, never Whisper's.")

b = J(f"{P}/analysis/beats.json"); dh = J(f"{P}/analysis/drum_hits.json")
a, e = 52.4, 58.4
X = lambda t: (t - a) / (e - a) * W
y = audio(f"{P}/analysis/song-drums.mp3", a, e)
g = [f'<g transform="translate(0,40)">{wave(y, 120, "#e3c2a8")}</g>']
for r in dh["runs"]:
    if r["end"] > a and r["start"] < e:
        g.append(f'<rect x="{X(r["start"]) - 4:.1f}" y="36" width="{X(r["end"]) - X(r["start"]) + 8:.1f}" height="128" fill="#f6d36b" opacity=".35"/>'
                 f'<text x="{X(r["start"]):.1f}" y="186" font-size="19" fill="#8a6d00">run of {r["n"]}</text>')
for t in b["beats"]:
    if a <= t <= e:
        down = any(abs(t - d) < 0.03 for d in b["downbeats"])
        g.append(f'<line x1="{X(t):.1f}" y1="{6 if down else 18}" x2="{X(t):.1f}" y2="166" stroke="#1d2433" '
                 f'stroke-width="{3 if down else 1.2}" opacity="{1 if down else .55}"/>'
                 + (f'<text x="{X(t) + 4:.1f}" y="22" font-size="19" font-weight="700" fill="#1d2433">bar</text>' if down else ""))
for h in dh["hits"]:
    if a <= h["t"] <= e:
        g.append(f'<circle cx="{X(h["t"]):.1f}" cy="100" r="{4 + 8 * h["strength"]:.1f}" fill="#b8642e"/>')
body = (f'<div class="lab">Drum stem, {mmss(a)}–{mmss(e)}</div><svg width="{W}" height="196">{"".join(g)}</svg>'
        + axis(a, e, 32) +
        '<div class="key"><span><i style="background:#1d2433;width:4px"></i>bar line (downbeat)</span>'
        '<span><i style="background:#1d2433;opacity:.55;width:2px"></i>beat</span>'
        '<span><i style="background:#b8642e;border-radius:50%"></i>deep drum hit</span>'
        '<span><i style="background:#f6d36b"></i>drum run</span></div>')
fig("3c-beats-drum-runs", T3, "Beats, bars and drum runs", body,
    f"The beat tracker found {float(b['tempo']):.0f} beats a minute; every shot starts on a bar line. Quick runs of "
    "deep drum hits (yellow) are where light bursts can go, and the directors are told where they are.")

for f, n in (("beat-detection.svg", "3d-beat-detection.svg"), ("demucs.svg", "3e-demucs.svg"), ("onset.svg", "3f-onsets.svg")):
    shutil.copy(f"pieces/_bench/review/2026-11-10/{f}", f"{OUT}/{n}")

# ---------------------------------------------------------------- task 4: design the story and the world it's in
T4 = "Task 4 · Design the story and the world it’s in"
env = dict(l.split("=", 1) for l in open(f"{P}/piece.env").read().splitlines() if "=" in l and not l.startswith("#"))
val = lambda k: env[k].split("  #")[0].strip().strip('"')
av = save(Image.open("pieces/artists/r.amber.jpeg"), "4a-avatar.jpg", 300)
fig("4a-singer", T4, "The singer: one still and two lines", f'<div class="side"><img src="{av}" width="300">'
    f'<div><div class="lab">Avatar still</div><p class="big">R.Amber</p><div class="lab">Her look, one line</div>'
    f'<p class="big">“{esc(val("PERFORMER"))}”</p><div class="lab">What she wears in this piece</div>'
    f'<p>“{esc(val("WARDROBE"))}”</p></div></div>',
    "Everything she looks like downstream comes from these: the still is given to the image editor for every shot "
    "she is in; the two lines go into every picture description.")

st = J(f"{P}/inputs/story-scenes.json")
rows = "".join(f'<tr><td><b>{esc(s["id"])}</b><br>{mss(s["start"])}</td><td><b>{esc(s["movement"])}</b> · '
               f'{esc(s["place"])}<br><span class="dim">{esc(s["action"].split(". ")[0])}.</span></td></tr>' for s in st["scenes"])
fig("4b-story", T4, "The directors' story", f'<p class="logline">{esc(st["logline"])}</p><table class="tl">{rows}</table>',
    "The directors' treatment, cut into its movements against the song (start times shown). The full treatment is "
    "prose; each movement here is its place and first action.")

u = J(f"{P}/inputs/universes.json"); faces = J(f"{P}/inputs/universes/faces.json")
picks = [p["name"] for p in u["picks"]]
chips = "".join(f'<span class="chip{" pick" if c["name"] in picks else ""}">{esc(c["name"])}'
                f'{" · directors’ own" if c["name"] == "Hand-Tinted Noir" else ""}</span>' for c in u["candidates"])
fig("4c-universes-proposed", T4, "15 universes proposed, 3 picked", f'<div class="chips">{chips}</div>',
    "What the world is made of, proposed by the language model: open ideas, ideas from the song's central image, and "
    "the directors' own. The producer picked the three outlined; each is drawn in two test shots next.")

body = ""
for k, p in enumerate(u["picks"], 1):
    ims = "".join(f'<div><img src="{save(Image.open(f"{P}/inputs/universes/pick{k}_{s}.png"), f"4d-pick{k}-{s}.jpg", 352)}" width="352">'
                  f'<div class="lab">{"opening" if s == "opening" else "the turn"} · face {faces[f"pick{k}_{s}"]:.2f}</div></div>'
                  for s in ("opening", "turn"))
    body += (f'<div class="pk"><p class="big"><b>{k}. {esc(p["name"])}</b>{"  ✓ chosen" if k == 1 else ""}</p>'
             f'<p class="dim">{esc(p["made_of"].split(". ")[0])}.</p><div class="two">{ims}</div></div>')
fig("4d-universe-sheet", T4, "Each pick drawn in the same two shots", body,
    "Two test shots with no material named in them, drawn in each picked universe, with her face put in from the "
    "avatar still. Face: how like her photo (0–1); 0.06 means the face pass failed. Noble Hunt v6 itself was drawn "
    "before this step, in the directors' own look (Hand-Tinted Noir), kept photographic as asked.")

# ---------------------------------------------------------------- task 5: design the video scenes
T5 = "Task 5 · Design the video scenes"
shots = J(f"{V}/shots.json")["shots"]
cnt = {s["id"]: sum(1 for x in shots if s["start"] - .05 <= x["start"] < s["end"] - .05) for s in st["scenes"]}
rows = "".join(f'<tr><td><b>{esc(s["id"])}</b><br>{mss(s["start"])}–{mss(s["end"])}</td>'
               f'<td>{esc(s["place"])}<br><span class="dim">{cnt[s["id"]]} bars</span></td></tr>' for s in st["scenes"])
fig("5a-storyboard", T5, "Storyboard: the story as scenes", f'<table class="tl">{rows}</table>',
    f"The directors turn the treatment into {len(st['scenes'])} scenes, each one place, and give every lyric line and "
    f"instrumental gap a shot; split at the bar lines that is {len(shots)} stills for a 4-minute song.")

seg7 = next(s for s in segs if s["text"].startswith("Free is all we want"))
sh = shots[20]
prompt = sh["prompt"]
cut = prompt[:prompt.find(" ", 420)] + " …"
img = save(Image.open(f"{V}/keys/shot_020.png"), "5b-still.jpg")
fig("5b-line-to-picture", T5, "One line, from lyric to picture",
    f'<div class="lab">Lyric line · {mmss(seg7["start"])}</div><p class="big">“{esc(seg7["text"])}”</p>'
    f'<div class="lab">The directors\' shot</div><p class="big">{esc(sh["cast_note"].split(": ", 1)[1])}</p>'
    f'<div class="lab">Picture description (first 420 characters of {len(prompt)})</div><p class="mono">{esc(cut)}</p>'
    f'<div class="lab">The still drawn from it (task 6)</div><img src="{img}" width="{W}">',
    "The picture description starts with the look, then the shot, then her wardrobe, written with the whole lyric "
    "and the previous line in view so consecutive shots stay in one world.")

# ---------------------------------------------------------------- task 6: key frames
T6 = "Task 6 · Create key frames"
sc = J(f"{V}/scenes.json")["scenes"]
s12 = sc[12]
ims = "".join(f'<div><img src="{save(Image.open(f"{V}/keys/shot_{i:03d}.png"), f"6a-bar{i}.jpg", 352)}" width="352">'
              f'<div class="lab">bar {k + 1} · {mmss(shots[i]["start"])}</div></div>' for k, i in enumerate(range(30, 34)))
fig("6a-one-still-per-bar", T6, "One shot, one still per bar", f'<p class="big">{esc(s12["cast_note"].split(": ", 1)[1])}</p>'
    f'<div class="two">{ims}</div>',
    f"A 7.6 s instrumental shot, 4 bars long: four stills, one per bar, so the picture can change on the bar line. "
    f"Neighbouring bars of one shot are then merged into video segments: {len(shots)} stills became {len(sc)} segments.")

kinds = [(20, "Her whole figure"), (44, "Only part of her (hand, back)"), (59, "Singing")]
ims = "".join(f'<div><img src="{save(Image.open(f"{V}/keys/shot_{i:03d}.png"), f"6b-{i}.jpg", 352)}" width="352">'
              f'<div class="lab">{esc(n)} · {mmss(shots[i]["start"])}</div></div>' for i, n in kinds)
av = Image.open("pieces/artists/r.amber.jpeg"); av2 = save(av.crop((0, 60, av.width, 60 + round(av.width * 9 / 16))), "6b-avatar.jpg", 352)
fig("6b-drawn-from-avatar", T6, "Drawn from the avatar still",
    f'<div class="two"><div><img src="{av2}" width="352"><div class="lab">The avatar still (input)</div></div>{ims}</div>',
    "Every still she is in is drawn by the image editor from the avatar still and the shot's words, in her wardrobe "
    "for the piece; there are three kinds of shot, depending on how much of her is seen.")

# ---------------------------------------------------------------- task 7: create the scenes
T7 = "Task 7 · Create the scenes"
ev = sc[19]
fr = [2885, 2905, 2918, 2926, 2934, 2960]
ims = "".join(f'<div><img src="{save(Image.open(f"{V}/ltx_chain/{n:05d}.png"), f"7a-{n}.jpg", 352)}" width="352">'
              f'<div class="lab">{k + 1} · {mmss(n / 24)}</div></div>' for k, n in enumerate(fr))
fig("7a-transition-event", T7, "A change of scene, as an event",
    f'<div class="lab">The transition, written by the language model</div><p class="mono">{esc(ev["transition_in"])}</p>'
    f'<div class="two">{ims}</div>',
    f"From the corridor (story scene S3) to the plaza (S4) at {mmss(ev['start'])}: LTX makes the dust storm the "
    "transition describes and comes out of it in the next scene, in one continuous take. Video frames, in order.")

mv = sc[9]                                   # a clean move; several other scenes drift into doorways (see the caption's note)
a_, b_ = mv["frames"]; fr = [a_ + 5 + (b_ - a_ - 10) * k // 3 for k in range(4)]
ims = "".join(f'<div><img src="{save(Image.open(f"{V}/ltx_chain/{n:05d}.png"), f"7b-{n}.jpg", 352)}" width="352">'
              f'<div class="lab">{k + 1} · {mmss(n / 24)}</div></div>' for k, n in enumerate(fr))
fig("7b-camera-move", T7, "Inside a scene, a camera move",
    f'<div class="lab">The move, written by the language model</div><p class="mono">{esc(mv["transition_in"].split(". ")[1].split(", ")[0])}.</p>'
    f'<div class="two">{ims}</div>',
    f"Within one story scene the change of still is written as a camera move, not an event, so the shot stays one "
    f"place: here {mv['end'] - mv['start']:.1f} s across {mv['bars'][1] - mv['bars'][0] + 1} bars. Not always: in several "
    "other Noble Hunt moves the street drifts into a doorway on the way to the next still.")

ins = J(f"{V}/sing/inserts.json")[0]
sung = [s["text"] for s in segs if s["end"] > ins["start"] + .3 and s["start"] < ins["end"] - .3]
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            f"{V}/sing/it_out_0.mp4"], capture_output=True, text=True).stdout)
ts = [0.5, 1.4, 2.3, 3.2, 4.1, 5.0]
FACE = (370, 0, 530, 160)                    # her head in the 896x448 clip (she stands still; checked by eye)


def face(t, name):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", f"{V}/sing/it_out_0.mp4", "-frames:v", "1",
                    f"{SRC}/_f.png"], check=True)
    return save(Image.open(f"{SRC}/_f.png").crop(FACE), name, 232)


ims = (f'<div class="lab">Its keyframe, drawn from the avatar still</div>'
       f'<img src="{save(Image.open(f"{V}/sing/insert_0.png"), "7c-key.jpg")}" width="{W}">'
       '<div class="lab">The clip, close on her face</div><div class="three">'
       + "".join(f'<div><img src="{face(t, f"7c-{k}.jpg")}" width="232"><div class="lab">{k + 1} · {mmss(ins["start"] + t)}</div></div>'
                 for k, t in enumerate(ts)) + "</div>")
fig("7c-lip-sync", T7, "The singing clip",
    f'<div class="lab">Sung here · {mmss(ins["start"])}–{mmss(ins["end"])}</div>'
    + "".join(f'<p class="big">“{esc(l)}”</p>' for l in sung) + ims,
    f"InfiniteTalk animates the keyframe from the vocal stem for exactly this stretch ({dur:.1f} s): mouth, head and "
    "breathing follow the voice. It is spliced into the video in task 8.")

# ---------------------------------------------------------------- task 8: assemble the video
T8 = "Task 8 · Assemble the video"
final = f"{V}/noble-hunt-v6.mp4"
times = [(s["start"] + s["end"]) / 2 for s in st["scenes"]]
ims = "".join(f'<div><img src="{frame(final, t, f"8a-{k}.jpg", 352)}" width="352"><div class="lab">{s["id"]} · {mss(t)}'
              f'</div></div>' for k, (t, s) in enumerate(zip(times, st["scenes"])))
fig("8a-the-video", T8, "The finished video, scene by scene", f'<div class="two">{ims}</div>',
    "One frame from the middle of each story scene of the finished 4-minute video: the arc from hiding indoors to "
    "standing in the open at dawn. Frames retimed to the beat (each bar played at 0.83–1.14× speed, median 1.01×), "
    "colour graded, deflared, song added.")

pts = [(ins["start"] - 1.0, "before"), (ins["start"] + 2.5, "singing clip"), (ins["end"] + 1.0, "after")]
ims = "".join(f'<div><img src="{frame(final, t, f"8b-{k}.jpg", 352)}" width="352"><div class="lab">{n} · {mmss(t)}</div></div>'
              for k, (t, n) in enumerate(pts))
fig("8b-splice", T8, "The singing clip, spliced in", f'<div class="two">{ims}</div>',
    f"The singing clip replaces the video from {mmss(ins['start'])} to {mmss(ins['end'])}: its sung line plus a "
    "quarter-second either side. It goes in after the grade, so it is not graded.")

# ---------------------------------------------------------------- overview: one row per task, with a real illustration
import re
pg = open("docs/design/20261107-pipeline.html").read()
TT = {int(n): (re.sub("<[^>]+>", "", t), re.sub("<[^>]+>", "", d)) for n, t, d in
      re.findall(r'<article class="stage[^"]*" id="stage-(\d)">\s*<header class="st-h"><i>\d</i><div><h3>(.*?)</h3><p>(.*?)</p>', pg)}
IW = 370                                      # illustration width


def tile(paths, name, cols, h_each):
    """Images side by side / in a grid, 6 px gaps, IW wide."""
    rows = (len(paths) + cols - 1) // cols; w = (IW - 6 * (cols - 1)) // cols
    im = Image.new("RGB", (IW, rows * h_each + 6 * (rows - 1)), "white")
    for k, f in enumerate(paths):
        t = Image.open(f).convert("RGB"); r = max(w / t.width, h_each / t.height)
        t = t.resize((round(t.width * r), round(t.height * r)), Image.LANCZOS)
        x0, y0 = (t.width - w) // 2, (t.height - h_each) // 2
        im.paste(t.crop((x0, y0, x0 + w, y0 + h_each)), ((k % cols) * (w + 6), (k // cols) * (h_each + 6)))
    return save(im, name, IW)


U = "pieces/unrequited"
ill = {}
avf = Image.open("pieces/artists/r.amber.jpeg"); avf = avf.crop((80, 40, avf.width - 80, 40 + avf.width - 160))
ill[0] = (f'<div class="in0"><img src="{save(avf, "ov0.jpg", 150)}" width="150"><div><div class="chiplab">Song prompt</div>'
          f'<p class="paper">“A song about unrequited love”</p><div class="chiplab">Avatar still ←</div></div></div>')
ly = [l for l in open(f"{U}/inputs/lyrics.txt").read().splitlines() if l.strip()][1:8]
ill[1] = '<div class="paper lyr">' + "".join(f'<div class="{"hd" if l.startswith("[") else ""}">{esc(l.strip("[]"))}</div>' for l in ly) + "</div>"
sc_ = {l.split()[0].split("/")[-1]: l for l in open(f"{U}/outputs/song-song/scores.txt").read().splitlines() if l.strip()}
tk = ""
for f, colour in (("song_xl-sft_s1.wav", "#c9ced8"), ("song_xl-sft_s2.wav", "#2f7d9e")):
    y = audio(f"{U}/outputs/song-song/{f}", 0, 200, 2000)
    pct = sc_[f].split()[-1]
    tk += (f'<div class="chiplab">Take {f[-5]} · Whisper heard {pct} of the lyric{" · kept" if f.endswith("s2.wav") else ""}</div>'
           f'<svg width="{IW}" height="44">{wave(y, 44, colour, W=IW)}</svg>')
ill[2] = tk
st_ = ""
for n, f, c in (("vocals", f"{P}/analysis/song-vocals.mp3", "#2f7d9e"), ("drums", f"{P}/analysis/song-drums.mp3", "#b8642e"),
                ("bass", f"{P}/analysis/song-bass.mp3", "#6b4fa0"), ("other", f"{P}/analysis/song-other.mp3", "#2f9e6b")):
    y = audio(f, 58, 66)
    st_ += f'<div class="stem"><span>{n}</span><svg width="{IW - 70}" height="34">{wave(y, 34, c, W=IW - 70)}</svg></div>'
ill[3] = st_
ill[4] = (f'<img src="{tile([f"{P}/inputs/universes/pick{k}_turn.png" for k in (1, 2, 3)], "ov4.jpg", 3, 150)}" width="{IW}">'
          '<div class="chiplab">3 of 15 worlds tried: felted ash, wet clay, stained glass</div>')
ill[5] = (f'<div class="ovl"><img src="{save(Image.open(f"{V}/keys/shot_020.png"), "ov5.jpg", IW)}" width="{IW}">'
          '<span>“Free is all we want to be” → her back, mid-stride, the cloth visible</span></div>')
ill[6] = (f'<img src="{tile([f"{V}/keys/shot_{i:03d}.png" for i in range(30, 34)], "ov6.jpg", 2, 101)}" width="{IW}">'
          '<div class="chiplab">one still per bar: 4 bars, 4 stills</div>')
ill[7] = (f'<img src="{tile([f"{V}/ltx_chain/{n:05d}.png" for n in (2885, 2918, 2934)], "ov7.jpg", 3, 100)}" width="{IW}">'
          '<div class="chiplab">the corridor becomes the plaza in a dust storm</div>')
ill[8] = (f'<div class="ovl"><img src="{frame(f"{V}/noble-hunt-v6.mp4", 180.0, "ov8.jpg", IW)}" width="{IW}">'
          '<span>▶ The finished video · 4:02</span></div>')
rows = "".join(f'<div class="ovr"><div class="ovt"><i>{n}</i><div><h3>{esc(TT[n][0])}</h3><p>{esc(TT[n][1])}</p></div></div>'
               f'<div class="ovi">{ill[n]}</div></div>' for n in range(9))
figs.insert(0, f'<figure class="ex ov" id="0-overview"><div class="task">poptok · prompt to music video</div>'
               f'<h2>From one line of text to a music video</h2><div class="ovs">{rows}</div>'
               f'<p class="src">Tasks 0–2: Unrequited, a song the pipeline wrote and sang. Tasks 3–8: Noble Hunt v6.</p></figure>')

# ---------------------------------------------------------------- page
CSS = """
body { margin: 0; background: #eef0f4; font-family: "IBM Plex Sans", system-ui, sans-serif; color: #1d2433; }
main { display: grid; gap: 32px; justify-content: center; padding: 32px 0; }
figure.ex { width: 760px; box-sizing: border-box; margin: 0; padding: 26px 24px 20px; background: #fff; }
.task { font: 600 17px/1 "IBM Plex Mono", monospace; letter-spacing: .06em; text-transform: uppercase; color: #5b6475; }
h2 { font: 700 32px/1.15 "Archivo Narrow", sans-serif; margin: 10px 0 16px; }
.lab { font-size: 19px; color: #5b6475; margin: 14px 0 5px; } .lab b { color: #1d2433; }
.big { font-size: 23px; line-height: 1.35; margin: 0 0 4px; }
p { font-size: 20px; line-height: 1.4; margin: 0 0 6px; }
.mono { font: 18px/1.45 "IBM Plex Mono", monospace; background: #f4f6f9; padding: 10px 12px; color: #2b3445; }
.dim { color: #5b6475; font-size: 19px; }
.cap { font-size: 20px; line-height: 1.42; margin-top: 18px; padding-top: 14px; border-top: 1.5px solid #d9dde5; }
.src { font: 15px "IBM Plex Mono", monospace; color: #8a92a3; margin: 6px 0 0; }
img { display: block; }
.two { display: grid; grid-template-columns: 352px 352px; gap: 8px; } .two .lab, .three .lab { margin: 6px 0 10px; }
.three { display: grid; grid-template-columns: repeat(3, 232px); gap: 8px; }
.side { display: grid; grid-template-columns: 300px 1fr; gap: 20px; align-items: start; }
.side .lab:first-child { margin-top: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.chip { font-size: 21px; line-height: 1.3; padding: 8px 14px; border: 1.5px solid #c9ced8; border-radius: 6px; color: #5b6475; }
.chip.pick { border-color: #1d2433; box-shadow: 0 0 0 1.5px #1d2433; color: #1d2433; font-weight: 600; }
.pk { margin-bottom: 18px; } .pk .big { margin-top: 6px; }
.logline { font-size: 23px; font-style: italic; line-height: 1.4; margin-bottom: 14px; }
table.tl { border-collapse: collapse; width: 100%; } .tl td { font-size: 20px; line-height: 1.38; padding: 10px 8px 10px 0;
  border-top: 1px solid #d9dde5; vertical-align: top; } .tl td:first-child { width: 112px; padding-right: 18px; white-space: nowrap; }
.key { display: flex; flex-wrap: wrap; gap: 8px 22px; font-size: 19px; color: #5b6475; margin-top: 8px; }
.key i { display: inline-block; width: 16px; height: 16px; margin-right: 8px; vertical-align: -2px; }
.ovs { position: relative; } .ovs::before { content: ""; position: absolute; left: 19px; top: 30px; bottom: 60px; width: 3px; background: #d9dde5; }
.ovr { display: grid; grid-template-columns: 318px 370px; gap: 24px; align-items: center; padding: 14px 0; border-top: 1.5px solid #eceef2; position: relative; }
.ovr:first-child { border-top: 0; }
.ovt { display: grid; grid-template-columns: 42px 1fr; gap: 12px; align-items: start; }
.ovt i { font: 700 21px/40px "Archivo Narrow", sans-serif; font-style: normal; width: 40px; height: 40px; border-radius: 50%; background: #1d2433; color: #fff; text-align: center; position: relative; }
.ovt h3 { font: 700 25px/1.12 "Archivo Narrow", sans-serif; margin: 6px 0 6px; }
.ovt p { font-size: 18.5px; line-height: 1.35; color: #5b6475; margin: 0; }
.chiplab { font-size: 17px; color: #5b6475; margin: 4px 0 2px; }
.paper { background: #faf7f0; border: 1px solid #e6dfcf; padding: 8px 12px; font-size: 19px; line-height: 1.35; margin: 0 0 4px; }
.lyr { font-family: Georgia, serif; font-style: italic; font-size: 18px; } .lyr .hd { font-style: normal; font: 600 14px "IBM Plex Mono", monospace; color: #8a92a3; text-transform: uppercase; margin-top: 2px; }
.in0 { display: grid; grid-template-columns: 150px 1fr; gap: 12px; align-items: center; }
.stem { display: grid; grid-template-columns: 64px 1fr; gap: 6px; align-items: center; } .stem span { font-size: 17px; color: #5b6475; }
.ovl { position: relative; } .ovl span { position: absolute; left: 0; right: 0; bottom: 0; padding: 6px 10px; font-size: 17px; line-height: 1.3; color: #fff; background: rgba(15, 18, 26, .72); }
svg { display: block; font-family: "IBM Plex Sans", sans-serif; } svg.axis { margin-bottom: 4px; }
"""
open(PAGE, "w").write(
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<title>Pipeline examples</title><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@500;600;700'
    '&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap">'
    f'<style>{CSS}</style></head><body><main>{"".join(figs)}</main></body></html>')
os.remove(f"{SRC}/_f.png")
print(f"{len(figs)} figures -> {PAGE}")
subprocess.run(["node", "docs/design/page_cards.mjs", PAGE, OUT, "--select", "figure.ex", "--scale", "1", "--png-only"], check=True)
