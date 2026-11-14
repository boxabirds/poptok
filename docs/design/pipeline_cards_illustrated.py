"""Illustrated task cards 3-8 for the article: the page's own cards (docs/design/20261107-pipeline.html, copied verbatim, so
the words stay the page's), restyled to flow (chips, a step timeline, a softer card) and given small real thumbnails and
example quotes from Noble Hunt v6 (New Parade v4 for a shot without the singer, as every Noble Hunt shot has her).

    python3 docs/design/pipeline_cards_illustrated.py          (from poptok/, on gruntus: ffmpeg, numpy, PIL, node, chrome)
-> docs/design/20261107-pipeline-cards-illustrated.html and docs/design/20261107-pipeline-cards-illustrated/task-N-*.png/.svg
Thumbnails are attached by position (task, in/step/out, index), so if a card's steps are reordered, check THUMBS below.
"""
import html, json, os, re, subprocess

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
PAGE = "docs/design/20261107-pipeline.html"
OUTP = "docs/design/20261107-pipeline-cards-illustrated.html"
OUT = "docs/design/20261107-pipeline-cards-illustrated"
SRC = f"{OUT}/src"; REL = "20261107-pipeline-cards-illustrated/src"
P, V, NP = "pieces/noble-hunt", "pieces/noble-hunt/outputs/v6", "pieces/new-parade/outputs/v4"
os.makedirs(SRC, exist_ok=True)
esc = html.escape
J = lambda f: json.load(open(f))


def mmss(t):
    t = round(t, 1); return f"{int(t // 60)}:{t % 60:04.1f}"


def thumb(src, name, w, h, box=None):
    """Centre-crop to w:h (after an optional crop box), saved at 2x for the 2x export; returns an <img>."""
    im = Image.open(src).convert("RGB")
    if box: im = im.crop(box)
    r = max(2 * w / im.width, 2 * h / im.height)
    im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    x0, y0 = (im.width - 2 * w) // 2, (im.height - 2 * h) // 2
    im.crop((x0, y0, x0 + 2 * w, y0 + 2 * h)).save(f"{SRC}/{name}", quality=88)
    return f'<img src="{REL}/{name}" width="{w}" height="{h}">'


def vframe(video, t, name):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", f"{SRC}/{name}"], check=True)
    return f"{SRC}/{name}"


def audio(path, t0, t1, sr=4000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0}", "-t", f"{t1 - t0}", "-i", path, "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def wave(y, w, h, colour, scale=None, extra=""):
    cols = np.array_split(y, w); s = scale or max(1e-6, np.abs(y).max())
    hi = [c.max() for c in cols]; lo = [c.min() for c in cols]
    top = " ".join(f"L{x},{h / 2 - hi[x] / s * h / 2 * .92:.1f}" for x in range(w))
    bot = " ".join(f"L{x},{h / 2 - lo[x] / s * h / 2 * .92:.1f}" for x in range(w - 1, -1, -1))
    return f'<svg width="{w}" height="{h}" class="wv">{extra}<path d="M0,{h / 2} {top} {bot} Z" fill="{colour}"/></svg>'


def strip(*imgs, cap=""):
    return '<div class="th"><div class="row">' + "".join(imgs) + "</div>" + (f'<div class="tc">{esc(cap)}</div>' if cap else "") + "</div>"


def quote(text, cap=""):
    return f'<div class="eg">{(f"<b>{esc(cap)}</b>") if cap else ""}“{esc(text)}”</div>'


shots = J(f"{V}/shots.json")["shots"]; sc = J(f"{V}/scenes.json")["scenes"]
ins = J(f"{V}/sing/inserts.json")[0]
st = J(f"{P}/inputs/story-scenes.json"); u = J(f"{P}/inputs/universes.json")
av = "pieces/artists/r.amber.jpeg"; AVF = (90, 60, 694, 664)          # her face, square
FACE = (370, 0, 530, 160)                                           # her head in the 896x448 lip-sync clip
key = lambda i: f"{V}/keys/shot_{i:03d}.png"
ltx = lambda n: f"{V}/ltx_chain/{n:05d}.png"

# mini audio pictures (task 3)
mix, voc, drm = f"{P}/inputs/song.wav", f"{P}/analysis/song-vocals.mp3", f"{P}/analysis/song-drums.mp3"
stems = [(voc, "#2f7d9e"), (drm, "#b8642e"), (f"{P}/analysis/song-bass.mp3", "#6b4fa0"), (f"{P}/analysis/song-other.mp3", "#2f9e6b")]
ys = [audio(f, 58, 66) for f, _ in stems]; pk = max(np.abs(y).max() for y in ys)
stem_svg = '<div class="stems">' + "".join(wave(y, 300, 28, c, pk) for y, (_, c) in zip(ys, stems)) + "</div>"
b = J(f"{P}/analysis/beats.json"); dh = J(f"{P}/analysis/drum_hits.json")
a, e = 52.4, 58.4; X = lambda t: (t - a) / (e - a) * 300
ticks = "".join(f'<line x1="{X(t):.1f}" y1="0" x2="{X(t):.1f}" y2="70" stroke="#1d2433" stroke-width="{2.5 if any(abs(t - d) < .03 for d in b["downbeats"]) else .8}"/>'
                for t in b["beats"] if a <= t <= e)
runs = "".join(f'<rect x="{X(r["start"]) - 2:.1f}" y="0" width="{X(r["end"]) - X(r["start"]) + 4:.1f}" height="70" fill="#f6d36b" opacity=".55"/>'
               for r in dh["runs"] if r["end"] > a and r["start"] < e)
yd = audio(drm, a, e)
seg = J(f"{P}/analysis/song-true-whisper-segments.json")["segments"][2]
wd = [w for s in J(f"{P}/analysis/song-medium.json")["segments"] for w in s.get("words", []) if seg["start"] - .4 <= w["start"] and w["end"] <= seg["end"] + .1]
xa, xb = seg["start"] - .3, seg["end"] + .2; XW = lambda t: (t - xa) / (xb - xa) * 300
boxes = "".join(f'<rect x="{XW(w["start"]):.1f}" y="{46 + 11 * (k % 2)}" width="{max(2, XW(w["end"]) - XW(w["start"]) - 1.5):.1f}" height="10" rx="2" fill="#2f7d9e" opacity=".55"/>'
                for k, w in enumerate(wd))
lyr_svg = wave(audio(voc, xa, xb), 300, 70, "#9fbcc9", extra=f'<g transform="translate(0,0) scale(1,.6)"></g>').replace("</svg>", boxes + "</svg>")

ev, mv = sc[19], sc[9]
fa, fb = mv["frames"]
THUMBS = {                                   # (task, part, index) -> html added to that chip / step
    (3, "in", 0): wave(audio(mix, 0, 242, 400), 60, 30, "#1d2433"),
    (3, "step", 0): strip(stem_svg, cap="vocals · drums · bass · other"),
    (3, "step", 1): strip(lyr_svg, cap="“A radio hums through someone’s wall”"),
    (3, "step", 2): strip(wave(yd, 300, 70, "#e3c2a8", extra=ticks), cap="126 beats a minute; thick = bar line"),
    (3, "step", 3): strip(wave(yd, 300, 70, "#b8642e", extra=runs), cap="yellow = a run of deep drum hits"),
    (3, "out", 0): wave(audio(voc, 58, 66), 60, 30, "#2f7d9e"),
    (4, "in", 0): thumb(av, "4in-av.jpg", 40, 40, AVF),
    (4, "step", 1): quote("a young woman with long wavy blond hair"),
    (4, "step", 2): quote(st["logline"], "Logline · "),
    (4, "step", 4): '<div class="th"><div class="names">' + " · ".join(
        f'<span class="{"pkn" if c["name"] in [p["name"] for p in u["picks"]] else ""}">{esc(c["name"])}</span>' for c in u["candidates"]) + "</div></div>",
    (4, "step", 5): strip(*(thumb(f"{P}/inputs/universes/pick{k}_opening_world.png", f"4s5-{k}.jpg", 150, 84) for k in (1, 2, 3)),
                          cap="felted ash · wet clay · stained glass, before her face goes in"),
    (4, "step", 6): strip(*(thumb(f"{P}/inputs/universes/pick{k}_turn.png", f"4s6-{k}.jpg", 150, 84) for k in (1, 2, 3)),
                          cap="her face put in: 0.23 · 0.66 · 0.65 like her photo"),
    (4, "out", 3): thumb(f"{P}/inputs/universes/sheet.jpg", "4out-sheet.jpg", 40, 40),
    (4, "out", 4): thumb(f"{P}/inputs/universes/pick1_turn.png", "4out-u1.jpg", 40, 40),
    (5, "step", 0): quote(next(s for s in shots if abs(s["start"] - 84.14) < .1)["cast_note"].split(": ", 1)[1], "1:24 · instrumental · "),
    (5, "step", 3): quote(shots[20]["prompt"][:shots[20]["prompt"].find(" ", 170)] + " …", "1:05 · "),
    (6, "in", 2): thumb(av, "6in-av.jpg", 40, 40, AVF),
    (6, "step", 0): strip(thumb(f"{NP}/keys/shot_000.png", "6s0-a.jpg", 150, 84), thumb(f"{NP}/keys/shot_010.png", "6s0-b.jpg", 150, 84),
                          cap="New Parade: no singer in shot"),
    (6, "step", 1): strip(thumb(av, "6s1-av.jpg", 84, 84, AVF), '<span class="arr">→</span>',
                          *(thumb(key(i), f"6s1-{i}.jpg", 150, 84) for i in (20, 59)), cap="from her photo: walking, singing"),
    (6, "step", 6): strip(*(thumb(key(i), f"6s6-{i}.jpg", 112, 63) for i in range(30, 34)), cap="4 bars → 1 video segment"),
    (6, "out", 0): thumb(key(31), "6out.jpg", 40, 40),
    (7, "in", 0): thumb(key(50), "7in.jpg", 40, 40),
    (7, "in", 3): wave(audio(voc, 138, 145), 60, 30, "#2f7d9e"),
    (7, "step", 0): quote(ev["transition_in"].split(". ")[0] + ".", "2:02 · "),
    (7, "step", 1): strip(*(thumb(ltx(fa + 5 + (fb - fa - 10) * k // 2), f"7s1-{k}.jpg", 150, 84) for k in range(3)),
                          cap="one take: each clip starts from the last frame"),
    (7, "step", 2): strip(*(thumb(ltx(n), f"7s2-{n}.jpg", 150, 84) for n in (2885, 2918, 2934)), cap="the corridor becomes the plaza"),
    (7, "step", 3): strip(*(thumb(vframe(f"{V}/sing/it_out_0.mp4", t, f"_ls{k}.png"), f"7s3-{k}.jpg", 84, 84, FACE)
                            for k, t in enumerate((0.5, 1.4, 2.3, 3.2, 4.1))), cap="“We’re carrying the only peace”"),
    (7, "out", 0): thumb(ltx(2960), "7out-f.jpg", 40, 40),
    (7, "out", 2): thumb(vframe(f"{V}/sing/it_out_0.mp4", 1.4, "_ls9.png"), "7out-s.jpg", 40, 40, FACE),
    (8, "step", 0): '<div class="th"><div class="pill">each bar played at 0.83–1.14× speed (median 1.01×)</div></div>',
    (8, "step", 4): strip(*(thumb(vframe(f"{V}/noble-hunt-v6.mp4", t, f"_sp{k}.png"), f"8s4-{k}.jpg", 150, 84)
                            for k, t in enumerate((ins["start"] - 1, ins["start"] + 2.5, ins["end"] + 1))), cap="before · singing clip · after"),
    (8, "out", 0): thumb(vframe(f"{V}/noble-hunt-v6.mp4", 180, "_fin.png"), "8out.jpg", 40, 40),
}

pg = open(PAGE).read()
style = re.search(r"<style>(.*?)</style>", pg, re.S).group(1)
cards = [re.search(rf'<article class="stage" id="stage-{n}">.*?</article>', pg, re.S).group(0) for n in range(3, 9)]
data = {f"{t}|{p}|{i}": h for (t, p, i), h in THUMBS.items()}

CSS = open("docs/design/card_style.css").read()
JS = """
const T = %s;
document.querySelectorAll('article.stage').forEach(card => {
  const n = card.id.split('-')[1];
  const put = (el, k, chip) => { const h = T[n + '|' + k]; if (!h || !el) return;
    if (chip) { el.insertAdjacentHTML('afterbegin', h); el.classList.add('has'); } else el.insertAdjacentHTML('beforeend', h); };
  card.querySelectorAll('.st-in li').forEach((li, i) => put(li, 'in|' + i, true));
  card.querySelectorAll('.st-out li').forEach((li, i) => put(li, 'out|' + i, true));
  card.querySelectorAll('.fsteps > li').forEach((li, i) => put(li, 'step|' + i, false));
  // hardware table -> one readable block per row
  card.querySelectorAll('table.hwt').forEach(t => {
    const rows = [...t.querySelectorAll('tbody tr')].map(r => [...r.children].map(c => c.innerHTML));
    const div = document.createElement('div'); div.className = 'hwb';
    div.innerHTML = rows.map(c => `<div><div class="w">${c[0]}</div><div class="m">${c[1]} · ${c[2]}</div>` +
      `<div class="y">${c[3]} <i>Runs on: ${c[4]}</i></div></div>`).join('');
    t.closest('.hwt-scroll').replaceWith(div);
  });
});
""" % json.dumps(data)
open(OUTP, "w").write(
    '<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<title>Illustrated task cards</title><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@500;600;700'
    '&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">'
    f'<style>{style}{CSS}</style></head><body class="ill"><main>{"".join(cards)}</main><script>{JS}</script></body></html>')
for f in os.listdir(SRC):
    if f.startswith("_"): os.remove(f"{SRC}/{f}")
print(f"{len(cards)} cards -> {OUTP}")
subprocess.run(["node", "docs/design/page_cards.mjs", OUTP, OUT], check=True)
