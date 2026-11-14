"""Illustrated card for the 3D-world experiments (AI+World, 26-28 Oct 2026), in the style of the illustrated task cards.
Facts from docs/reports/20261108-3d-ground-truths.md; frames from the runs' own videos and sheets (pieces/_bench/...).

    python3 docs/design/world_card.py          (from poptok/, on gruntus: ffmpeg, PIL, node, chrome)
-> docs/design/20261108-world-card.html and docs/design/20261108-world-card/world-experiments.png/.svg
"""
import html, os, re, subprocess

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUTP, OUT = "docs/design/20261108-world-card.html", "docs/design/20261108-world-card"
SRC, REL = f"{OUT}/src", "20261108-world-card/src"
B = "pieces/_bench"
os.makedirs(SRC, exist_ok=True)
esc = html.escape


def thumb(src, name, w=150, h=84, t=None, box=None):
    """A frame (video at t s, or an image), optional crop box, centre-cropped to w:h, saved at 2x; returns an <img>."""
    if t is not None:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t}", "-i", src, "-frames:v", "1", f"{SRC}/_f.png"], check=True)
        src = f"{SRC}/_f.png"
    im = Image.open(src).convert("RGB")
    if box: im = im.crop(box)
    r = max(2 * w / im.width, 2 * h / im.height)
    im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    x0, y0 = (im.width - 2 * w) // 2, (im.height - 2 * h) // 2
    im.crop((x0, y0, x0 + 2 * w, y0 + 2 * h)).save(f"{SRC}/{name}", quality=88)
    return f'<img src="{REL}/{name}" width="{w}" height="{h}">'


def side(*imgs, cap=""):
    return '<div class="sd">' + "".join(imgs) + (f'<div class="tc">{cap}</div>' if cap else "") + "</div>"


def strip(*imgs, cap=""):
    return '<div class="th"><div class="row">' + "".join(imgs) + "</div>" + (f'<div class="tc">{cap}</div>' if cap else "") + "</div>"


def step(do, by, extra=""):
    return f'<li><span class="fdo">{do}</span><span class="fby">{by}</span>{extra}</li>'


V = {"yes": ("✓", "worked"), "part": ("◐", "partly"), "no": ("✗", "failed")}


def exp(v, title, what, extra=""):
    icon, word = V[v]
    return (f'<li class="v-{v}"><div><span class="fdo"><b class="vd">{icon} {word}</b><br><b>{title}</b></span>'
            f'<span class="vt">{what}</span></div>{extra}</li>')


R10, R26, R28 = f"{B}/review/2026-10-26", f"{B}/review/2026-10-26", f"{B}/review/2026-10-28"
flow = ('<div class="flow">' + "".join(
    f'<div><img src="{REL}/{n}" width="160" height="90"><span>{c}</span></div>' + ('<b>→</b>' if k < 3 else "")
    for k, (n, c) in enumerate([
        (thumb(f"{B}/world/room/clay_A.mp4", "f1.jpg", 160, 90, t=1.7) and "f1.jpg", "1 the room in Blender"),
        (thumb(f"{B}/world/flat2/flat_A.mp4", "f2.jpg", 160, 90, t=1.7) and "f2.jpg", "2 flat-colour layout"),
        (thumb(f"{B}/world/flat2/views/001_styled.png", "f3.jpg", 160, 90) and "f3.jpg", "3 styled training view"),
        (thumb(f"{R28}/world-lora-1000-nofirst-B.mp4", "f4.jpg", 160, 90, t=1.7) and "f4.jpg", "4 LTX + world LoRA")])) + "</div>")
pipeline = "".join([
    step("Build the room and two camera moves (A in by the door; B back from the window), and her body, dressed like her and moved by a motion-capture clip",
         "Blender by script + MakeHuman + CMU motion capture"),
    step("Render the layout in flat colours, one per kind of thing (the Matrix-Game 4.0 idea)", "Blender"),
    step("Restyle 80 views into the felt world; keep the 58 that match their layout", "Image editing model (Qwen-Image-Edit 2511)"),
    step("Train a world add-on (LoRA) on those pairs", "Lightricks' LTX-2 trainer, starting from their Layout-To-Render add-on: 197 min on a Spark"),
    step("Render each shot from the layout video and the LoRA; put her face into the first frame", "Lightricks LTX-2.5 + Layout-To-Render (an IC-LoRA); face pass"),
])
experiments = "".join([
    exp("yes", "The video follows the 3D layout", "Lightricks' Layout-To-Render takes a plain 3D render as the shot's motion: camera and geometry followed frame by frame, no reinvented room, no walk-and-rewind (rounds 1, 4b).",
        side(thumb(f"{B}/world/room/clay_A.mp4", "e1a.jpg", 112, 63, t=1.0),
              thumb(f"{R26}/world-round4b-l2r_nokeys_A.mp4", "e1b.jpg", 112, 63, t=1.0), cap="layout · LTX")),
    exp("no", "Styled stills pinned as the look", "Each still was styled on its own, so each shot was a different room; the stand-in became a toy soldier (rounds 1–3).",
        side(thumb(f"{B}/world/room/l2r_look2_A.mp4", "e2a.jpg", 112, 63, t=1.7), thumb(f"{B}/world/room/l2r_look2_B_indep.mp4", "e2b.jpg", 112, 63, t=1.7),
              cap="shot A · shot B")),
    exp("no", "Painting the look onto the 3D room", "Textures projected from styled views streaked and became a patchwork: “a comically incoherent mess” (rounds 4–5).",
        side(thumb(f"{R26}/world-round5-l2r_ing_nokeys_B.mp4", "e3.jpg", t=0.0))),
    exp("no", "Wan VACE as the renderer", "No figure at all, just a flat silhouette, at a lower resolution.",
        side(thumb(f"{R26}/world-round5-vace_B.mp4", "e4.jpg", t=1.7))),
    exp("no", "Flat colours, no training", "LTX copies the flat colours literally: green walls, a brown floor. The look has to come from training.",
        side(thumb(f"{R26}/world-control-l2r_nokeys_A.mp4", "e5.jpg", t=1.7))),
    exp("no", "A reference sheet (Lightricks' Ingredients add-on)", "No effect on the room; the uniform sheet turned her dark-haired."),
    exp("part", "A dressed, motion-captured body", "Her figure (hair, clothes, pose, motion) from front and back, but a slightly CG woman, not her face.",
        side(thumb(f"{R26}/her-dressed-l2r_nokeys_A.mp4", "e7.jpg", t=1.7, box=(840, 600, 1340, 881)))),
    exp("yes", "A world LoRA trained on one room", "The two camera moves show the same felt room: the first time two shots matched. 58 pairs, ~3 hours.",
        side(thumb(f"{R28}/world-lora-1000-nofirst-A.mp4", "e8a.jpg", 112, 63, t=1.7), thumb(f"{R28}/world-lora-1000-nofirst-B.mp4", "e8b.jpg", 112, 63, t=1.7),
              cap="shot A · shot B")),
    exp("no", "Keeping her face in the video", "Her at frame 0 (0.61 like her photo; pinned frames face-passed to 0.87–0.89), but redrawn as a similar woman within 8–32 frames: 0.26–0.33.",
        side(thumb(f"{R28}/world-lora-1000-d19s2-B.mp4", "e9a.jpg", 112, 63, t=0.0), thumb(f"{R28}/world-lora-1000-d19s2-B.mp4", "e9b.jpg", 112, 63, t=1.7),
              cap="frame 0 · 1.7 s")),
])
nxt = "".join(f"<li>{x}</li>" for x in [
    "<b>Close shots for identity (D28):</b> the same clip with her face ≥ 200 px wide. Is a small face the cause?",
    "<b>Her 3D mesh through LTX:</b> built from one photo (TRELLIS.2), never rendered in the room.",
    "<b>Train her into the world LoRA</b>, with restyled targets that keep her blond.",
    "<b>Across a real cut:</b> many shots and places, not two moves through one room.",
])
card = f'''<article class="stage world" id="world-experiments">
<header class="st-h"><i>3D</i><div><h3>Experiment: a 3D world as ground truth</h3>
<p>Can a Blender scene fix the place, the camera and her body, so the video model only adds the look? Built on Lightricks' Layout-To-Render add-on for their LTX-2.5 video model. Noble Hunt's ruined room, 26–28 Oct</p></div></header>
<div class="st-io st-in"><div class="lbl">In</div><ul><li><b>A room built by script</b></li><li><b>Her body + a motion clip</b></li>
<li class="has">{thumb("pieces/artists/r.amber.jpeg", "in3.jpg", 40, 40, box=(90, 60, 694, 664))}<b>Her photo</b></li></ul></div>
<div class="st-arrow" aria-hidden="true"></div>
<div class="lbl">The pipeline that came out of it</div>{flow}<ul class="fsteps">{pipeline}</ul>
<div class="st-arrow" aria-hidden="true"></div>
<div class="lbl">What was tried, in order</div><ul class="fsteps exps">{experiments}</ul>
<p class="exp"><b>Verdict</b>The 3D scene works as ground truth for the place and the camera, and a world LoRA gives it a look that holds
from shot to shot. The singer's face doesn't hold: the video model redraws her within a second.</p>
<div class="st-io st-out nexts"><div class="lbl">Next (untested)</div><ul>{nxt}</ul></div>
<div class="st-needs"><div class="lbl">Hardware</div><div class="hwb">
<div><div class="w">Blender (in a container)</div><div class="m">processor · 76 s for both clay passes</div><div class="y">Any machine.</div></div>
<div><div class="w">World LoRA training (LTX-2.5 22B)</div><div class="m">peak 50 GB · 197 min for 1,500 steps (DGX Spark)</div><div class="y">Needs a 128 GB unified-memory machine.</div></div>
<div><div class="w">Video model + LoRA</div><div class="m">as task 7 · ~5 min per 81-frame clip (DGX Spark)</div><div class="y">As task 7.</div></div>
</div></div>
<p class="src">Report: docs/reports/20261108-3d-ground-truths.md</p>
</article>'''
EXTRA = """
.ill .st-h i { font-size: 20px; }
.ill .exps li::before { border-color: #9aa4ae; }
.ill .exps li.v-yes::before { background: #2f9e6b; border-color: #2f9e6b; }
.ill .exps li.v-no::before { background: #c0392b; border-color: #c0392b; }
.ill .exps li.v-part::before { background: #e8a043; border-color: #e8a043; }
.ill .vd { font: 600 13px var(--mono); letter-spacing: .05em; text-transform: uppercase; margin-right: 4px; }
.ill .v-yes .vd { color: #23794f; } .ill .v-no .vd { color: #a8322a; } .ill .v-part .vd { color: #b06d12; }
.ill .exps li { display: grid; grid-template-columns: minmax(0, 1fr) auto; column-gap: 14px; align-items: start; }
.ill .sd { display: grid; grid-template-columns: repeat(2, auto); gap: 4px; justify-content: end; }
.ill .sd img { border-radius: 7px; display: block; } .ill .sd img:only-of-type { grid-column: span 2; }
.ill .sd .tc { grid-column: 1 / -1; font-size: 13px; color: #7a828c; font-style: italic; text-align: right; }
.ill .flow { display: flex; align-items: center; gap: 4px; margin: 2px 0 12px; }
.ill .flow div { display: grid; gap: 4px; } .ill .flow img { border-radius: 8px; display: block; }
.ill .flow span { font-size: 13px; color: #5b6475; } .ill .flow b { color: #9aa4ae; font-size: 18px; }
.ill .vt { display: block; font-size: 16px; line-height: 1.45; color: #4d5661; margin-top: 4px; }
.ill .nexts ul { display: grid; gap: 8px; } .ill .nexts li { border-radius: 12px; display: block; font-size: 16.5px; line-height: 1.42; padding: 10px 14px; }
.ill .exp b { display: block; }
.ill .world .src { font: 13px var(--mono); color: #8a929c; margin: 14px 0 0; }
"""
pg = open("docs/design/20261107-pipeline.html").read()
style = re.search(r"<style>(.*?)</style>", pg, re.S).group(1) + open("docs/design/card_style.css").read() + EXTRA
open(OUTP, "w").write(
    '<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<title>3D world experiments</title><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@500;600;700'
    '&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
    f'<style>{style}</style></head><body class="ill"><main>{card}</main></body></html>')
if os.path.exists(f"{SRC}/_f.png"): os.remove(f"{SRC}/_f.png")
subprocess.run(["node", "docs/design/page_cards.mjs", OUTP, OUT, "--select", "article.world"], check=True)
