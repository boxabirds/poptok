"""poptok/macpass step 2: the keyframes, macvideo's identity chain (gen_keyframes.py) with Qwen-Image-Edit in place of
Gemini: each shot is drawn from its Pass B prompt with the keyframes before it as references, so the world, the light and
the people carry from shot to shot (macvideo keeps the last 4; Qwen-Edit takes 3 pictures, so the performer's photo and
the last 2). Runs on a Spark, in order (each keyframe needs the one before); a keyframe already drawn is kept.
    python3 macpass/keys.py PIECE        (reads PIECE/macpass/{shots,image_prompts}.json; writes PIECE/macpass/keyframes/)"""
import json, os, re, sys

sys.path.insert(0, "pipeline")
from PIL import Image

from qwen_edit import load
from workshop import read_env

WINDOW = 2
SIZE = (1664, 928)                        # Qwen-Image's own 16:9; LTX renders 1920x1088 from it
# Qwen-Edit edits what it is given, where Gemini composes afresh: with the last keyframes and her photo always passed
# (cuckoo, 2026-10-29), shots 2-5 came out as one frame four times, she was in every shot, and shot 1 was her colour
# photo pasted into the forest. So: her photo only in a shot whose words name her, the earlier shots named as the
# world and style to keep (never the picture to copy), and the style (with its colours) first
PROMPT = ("A {filt}, made in exactly that medium and its colours alone. {face}{chain}Draw this new keyframe of the music "
          "video, filling the whole frame, its own composition, framing and subject exactly as described: {prompt} "
          "No text.")
FACE = ("Picture 1 is a photograph of the narrator, {performer}: the woman in this shot has her face, drawn in the "
        "same {filt} (never the photograph itself pasted in, and only one of her). ")
CHAIN = ("{which} the shots just before this one in the same video: keep their world, their drawing style and their "
         "light, but never copy their composition, framing or subject; this shot shows something new. ")
HER = re.compile(r"\b(woman|girl|she|her|hers|narrator|figure|face)\b", re.I)

P = sys.argv[1].rstrip("/"); LAST = int(sys.argv[2]) if len(sys.argv) > 2 else 0     # a test: shots 1..LAST
e = read_env(f"{P}/piece.env"); O = f"{P}/macpass"; K = f"{O}/keyframes"
os.makedirs(K, exist_ok=True)
filt = json.load(open(f"{O}/brief.json"))["filter"]
shots = json.load(open(f"{O}/shots.json"))["shots"][:LAST or None]; ip = json.load(open(f"{O}/image_prompts.json"))
photo = os.path.normpath(f"{P}/{e['PERFORMER_PHOTO']}")
edit = None; recent = []
for s in shots:
    f = f"{K}/keyframe_{s['index']:03d}.png"
    if not os.path.exists(f):
        edit = edit or load()
        words = ip[f"shot_{s['index']:03d}"]; her = bool(HER.search(words))
        which = f"Picture{'s' if len(recent) > 1 else ''} {', '.join(str(k + 1 + her) for k in range(len(recent)))} " \
                f"{'are' if len(recent) > 1 else 'is'}"
        im = edit(([photo] if her else []) + recent or [Image.new("RGB", SIZE, "white")],
                  PROMPT.format(filt=filt, face=FACE.format(performer=e["PERFORMER"], filt=filt) if her else "",
                                chain=CHAIN.format(which=which) if recent else "", prompt=words),
                  seed=s["index"], size=SIZE)
        im.convert("RGB").save(f)
        print(f"keyframe {s['index']}: {'her photo, ' if her else ''}{s['target_text']}", flush=True)
    recent = (recent + [f])[-WINDOW:]
