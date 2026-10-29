"""poptok/macpass step 2: the keyframes, macvideo's identity chain (gen_keyframes.py) with Qwen-Image-Edit in place of
Gemini: each shot is drawn from its Pass B prompt with the keyframes before it as references, so the world, the light and
the people carry from shot to shot (macvideo keeps the last 4; Qwen-Edit takes 3 pictures, so the performer's photo and
the last 2). Runs on a Spark, in order (each keyframe needs the one before); a keyframe already drawn is kept.
    python3 macpass/keys.py PIECE        (reads PIECE/macpass/{shots,image_prompts}.json; writes PIECE/macpass/keyframes/)"""
import json, os, sys

sys.path.insert(0, "pipeline")
from PIL import Image

from qwen_edit import load
from workshop import read_env

WINDOW = 2
SIZE = (1664, 928)                        # Qwen-Image's own 16:9; LTX renders 1920x1088 from it
PROMPT = ("Picture 1 shows the face of the narrator, {performer}: when she is in the shot, she has exactly this face "
          "(only her face from it; never picture 1 itself pasted in, and never two of her). {chain}Draw this keyframe "
          "of the music video, filling the whole frame: {prompt} No text.")
CHAIN = ("The other pictures are the shots just before this one in the same video: the same world, the same place where "
         "the shot stays there, the same drawing style and the same light. ")

P = sys.argv[1].rstrip("/"); e = read_env(f"{P}/piece.env"); O = f"{P}/macpass"; K = f"{O}/keyframes"
os.makedirs(K, exist_ok=True)
shots = json.load(open(f"{O}/shots.json"))["shots"]; ip = json.load(open(f"{O}/image_prompts.json"))
photo = os.path.normpath(f"{P}/{e['PERFORMER_PHOTO']}")
edit = None; recent = []
for s in shots:
    f = f"{K}/keyframe_{s['index']:03d}.png"
    if not os.path.exists(f):
        edit = edit or load()
        im = edit([photo] + recent, PROMPT.format(performer=e["PERFORMER"], chain=CHAIN if recent else "",
                                                 prompt=ip[f"shot_{s['index']:03d}"]), seed=s["index"], size=SIZE)
        im.convert("RGB").save(f)
        print(f"keyframe {s['index']}: {s['target_text']}", flush=True)
    recent = (recent + [f])[-WINDOW:]
