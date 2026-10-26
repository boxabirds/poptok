"""World-building test, round 3 look frames (Qwen-Image-Edit-2511, in morph-spark on a Spark).

Round 2 lessons: restyling a clay frame on its own keeps the layout; a second reference image makes the model copy
that image's camera; the stand-in figure gets styled as an object. So here:
  1. each clean frame (no stand-in) is restyled alone, with the same prompt and seed: look3_{A,B}_{0001,0041,0081}_room
  2. she is added where the stand-in was (OUT/figure.json), from her photo: look3_..._{frame}.png (frames where the
     stand-in is out of shot keep the room alone)
  3. a reference sheet for the Ingredients IC-LoRA: her face close-up, her full-body turnaround in the felt uniform,
     the cloth, and the room (look3_A_0001_room), on black, no text: sheet.png (1920x1088)

    python3 pipeline/world/looks3.py OUT PHOTO
"""
import json, os, sys, time

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qwen_edit import load

out, photo = sys.argv[1], sys.argv[2]
W, H = 1920, 1088
UNI = ("Felted Ruin: a dense, tactile world of raw wool felt, where walls, floor, furniture and rubble are thick, matte "
       "blocks of matted fibres. Light is absorbed rather than reflected, deep soft shadows; a muted, earthy palette of "
       "greens, greys and browns; every surface slightly fuzzy, catching the glow in a soft halo.")
ROOM = ("a ruined apartment room at night: a window onto a wet street with a distant glow, a tilted table, a chair on "
        "its side, rubble blocks and a crate, a collapsed ceiling slab, an open door")
RESTYLE = ("Restyle this image. Keep EXACTLY its composition: the same camera position, angle and lens, the same "
           "framing, every object and wall in the same place with the same size and outline. Only change the materials "
           "and light: " + UNI + " The room is " + ROOM + ". No people. No text.")
UNIFORM = ("a worn soldier's field uniform of dark green-brown felted wool, her face streaked with green and black "
           "camouflage paint")
CLOTH = ("a cream wool cloth with a hand-stitched dark peace symbol (a circle with one vertical line and two lines from "
         "its centre down to the lower left and lower right)")


def place(x, y):
    h = "on the left" if x < 0.38 else "on the right" if x > 0.62 else "in the middle"
    v = "low in the frame" if y > 0.6 else "high in the frame" if y < 0.4 else "at mid height"
    return f"{h}, {v}"


ADD = ("Keep this image exactly (composition, camera, materials, light). Add the woman from image 2, the same person "
       "(her face, long wavy blond hair, eyes and age exactly), life-size and real, not a doll or a toy: she sits on the "
       "floor with her back against the wall under the window, knees drawn up, holding {cloth}, wearing {uniform}, "
       "lit by the room's own dim light. She is {where}. No text.")
edit = load()
fig = json.load(open(f"{out}/figure.json"))


def make(name, refs, prompt, seed=7, size=(W, H)):
    if os.path.exists(f"{out}/{name}.png"):
        return
    t = time.time()
    edit(refs, prompt, seed=seed, size=size).save(f"{out}/{name}.png")
    print(f"{name}: {time.time() - t:.0f} s", flush=True)


for p in "AB":
    for f in ("0001", "0041", "0081"):
        make(f"look3_{p}_{f}_room", [f"{out}/clean_{p}/{f}.png"], RESTYLE)
        k = fig[f"{p}_{f}"]
        if k["visible"]:
            make(f"look3_{p}_{f}", [f"{out}/look3_{p}_{f}_room.png", photo],
                 ADD.format(cloth=CLOTH, uniform=UNIFORM, where=place(k["x"], k["y"])))
        elif not os.path.exists(f"{out}/look3_{p}_{f}.png"):
            Image.open(f"{out}/look3_{p}_{f}_room.png").save(f"{out}/look3_{p}_{f}.png")
# the Ingredients reference sheet: one clean panel per element, on black, no text
make("ref_face", [photo], f"A front-facing close-up portrait of the woman from the reference image, the same person "
     f"(face, hair, eyes and age exactly), her face streaked with green and black camouflage paint, the collar of {UNIFORM}, "
     f"soft even light, plain black background. No text.", size=(768, 1024))
make("ref_body", [photo], f"A character turnaround of the woman from the reference image, the same person: three full-body "
     f"views side by side, front, side and back, standing, wearing {UNIFORM}, felt boots; even studio light, plain black "
     f"background. No text.", size=(1536, 1024))
make("ref_cloth", [photo], f"A product photo of {CLOTH} alone, held flat, its peace symbol clearly visible, soft even "
     f"light, plain black background. No people. No text.", size=(768, 768))
if not os.path.exists(f"{out}/sheet.png"):
    sheet = Image.new("RGB", (W, H), "black")
    face = Image.open(f"{out}/ref_face.png").resize((480, 640))
    body = Image.open(f"{out}/ref_body.png").resize((960, 640))
    cloth = Image.open(f"{out}/ref_cloth.png").resize((400, 400))
    room = Image.open(f"{out}/look3_A_0001_room.png").resize((720, 408))
    sheet.paste(face, (20, 20)); sheet.paste(body, (520, 20))
    sheet.paste(cloth, (20, 670)); sheet.paste(room, (1180, 670))
    sheet.save(f"{out}/sheet.png")
    print("sheet.png", flush=True)
