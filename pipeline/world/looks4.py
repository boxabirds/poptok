"""Round 4 look frames: her added (from her photo) to the textured world's own frames (project_texture.py:
OUT/texclean_{A,B}/{0001,0041,0081}.png), so every look frame shows the same room. Frames where her stand-in is out of
shot (OUT/../figure.json) keep the room alone.
    python3 pipeline/world/looks4.py OUT FIGURE_JSON PHOTO
"""
import json, os, sys, time

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qwen_edit import load

out, figj, photo = sys.argv[1:4]
UNIFORM = ("a worn soldier's field uniform of dark green-brown felted wool, her face streaked with green and black "
           "camouflage paint")
CLOTH = ("a cream wool cloth with a hand-stitched dark peace symbol (a circle with one vertical line and two lines from "
         "its centre down to the lower left and lower right)")
ADD = ("Keep this image exactly (composition, camera, materials, light). Add the woman from image 2, the same person "
       "(her face, long wavy blond hair, eyes and age exactly), life-size and real, not a doll or a toy: she sits on the "
       "floor with her back against the wall under the window, knees drawn up, holding {cloth}, wearing {uniform}, lit "
       "by the room's own dim light. She is {where}. No text.")


def place(x, y):
    h = "on the left" if x < 0.38 else "on the right" if x > 0.62 else "in the middle"
    v = "low in the frame" if y > 0.6 else "high in the frame" if y < 0.4 else "at mid height"
    return f"{h}, {v}"


fig = json.load(open(figj))
edit = load()
for p in "AB":
    for f in ("0001", "0041", "0081"):
        dst = f"{out}/look4_{p}_{f}.png"
        if os.path.exists(dst):
            continue
        k = fig[f"{p}_{f}"]
        if not k["visible"]:
            Image.open(f"{out}/texclean_{p}/{f}.png").convert("RGB").save(dst)
            continue
        t = time.time()
        edit([f"{out}/texclean_{p}/{f}.png", photo], ADD.format(cloth=CLOTH, uniform=UNIFORM, where=place(k["x"], k["y"])),
             seed=7, size=(1920, 1088)).save(dst)
        print(f"look4_{p}_{f}: {time.time() - t:.0f} s", flush=True)
