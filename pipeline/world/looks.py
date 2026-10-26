"""World-building test, look frames (Qwen-Image-Edit-2511, in morph-spark on a Spark): the art direction for
Layout-To-Render, made from each clay pass's first frame, and the baseline stills made the way poptok makes them
today (text and her photo only, no layout).

    python3 pipeline/world/looks.py OUT PHOTO
Reads OUT/clay_A/0001.png, OUT/clay_B/0001.png; writes OUT/look_A.png, look_B_indep.png (B styled on its own),
look_B_carry.png (B styled with look_A as a second reference: the same room's materials carried over),
base_A.png, base_B.png.
"""
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qwen_edit import load

out, photo = sys.argv[1], sys.argv[2]
W, H = 1920, 1088
UNI = ("Felted Ruin: a dense, tactile world of raw wool felt, where walls, floor, furniture and rubble are thick, matte "
       "blocks of matted fibres. Light is absorbed rather than reflected, deep soft shadows; a muted, earthy palette of "
       "greens, greys and browns; every surface slightly fuzzy, catching the glow in a soft halo.")
HER = ("a young woman with long wavy blond hair, a soldier in a worn field uniform of dark felt, her face streaked with "
       "green and black camouflage paint, holding a wool cloth with a hand-stitched peace symbol (a circle with one "
       "vertical line and two lines from its centre down to the lower left and lower right)")
ROOM = ("a ruined apartment room at night: a window in the far wall onto a wet street with a distant glow, a tilted "
        "table, a chair on its side, rubble blocks and a crate, a collapsed ceiling slab, an open door in the near wall")
LAYOUT = ("Image 1 is a grey 3D layout. Turn it into a finished film frame: keep exactly its camera, framing, perspective "
          "and the position, size and shape of every object. Everything is made of felted wool. " + UNI + " The room is "
          + ROOM + ". The seated figure under the window is the woman from image {her}, the same face and hair: " + HER +
          ". No text or lettering.")
CARRY = (" Image 2 is the same room seen from another angle: the same materials, colours, textures, light and objects, "
         "so this frame shows the same place.")
edit = load()
jobs = [
    ("look_A", [f"{out}/clay_A/0001.png", photo], LAYOUT.format(her=2)),
    ("look_B_indep", [f"{out}/clay_B/0001.png", photo], LAYOUT.format(her=2)),
    ("look_B_carry", [f"{out}/clay_B/0001.png", f"{out}/look_A.png", photo], LAYOUT.format(her=3) + CARRY),
    ("base_A", [photo], f"{UNI} {ROOM.capitalize()}. Seen from the open door, a wide shot across the room to the "
                        f"window; under the window sits the woman from the reference image, the same face and hair: "
                        f"{HER}. No text or lettering."),
    ("base_B", [photo], f"{UNI} {ROOM.capitalize()}. Seen from the window corner, close beside the woman from the "
                        f"reference image sitting under the window (the same face and hair: {HER}), looking back "
                        f"across the room to the open door. No text or lettering."),
]
for name, refs, prompt in jobs:
    if os.path.exists(f"{out}/{name}.png"):
        continue
    t = time.time()
    edit(refs, prompt, seed=7, size=(W, H)).save(f"{out}/{name}.png")
    print(f"{name}: {time.time() - t:.0f} s", flush=True)
