"""Round 4: repaint the streaked surfaces of a textured render (surfaces the projected view never saw) to match the
clean ones, keeping the clean parts and the composition (one image only: a second reference makes Qwen-Edit copy its
camera, round 2).
    python3 pipeline/world/repair.py IN.png OUT.png
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qwen_edit import load

PROMPT = ("This is a view of a ruined apartment room made of thick felted wool. Some surfaces show stretched streaks "
          "and smears (texture that was never painted). Repaint ONLY those streaked surfaces as the same felted wool "
          "walls, floor, ceiling and objects, matching the clean parts in material, colour, fibre texture and light. "
          "Keep the clean parts exactly, and keep the composition, camera and every object's shape and place. "
          "No people. No text.")
edit = load()
edit([sys.argv[1]], PROMPT, seed=7, size=(1920, 1088)).save(sys.argv[2])
print("repaired", sys.argv[2])
