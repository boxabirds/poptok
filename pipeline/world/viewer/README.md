Browser viewer for a painted world (published as the "Felt Room Viewer" artifact, 2026-10-26). Build its model file
from export_world.py's painted.glb:  python3 -c "import base64; b=open('painted.glb','rb').read(); open('room.js','w').write('window.ROOM_GLB=\"'+base64.b64encode(b).decode()+'\";')"
(the artifact host serves no .glb, so the model ships base64 inside room.js). Orbit, shot cameras A/B, her animation.
