"""Let LONGCAT_ATTN=xformers|flash2|flash3 override the attention backend that LongCat's model config.json asks for
(flash-attn 2, which isn't built for GB10). Inserted in every attention class's __init__ (self- and cross-attention,
and the avatar audio cross-attention) right after its flags are set.
    python3 attn_override.py FILE...
"""
import re, sys

HOOK = '''        _b = __import__("os").environ.get("LONGCAT_ATTN")
        if _b:
            self.enable_flashattn3, self.enable_flashattn2, self.enable_xformers = (
                _b == "flash3", _b == "flash2", _b == "xformers")
'''
for path in sys.argv[1:]:
    s = open(path).read()
    anchor = "        self.enable_xformers = enable_xformers\n"
    if anchor not in s:
        sys.exit(f"{path}: no '{anchor.strip()}' to hook after")
    s = s.replace(anchor, anchor + HOOK)
    open(path, "w").write(s)
    print("patched", path, s.count(HOOK))
