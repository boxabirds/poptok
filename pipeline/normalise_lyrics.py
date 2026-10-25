"""Songwriter lyrics -> the pipeline's canonical form, where every line that isn't sung is in [brackets] (the aligners,
the workshop and the cue readers all skip or read bracketed lines as structure):
  - a bare section heading, with or without a hint in parentheses: "Chorus 2 (a small group has formed — footsteps
    syncing)" -> "[Chorus 2 (a small group has formed — footsteps syncing)]" (the hint says what the action should be)
  - a whole-line stage direction: "(silence)", "(the cruel voice, spoken/flat)" -> "[(silence)]"
Sung lines and already-bracketed lines are left as they are.

    python3 pipeline/normalise_lyrics.py LYRICS.txt OUT.txt      -> prints how many lines changed (0: already canonical)
"""
from __future__ import annotations

import re, sys

HEADING = re.compile(r"^((final|last|pre|post)[- ]?)?(verse|chorus|bridge|outro|intro|pre-chorus|hook|refrain|break|"
                     r"middle 8|interlude)\b[\w\s\-]*(\(.*\))?\s*$", re.I)


def normalise(text: str) -> tuple[str, int]:
    out, n = [], 0
    for raw in text.split("\n"):
        t = raw.strip()
        if t and not t.startswith("[") and (HEADING.match(t) or re.match(r"^\(.*\)$", t)):
            out.append(f"[{t}]")
            n += 1
        else:
            out.append(raw.rstrip())
    return "\n".join(out), n


if __name__ == "__main__":
    txt, n = normalise(open(sys.argv[1]).read())
    open(sys.argv[2], "w").write(txt)
    print(n)
