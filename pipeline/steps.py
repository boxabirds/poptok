"""The pipeline's dependency graph, read from pipeline/steps.ini (one static file, like a Makefile's rules), for
bin/deps.sh's `need STEP` and for people.

    python3 pipeline/steps.py args STEP     the step's record folder, target, mode, outputs and inputs, expanded from
                                            the environment, NUL-separated (bin/deps.sh reads them)
    python3 pipeline/steps.py graph         every step, what it is built from and which step makes each input; checks
                                            that each input is made by an earlier step or is a source file
"""
from __future__ import annotations

import configparser, os, re, sys

INI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps.ini")
FIELDS = {"about", "when", "dir", "target", "outputs", "inputs", "settings", "mode"}


def load() -> dict[str, dict]:
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(INI)
    steps = {}
    for name in cp.sections():
        s = cp[name]
        bad = set(s) - FIELDS
        if bad or s.get("mode") not in ("fresh", "resume", "manual") or not s.get("dir") or not s.get("target"):
            sys.exit(f"steps.ini [{name}]: needs dir, target and mode fresh/resume/manual"
                     + (f"; unknown fields {sorted(bad)}" if bad else ""))
        steps[name] = {"about": " ".join(s.get("about", "").split()), "when": s.get("when", ""), "dir": s["dir"],
                       "target": s["target"], "mode": s["mode"], "outputs": s.get("outputs", "").split(),
                       "inputs": s.get("inputs", "").split(), "settings": s.get("settings", "").split()}
    return steps


def expand(t: str, step: str) -> str:
    def var(m):
        if m.group(1) not in os.environ:
            sys.exit(f"steps.ini [{step}]: {{{m.group(1)}}} is not set by the script that runs it")
        return os.environ[m.group(1)]
    return re.sub(r"\{([A-Z_]+)\}", var, t)


def args(name: str) -> list[str]:
    s = load().get(name) or sys.exit(f"steps.ini: no step [{name}]")
    ins = [x for x in (expand(t, name) for t in s["inputs"]) if x]
    for st in s["settings"]:
        k, _, default = st.partition("=")
        ins.append(f"{k}={os.environ.get(k) or default}")
    outs = "|".join(expand(t, name) for t in s["outputs"])
    return [expand(s["dir"], name), expand(s["target"], name), s["mode"], outs, *ins]


def graph() -> int:
    steps = load()
    made_by, order, problems = {}, list(steps), []
    for n, s in steps.items():
        for t in [s["target"], *s["outputs"]]:
            made_by.setdefault(t, n)
    for i, (n, s) in enumerate(steps.items()):
        print(f"{n}  ({s['mode']}{', when ' + s['when'] if s['when'] else ''})  {s['about']}")
        print(f"    makes  {s['target']}")
        for t in s["inputs"]:
            src = made_by.get(t)
            if src == n:
                problems.append(f"[{n}] lists its own output {t} as an input")
            elif src and order.index(src) > i:
                problems.append(f"[{n}] needs {t}, made by [{src}], which comes later in steps.ini")
            print(f"    <- {t}" + (f"   (from {src})" if src else "   (source)"))
        if s["settings"]:
            print(f"    <- settings {' '.join(s['settings'])}")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "graph"
    if cmd == "args":
        sys.stdout.write("\0".join(args(sys.argv[2])))
    elif cmd == "graph":
        sys.exit(graph())
    else:
        sys.exit(__doc__)
