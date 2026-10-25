"""Step dependencies for the pipeline, like a Makefile but by content: a step runs again when any of its inputs (files,
folders, settings) has changed since it was built, not when a file merely exists. Added 2026-10-25, after four renders
silently reused yesterday's shot prompts: every step skipped when its output existed, so a new storyboard
(lyrics-cued.txt) never reached the stills. Timestamps (make) would not have caught it either: rsync -a keeps the
source's mtimes and a restored backup is older than what it replaces.

Each step keeps a record in a .deps/ folder beside its outputs: STEP.done (the inputs it was built from, by hash) and,
while it runs, STEP.started. Used from bash through bin/deps.sh (need / made):

    python3 pipeline/deps.py check DIR STEP TARGET MODE -- INPUT...     exit 0 up to date, 10 run fresh, 11 resume
    python3 pipeline/deps.py start DIR STEP -- INPUT...                 the inputs are recorded now ...
    python3 pipeline/deps.py done DIR STEP                              ... and become STEP.done when it finishes
    python3 pipeline/deps.py accept DIR STEP...                         keep the current outputs once (see below)
An INPUT is a file or folder (hashed by content; a folder of more than 500 files by names and sizes) or a setting
NAME=VALUE (NAME in capitals). TARGET is the file that exists once the step is complete. MODE: "fresh" (a stale step's
outputs are discarded), "resume" (it can pick up an unfinished run with the same inputs: keyframes, video, lip-sync) or
"manual" (a step whose outputs people edit, such as the workshops: never re-run by itself, only reported).
check: up to date when STEP.done matches the inputs and TARGET exists; resume when STEP.started matches (an
interrupted run); otherwise stale, with the inputs that changed printed. accept: the step's outputs were edited on
purpose (scenes.py apply, a hand fix, a piece made before this file existed); its next check keeps them: a resumable
step resumes to fill gaps, any other is recorded as built from the current inputs.
Environment: DEPS_REDO / DEPS_ACCEPT (comma-separated steps, from ./poptok --redo / --accept); DEPS_PENDING (a file:
in a dry run (--explain) the outputs of steps that would run, so the steps after them are reported as stale too).
"""
from __future__ import annotations

import hashlib, json, os, re, sys, time

SETTING = re.compile(r"^[A-Z][A-Z0-9_]*=")


def file_hash(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def dir_hash(p: str) -> str:
    files = sorted(os.path.relpath(os.path.join(r, n), p) for r, _, ns in os.walk(p) for n in ns)
    h = hashlib.sha256()
    big = len(files) > 500                            # thousands of video frames: names and sizes
    for f in files:
        fp = os.path.join(p, f)
        h.update(f"{f}\t{os.path.getsize(fp)}\t{'' if big else file_hash(fp)}\n".encode())
    return h.hexdigest()[:16]


def hashes(items: list[str]) -> dict[str, str]:
    out = {}
    for x in items:
        if SETTING.match(x):
            k, v = x.split("=", 1)
            out[k] = hashlib.sha256(v.encode()).hexdigest()[:16]
        elif os.path.isdir(x):
            out[x] = dir_hash(x)
        elif os.path.exists(x):
            out[x] = file_hash(x)
        else:
            out[x] = "absent"
    return out


def rec(d: str, step: str, kind: str) -> dict | None:
    p = os.path.join(d, ".deps", f"{step}.{kind}")
    return json.load(open(p)) if os.path.exists(p) else None


def write(d: str, step: str, kind: str, inputs: dict) -> None:
    os.makedirs(os.path.join(d, ".deps"), exist_ok=True)
    json.dump({"inputs": inputs, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(d, ".deps", f"{step}.{kind}"), "w"), indent=1)


def drop(d: str, step: str, kind: str) -> None:
    p = os.path.join(d, ".deps", f"{step}.{kind}")
    if os.path.exists(p):
        os.remove(p)


def changed(old: dict, new: dict) -> str:
    diff = [k for k in new if old.get(k) != new[k]] + [k for k in old if k not in new]
    names = [os.path.basename(k) if "/" in k else k for k in diff]
    return ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")


def listed(var: str, step: str) -> bool:
    return step in os.environ.get(var, "").replace(" ", "").split(",")


def check(d: str, step: str, target: str, mode: str, items: list[str]) -> int:
    pend_f = os.environ.get("DEPS_PENDING")
    pending = set(open(pend_f).read().split("\n")) - {""} if pend_f and os.path.exists(pend_f) else set()
    now = hashes(items)
    done, started = rec(d, step, "done"), rec(d, step, "started")
    if listed("DEPS_ACCEPT", step) and os.path.exists(target) or os.path.exists(os.path.join(d, ".deps", f"{step}.accept")):
        if mode == "resume":
            print("existing outputs accepted; filling any gaps")
            return 11
        if not pend_f:
            write(d, step, "done", now)
            drop(d, step, "accept")
        print("existing outputs accepted as built from the current inputs")
        return 0
    up = [x for x in items if os.path.normpath(x) in pending]
    if listed("DEPS_REDO", step):
        why = "redo asked for"
    elif up:
        why = f"after {', '.join(os.path.basename(x) for x in up[:3])} is rebuilt"
    elif not os.path.exists(target) and done is None and started is None:
        why = "not built yet"
    elif done and done["inputs"] == now and os.path.exists(target):
        print("up to date")
        return 0
    elif mode == "resume" and started and started["inputs"] == now:
        print("an unfinished run with the same inputs")
        return 11
    elif done or started:
        why = "inputs changed: " + changed((done or started)["inputs"], now) if (done or started)["inputs"] != now else \
              f"{os.path.basename(target)} is missing"
    else:
        why = "no record of what it was built from (made before dependency tracking; --accept STEP keeps it)"
    print(why)
    return 10


def accept(d: str, *steps: str) -> None:
    """For tools that edit a step's outputs on purpose (scenes.py apply, fix_keys.py --in-place): keep them next run."""
    os.makedirs(os.path.join(d, ".deps"), exist_ok=True)
    for s in steps:
        open(os.path.join(d, ".deps", f"{s}.accept"), "w").close()


def main():
    a = sys.argv[1:]
    cmd = a.pop(0)
    items = a[a.index("--") + 1:] if "--" in a else []
    pos = a[:a.index("--")] if "--" in a else a
    if cmd == "check":
        d, step, target, mode = pos
        sys.exit(check(d, step, target, mode, items))
    if cmd == "start":
        d, step = pos
        write(d, step, "started", hashes(items))
    elif cmd == "done":
        d, step = pos
        s = rec(d, step, "started")
        if s is None:
            sys.exit(f"deps: {step} has no started record in {d}")
        write(d, step, "done", s["inputs"])
        drop(d, step, "started")
        drop(d, step, "accept")
    elif cmd == "accept":
        accept(pos[0], *pos[1:])
    else:
        sys.exit(f"deps: unknown command {cmd}")


if __name__ == "__main__":
    main()
