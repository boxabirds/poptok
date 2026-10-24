"""Lyric-intelligibility loop for ACE-Step (the owner's spec):
    generate music -> stem extraction -> transcribe -> compare with the true lyrics ->
    simple phonetic spelling VARIATIONS for the words that fail -> repeat until it matches.

    python3 pipeline/lyric_loop.py CAPTION.txt LYRICS.txt OUT [--bpm 0] [--llm qwen38-27b] [--variants-per-word 3] [--seed 2]

Per word (as written in the true lyrics, e.g. "LiteLLM"):
  - heard in the round-0 song            -> never touched
  - missed, and the LLM says it is a name / brand / acronym / number / unusual word -> it gets N simple spelling
    variants; each round sings the word's current variant; the first variant that is heard is KEPT (locked); a
    variant that is not heard is replaced by the next one; when a word's variants run out the LLM is asked for
    fresh ones (up to --max-asks), after that the word reverts to its original spelling and is marked "gave up"
  - missed, but an ordinary English word (the LLM returns no variants) -> left alone
Stops when every respellable word is either heard or given up ("until it matches"), with --max-rounds as a safety cap.
Scoring is always against the TRUE lyrics: a true word counts as heard if Whisper wrote it or the variant it was sung as.
Caveat: changing any spelling changes the whole performance (same seed, different text), so other words can flip too.

LLM: any OpenAI-compatible endpoint. --llm picks a preset (local server started for the step and stopped after,
so the GPU runs one job at a time) or pass --llm-url/--llm-model/--llm-cmd yourself.
Each round: OUT/rNN/ lyrics.txt, song wav, vocals.wav, heard.json, score.json; OUT/state.json is the word table.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, signal, subprocess, time, urllib.request
from difflib import SequenceMatcher
from pathlib import Path

PRESETS = {  # name: (url, model, command that starts the server, or "" if it is already running)
    "qwen38-27b": ("http://127.0.0.1:8091", "qwen38-27b",
                   "~/.local/bin/llama-server -m ~/.local/share/qwen38-27b/models/Qwen3.8-27B-GGUF/"
                   "Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -c 16384 --reasoning-budget 0 --host 127.0.0.1 --port 8091"),
    "gemma4-31b-thing1": ("http://127.0.0.1:11435", "gemma4-31b", ""),   # Ollama on thing1 via an ssh tunnel
}
ap = argparse.ArgumentParser()
ap.add_argument("caption")
ap.add_argument("lyrics")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--variant", default="xl-sft")
ap.add_argument("--variants-per-word", type=int, default=3)
ap.add_argument("--max-asks", type=int, default=2, help="times the LLM is asked for variants for one word")
ap.add_argument("--max-rounds", type=int, default=20)
ap.add_argument("--bpm", type=int, default=0)
ap.add_argument("--duration", type=float, default=-1)
ap.add_argument("--llm", default="qwen38-27b", choices=sorted(PRESETS))
ap.add_argument("--llm-url"); ap.add_argument("--llm-model"); ap.add_argument("--llm-cmd")
a = ap.parse_args()
url, model, cmd = PRESETS[a.llm]
url, model, cmd = a.llm_url or url, a.llm_model or model, a.llm_cmd if a.llm_cmd is not None else cmd
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
truth = open(a.lyrics).read()
(OUT / "caption.txt").write_text(open(a.caption).read())
UID, HOME = f"{os.getuid()}:{os.getgid()}", os.path.expanduser("~")
LOG = open(OUT / "loop.log", "a")


def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def gpu_free():
    """one GPU job per box: wait while another of our GPU containers runs here"""
    while subprocess.run("docker ps --format '{{.Image}}' | grep -qE '^(acestep|m2v-pre|zoommorph|music2vid|ltx2)'",
                         shell=True).returncode == 0:
        time.sleep(20)


def toks(s):
    return re.findall(r"[a-z0-9']+", re.sub(r"\[[^\]]*\]", " ", s.lower().replace("’", "'")))


def lyric_lines():
    return [l for l in (re.sub(r"\[[^\]]*\]", "", x).strip() for x in truth.splitlines()) if l]


# true words as written (keep the original form, e.g. "LiteLLM", "CERT-EU", "Archive-dot-org") with their lines
WORDS = []                                                   # (line index, original word)
for li, line in enumerate(lyric_lines()):
    for w in re.findall(r"[A-Za-z0-9][\w'’.-]*[\w]|[A-Za-z0-9]", line):
        WORDS.append((li, w))


NUM = set("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
          "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million "
          "billion point".split())


def numeric(w):
    """a number written in words or digits ("One-point-five", "1.5"): Whisper writes these as digits, so a miss is
    formatting, not pronunciation"""
    t = toks(w)
    return bool(t) and all(x in NUM or x.isdigit() for x in t)


def sing(m):
    """the lyrics as sung: true lyrics with each respelled word replaced (whole word, case-sensitive)"""
    out = truth
    for w in sorted(m, key=len, reverse=True):
        out = re.sub(rf"(?<![\w-]){re.escape(w)}(?![\w-])", m[w], out)
    return out


def heard_words(heard, m):
    """for each original word: True if every token of it (or of the variant it was sung as) was heard in order"""
    hyp = toks(heard)
    seq, owner = [], []                                      # token stream of the TRUE lyrics, token -> word index
    for i, (_, w) in enumerate(WORDS):
        for t in toks(w):
            seq.append(t); owner.append(i)
    hit_tok = [False] * len(seq)
    for b in SequenceMatcher(None, seq, hyp, autojunk=False).get_matching_blocks():
        for k in range(b.size):
            hit_tok[b.a + k] = True
    ok = [True] * len(WORDS)
    for t, i in zip(hit_tok, owner):
        ok[i] = ok[i] and t
    hyp_s = " " + " ".join(hyp) + " "
    for i, (_, w) in enumerate(WORDS):                       # sung as a variant: accept the variant's tokens
        if not ok[i] and w in m and (" " + " ".join(toks(m[w])) + " ") in hyp_s:
            ok[i] = True
    per = {}
    for i, (_, w) in enumerate(WORDS):                       # a word that occurs several times: majority heard
        per.setdefault(w, []).append(ok[i])
    return sum(ok) / len(ok), {w: sum(v) * 2 >= len(v) for w, v in per.items()}


def llm(prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.4,
                       "max_tokens": 1500, "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])


def ask_variants(words, tried):
    """{word: [variants]} for the words that are names/brands/acronyms/numbers/unusual; [] for ordinary words"""
    lines = lyric_lines()
    ctx = "\n".join(f"- {w}  (in: \"{lines[next(li for li, x in WORDS if x == w)]}\")"
                    + (f"  already tried, not understood: {tried[w]}" if tried.get(w) else "") for w in words)
    prompt = (
        "An AI singer performs these rap lyrics, and listeners misheard the words listed below. For each word, if it "
        "is a brand, product or company name, an acronym, a number written oddly, or another unusual word that a "
        f"singer could mispronounce, give {a.variants_per_word} SIMPLE phonetic spellings that a singer reading them "
        "would say correctly. Simple means: plain English syllables a child could read, separated by spaces or "
        "hyphens; acronyms spelled out as letter names (\"L-L-M\" or \"ell ell em\"); keep the real stress and the "
        "same number of syllables. First work out how the word is REALLY pronounced (e.g. TiDB is \"tie-D-B\", "
        "SSH is \"ess ess aitch\", WordPress is two ordinary words \"word press\", Zeabur is \"zay-burr\"), then write "
        "spellings of THAT pronunciation. The variants must be DIFFERENT from each other (different splits, "
        "different letters, e.g. \"lite ell ell em\" / \"light L L M\" / \"lyte el-el-em\") and from anything "
        "already tried. If the word is an ordinary English word, return an empty list for it.\n\n"
        f"Words:\n{ctx}\n\nReturn JSON only: {{\"variants\": {{\"word\": [\"...\", ...], ...}}}}")
    r = llm(prompt).get("variants") or {}
    out = {}
    for w in words:
        seen, vs = {w.lower(), *(t.lower() for t in tried.get(w, []))}, []
        for v in r.get(w) or []:
            if isinstance(v, str) and v.strip() and " ".join(toks(v)) not in {" ".join(toks(x)) for x in seen}:
                seen.add(v.strip()); vs.append(v.strip())
        out[w] = vs[:a.variants_per_word]
    return out


def with_llm(fn, *x):
    gpu_free()
    p = None
    if cmd:
        p = subprocess.Popen(os.path.expanduser(cmd), shell=True, stdout=open(OUT / "llm_server.log", "a"),
                             stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(240):
            try:
                urllib.request.urlopen(f"{url}/health", timeout=2); break
            except Exception:
                time.sleep(2)
        else:
            raise SystemExit("LLM server did not come up; see llm_server.log")
    try:
        return fn(*x)
    finally:
        if p:
            os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=60)


def render(rd, lyrics):
    rd.mkdir(exist_ok=True)
    (rd / "lyrics.txt").write_text(lyrics)
    song = rd / f"song_{a.variant}_s{a.seed}.wav"
    if not song.exists():
        gpu_free()
        subprocess.run(["docker", "run", "--gpus", "all", "--rm", "--user", UID, "--ipc=host", "-e", "HOME=/tmp",
                        "-e", f"USER={os.environ.get('USER', 'user')}", "-e", f"LOGNAME={os.environ.get('USER', 'user')}", "-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor",
                        "-e", "HF_HOME=/ckpt/hf", "-e", "ACESTEP_CHECKPOINTS_DIR=/ckpt", "-e", "PYTHONPATH=/app",
                        "-v", f"{HOME}/models/acestep:/ckpt", "-v", f"{ROOT}:/work", "-v", f"{OUT}:/loop", "-w", "/app",
                        "--entrypoint", "/app/.venv/bin/python", "acestep", "/work/pipeline/ace_gen.py",
                        f"/loop/{rd.name}", "--caption-file", "/loop/caption.txt", "--bpm", str(a.bpm), "--duration", str(a.duration), "--variants", a.variant, "--seeds", str(a.seed),
                        "--lyrics-file", f"/loop/{rd.name}/lyrics.txt", "--name", "song"],
                       check=True, stdout=open(rd / "ace.log", "w"), stderr=subprocess.STDOUT)
    if not (rd / "heard.json").exists():
        gpu_free()
        subprocess.run(["docker", "run", "--gpus", "all", "--rm", "--user", UID, "-e", "HOME=/tmp", "-e", "TORCH_HOME=/cache/torch",
                        "-v", f"{ROOT}:/work", "-v", f"{ROOT}/cache:/cache", "-v", f"{OUT}:/loop", "-w", "/work", "m2v-pre",
                        "python3", "pipeline/vocals_transcribe.py", f"/loop/{rd.name}/{song.name}", f"/loop/{rd.name}"],
                       check=True, stdout=open(rd / "transcribe.log", "w"), stderr=subprocess.STDOUT)
    return json.load(open(rd / "heard.json"))["text"]


# state: word -> {"status": candidate|trying|kept|gave_up|ordinary, "queue": [...], "tried": [...], "asks": n, "cur": str}
sp = OUT / "state.json"
state = json.load(open(sp)) if sp.exists() else {}
best = (-1.0, None)
for rnd in range(a.max_rounds):
    m = {w: s["cur"] for w, s in state.items() if s.get("cur")}
    rd = OUT / f"r{rnd:02d}"
    heard = render(rd, sing(m))
    frac, ok = heard_words(heard, m)
    for w, s in state.items():                               # judge the variants sung this round
        if s["status"] == "trying":
            if ok.get(w):
                s["status"] = "kept"
            else:
                s["tried"].append(s["cur"]); s["cur"] = None
    if rnd == 0:                                             # who fails with the true spelling
        for w, good in ok.items():
            if not good and not numeric(w):
                state[w] = {"status": "candidate", "queue": [], "tried": [], "asks": 0, "cur": None}
    need = [w for w, s in state.items() if s["status"] in ("candidate", "trying") and not s["queue"] and not s["cur"]
            and s["asks"] < a.max_asks]
    if need:
        got = with_llm(ask_variants, need, {w: state[w]["tried"] for w in need})
        for w in need:
            state[w]["asks"] += 1
            state[w]["queue"] = got.get(w, [])
            if not state[w]["queue"]:
                state[w]["status"] = "ordinary" if state[w]["asks"] == 1 and not state[w]["tried"] else "gave_up"
    for w, s in state.items():                               # next variant for every word still failing
        if s["status"] in ("candidate", "trying") and not s["cur"]:
            if s["queue"]:
                s["cur"], s["queue"] = s["queue"][0], s["queue"][1:]; s["status"] = "trying"
            elif s["asks"] >= a.max_asks:
                s["status"] = "gave_up"
    json.dump(state, open(sp, "w"), indent=1)
    json.dump({"score": frac, "missed": sorted(w for w, g in ok.items() if not g)}, open(rd / "score.json", "w"), indent=1)
    if frac > best[0]:
        best = (frac, rd.name)
    tally = {k: sum(1 for s in state.values() if s["status"] == k) for k in ("kept", "trying", "gave_up", "ordinary")}
    say(f"round {rnd}: {frac * 100:.1f}% of true words heard | {tally} | next: "
        + ", ".join(f"{w}->{s['cur']}" for w, s in state.items() if s["status"] == "trying"))
    if not any(s["status"] == "trying" for s in state.values()):
        say("every respellable word is either heard or given up: done"); break
say(f"best: {best[1]} at {best[0] * 100:.1f}%; kept spellings: "
    + json.dumps({w: s['cur'] for w, s in state.items() if s['status'] == 'kept'}))
shutil.copy(OUT / best[1] / f"song_{a.variant}_s{a.seed}.wav", OUT / "best.wav")
