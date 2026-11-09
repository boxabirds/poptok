"""Which lyric timing is right where Whisper-matched (align_lyrics.py) and forced alignment (force_align.py) disagree?

    python pipeline/compare_lyric_timing.py OUT_DIR PIECE [PIECE ...]     (in the m2v-pre image, GPU)
For each piece, reads analysis/*-fa-segments.json and *-true-whisper-segments.json (same lines, same order). For every
line whose two start times differ by more than --tol seconds, it cuts the vocal stem to each version's window and asks
Whisper (fresh, no lyrics given, each clip on its own) what is sung there. Two checks per version:
  "inside"  share of the line's words heard in [start-0.3, end+0.3] s (does the window hold the line?)
  "starts"  are the line's first two words heard in [start-0.3, start+2.5] s (does the line begin where it says?)
Lines that agree within --tol count as agreed. A window can hold the line by being too long, so "starts" is the sharper
test. Bias: the referee is Whisper too, though it hears each clip without context, so it can't copy its own song-wide
pass. Writes OUT_DIR/lyric-timing.json and prints a table.
"""
from __future__ import annotations

import argparse, glob, json, re
from difflib import SequenceMatcher
from pathlib import Path

import torch
import whisper_timestamped as whisper

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("pieces", nargs="+")
ap.add_argument("--tol", type=float, default=0.3)
ap.add_argument("--whisper", default="-true-whisper-segments.json", help="suffix of the Whisper-matched file")
a = ap.parse_args()
SR = 16000
toks = lambda s: re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))
model = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu", download_root="/cache/whisper")


def heard(audio, t0: float, t1: float) -> list[str]:
    clip = audio[max(0, int(t0 * SR)):max(0, int(t1 * SR))]
    if len(clip) < SR * 0.3:
        return []
    r = model.transcribe(clip, language="en", temperature=0.0, condition_on_previous_text=False)
    return toks(r["text"])


def recall(line: list[str], got: list[str]) -> float:
    m = SequenceMatcher(None, line, got, autojunk=False)
    return round(sum(b.size for b in m.get_matching_blocks()) / max(len(line), 1), 2)


def starts(line: list[str], got: list[str]) -> bool:
    head = line[:2]
    return any(got[k:k + len(head)] == head for k in range(len(got))) if head else False


rows = []
for piece in a.pieces:
    ana = Path("pieces") / piece / "analysis"
    fa = Path(glob.glob(str(ana / "*-fa-segments.json"))[0])
    stem = fa.name.removesuffix("-fa-segments.json")
    wh = next(ana.glob(f"{stem}*{a.whisper}"))
    F, W = json.load(open(fa))["segments"], json.load(open(wh))["segments"]
    assert len(F) == len(W), (piece, len(F), len(W))
    audio = whisper.load_audio(str(ana / f"{stem}-vocals.mp3"))
    for i, (f, w) in enumerate(zip(F, W)):
        r = dict(piece=piece, line=i, text=f["text"], fa=[f["start"], f["end"]], fa_score=f["score"],
                 wh=[w["start"], w["end"]], wh_matched=w.get("matched"), diff=round(abs(f["start"] - w["start"]), 2))
        if r["diff"] > a.tol:
            lt = toks(f["text"])
            for k, (s, e) in (("fa", r["fa"]), ("wh", r["wh"])):
                r[f"{k}_inside"] = recall(lt, heard(audio, s - 0.3, e + 0.3))
                r[f"{k}_starts"] = starts(lt, heard(audio, s - 0.3, s + 2.5))
            fs, ws = r["fa_starts"], r["wh_starts"]
            r["winner"] = "both" if fs and ws else "fa" if fs else "wh" if ws else \
                "fa" if r["fa_inside"] > r["wh_inside"] + 0.2 else "wh" if r["wh_inside"] > r["fa_inside"] + 0.2 else "neither"
        else:
            r["winner"] = "agree"
        rows.append(r)
        if r["winner"] != "agree":
            print(f"{piece[:14]:14s} {i:2d} FA {r['fa'][0]:6.1f} in={r['fa_inside']:.2f} st={int(r['fa_starts'])} | "
                  f"WH {r['wh'][0]:6.1f} in={r['wh_inside']:.2f} st={int(r['wh_starts'])} -> {r['winner']:7s} {f['text'][:40]}",
                  flush=True)
Path(a.out).mkdir(parents=True, exist_ok=True)
json.dump(rows, open(Path(a.out) / "lyric-timing.json", "w"), indent=1, ensure_ascii=False)
for piece in a.pieces:
    R = [r for r in rows if r["piece"] == piece]
    c = {k: sum(r["winner"] == k for r in R) for k in ("agree", "both", "fa", "wh", "neither")}
    print(f"{piece:20s} lines={len(R):3d} " + " ".join(f"{k}={v}" for k, v in c.items()))
