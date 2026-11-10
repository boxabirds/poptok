"""Do takes made from the same plan (ace_gen.py --codes-file) keep the music? Compares each take with a base take.

    python pipeline/compare_takes.py BASE.wav TAKE.wav [TAKE.wav ...] --lyrics LYRICS.txt --out OUT.json   (m2v-pre image)
Per take:
  length       seconds
  lyric match  share of lyric words Whisper hears in order (full mix, as ace_score.py)
  harmony      mean frame-by-frame correlation of the chroma (which notes sound when) with the base, 0-1 (same
               chords at the same moments = high; an unrelated song of the same key scores ~0.3-0.5)
  rhythm       correlation of the onset strength (when notes start) with the base
  lines        the first time of each sung line Whisper hears, to compare where the singing falls
"""
from __future__ import annotations

import argparse, json, re
from difflib import SequenceMatcher

import librosa, numpy as np, whisper

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("takes", nargs="+")
ap.add_argument("--lyrics", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
norm = lambda s: re.findall(r"[a-z0-9']+", re.sub(r"\[[^\]]*\]", " ", s.lower().replace("’", "'")))
ref = norm(open(a.lyrics).read())
model = whisper.load_model("medium", download_root="/cache/whisper")
SR, HOP = 22050, 2205                                         # 0.1 s frames


def feats(f):
    y, _ = librosa.load(f, sr=SR, mono=True)
    return y, librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP), librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)


def corr_frames(x, y):
    n = min(x.shape[1], y.shape[1])
    x, y = x[:, :n] - x[:, :n].mean(0), y[:, :n] - y[:, :n].mean(0)
    c = (x * y).sum(0) / (np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0) + 1e-9)
    return float(np.mean(c))


by, bc, bo = feats(a.base)
rows = []
for f in [a.base] + a.takes:
    y, c, o = feats(f)
    r = model.transcribe(f, language="en", condition_on_previous_text=False)   # no hallucination loops (ace_score.py)
    hyp = norm(r["text"])
    hit = sum(b.size for b in SequenceMatcher(None, ref, hyp, autojunk=False).get_matching_blocks())
    n = min(len(o), len(bo))
    row = {"file": f.split("/")[-1], "dir": f.split("/")[-2], "length": round(len(y) / SR, 1),
           "lyric_match": round(hit / len(ref), 3), "harmony": round(corr_frames(c, bc), 3),
           "rhythm": round(float(np.corrcoef(o[:n], bo[:n])[0, 1]), 3),
           "lines": [[round(s["start"], 1), s["text"].strip()[:40]] for s in r["segments"]]}
    rows.append(row)
    print(f"{row['dir']}/{row['file']:22s} {row['length']:6.1f}s  lyric {row['lyric_match']*100:5.1f}%  "
          f"harmony {row['harmony']:.2f}  rhythm {row['rhythm']:.2f}", flush=True)
json.dump(rows, open(a.out, "w"), indent=1, ensure_ascii=False)
