"""Forced alignment of the KNOWN lyrics to the vocal stem (torchaudio's MMS forced aligner, wav2vec2 CTC): instead of
transcribing and matching (align_lyrics.py, which fails where Whisper hears nothing, e.g. stacked chorus voices), it
finds when each known word is sung. Writes the same format as align_lyrics.py.

    python pipeline/force_align.py SONG.wav LYRICS.txt --analysis DIR [--out NAME]   -> DIR/<stem>-fa-segments.json
Runs in the m2v-pre image (torchaudio; the model downloads to TORCH_HOME once). Per line: start of its first word,
end of its last word, "score" = mean token confidence (0-1; low = unsure).
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from common import analysis_dir

ap = argparse.ArgumentParser()
ap.add_argument("song")
ap.add_argument("lyrics")
ap.add_argument("--analysis")
ap.add_argument("--out", help="output file name (default <stem>-fa-segments.json)")
ap.add_argument("--merge", metavar="TAG", help="also make these the pipeline's lyric timings: <stem><TAG>-segments.json "
                "(align_lyrics.py's Whisper-matched version is kept as <stem><TAG>-whisper-segments.json)")
a = ap.parse_args()
song = Path(a.song)
ana = analysis_dir(song, a.analysis)
lines = [l for l in (re.sub(r"\{[^}]*\}", "", x).strip() for x in open(a.lyrics)) if l and not re.match(r"^\[.*\]$", l)]

bundle = torchaudio.pipelines.MMS_FA
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = bundle.get_model(with_star=False).to(dev)
tokenizer, aligner = bundle.get_tokenizer(), bundle.get_aligner()
wav, sr = torchaudio.load(str(ana / f"{song.stem}-vocals.mp3"))
wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, bundle.sample_rate)


def norm(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower().replace("’", "'"))


words, owner = [], []
for i, l in enumerate(lines):
    for w in l.split():
        n = norm(w)
        if n:
            words.append(n); owner.append(i)

with torch.inference_mode():
    em = []                                               # emissions in 30 s chunks (memory), concatenated
    step = bundle.sample_rate * 30
    for s in range(0, wav.shape[1], step):
        e, _ = model(wav[:, s:s + step].to(dev))
        em.append(e)
    emission = torch.cat(em, dim=1)
    spans = aligner(emission[0], tokenizer(words))
ratio = wav.shape[1] / emission.shape[1] / bundle.sample_rate  # seconds per emission frame
seg = []
for i, l in enumerate(lines):
    ws = [k for k, o in enumerate(owner) if o == i]
    first, last = spans[ws[0]], spans[ws[-1]]
    sc = [t.score for k in ws for t in spans[k]]
    seg.append({"start": round(first[0].start * ratio, 2), "end": round(last[-1].end * ratio, 2), "text": l,
                "score": round(sum(sc) / len(sc), 3)})
out = ana / (a.out or f"{song.stem}-fa-segments.json")
json.dump({"source": str(a.lyrics), "method": "MMS forced alignment", "segments": seg}, open(out, "w"), indent=1)
if a.merge:
    tp = ana / f"{song.stem}{a.merge}-segments.json"
    wp = ana / f"{song.stem}{a.merge}-whisper-segments.json"
    if tp.exists() and not wp.exists():
        tp.rename(wp)
    wh = json.load(open(wp))["segments"] if wp.exists() else []
    merged = [dict(s, matched=max(s["score"], w.get("matched", 0)) if i < len(wh) else s["score"])
              for i, (s, w) in enumerate(zip(seg, wh + [{}] * len(seg)))]
    json.dump({"source": str(a.lyrics), "method": "MMS forced alignment (Whisper-matched kept in "
               f"{wp.name})", "segments": merged}, open(tp, "w"), indent=1)
    print(f"lyric timings now from forced alignment -> {tp}")
for s in seg:
    print(f"  {s['start']:6.2f}-{s['end']:6.2f}  score={s['score']:.2f}  {s['text'][:60]}")
print(f"{len(seg)} lines force-aligned -> {out}")
