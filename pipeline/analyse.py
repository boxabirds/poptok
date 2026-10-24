"""Song analysis: Demucs stems, tempo, Whisper transcript with word timestamps. Runs in the analysis image.
Adapted from music2vid's poc.py (github.com/boxabirds/music2vid), output names unchanged.

    python3 pipeline/analyse.py SONG.wav [--analysis DIR]
Writes DIR/<song>-{vocals,drums,bass,other}.mp3, metadata.json (tempo), <song>-medium.json (Whisper, with words),
<song>-medium-segments.json (lines with start/end). Skips if the transcript exists.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import analysis_dir

MODEL_SIZE = "medium"
ap = argparse.ArgumentParser()
ap.add_argument("song")
ap.add_argument("--analysis", default=None)
a = ap.parse_args()
song = Path(a.song); out = analysis_dir(song, a.analysis); out.mkdir(parents=True, exist_ok=True)
full = out / f"{song.stem}-{MODEL_SIZE}.json"
if full.exists():
    sys.exit(f"{full} exists, skipping")

import demucs.api, demucs.audio, librosa
import whisper_timestamped as whisper

stems = {s: out / f"{song.stem}-{s}.mp3" for s in ("vocals", "drums", "bass", "other")}
if not stems["vocals"].exists():
    sep = demucs.api.Separator()
    sep.load_model(model="mdx_extra")
    sep.load_audios_to_model(song)
    for _, sources in sep.separate_loaded_audio():
        for s, src in sources.items():
            demucs.audio.save_audio(src, stems[s], samplerate=sep._samplerate)
y, sr = librosa.load(stems["drums"])
tempo = librosa.feature.tempo(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr)
json.dump({"tempo": float(tempo[0])}, open(out / "metadata.json", "w"))
import torch
model = whisper.load_model(MODEL_SIZE, device="cuda" if torch.cuda.is_available() else "cpu",
                           download_root="/cache/whisper")                    # shared cache: no re-download per run   # music2vid used cpu
# no VAD: on a vocal stem the separation already removed the non-voice parts, and the speech-trained VAD cut sung
# choruses entirely (noble-hunt chorus 1 vanished with it, every line came back without it; 2026-10-24)
res = whisper.transcribe_timestamped(model, whisper.load_audio(str(stems["vocals"])), language="en", vad=False,
                                     beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
json.dump(res, open(full, "w"), indent=2, ensure_ascii=False)
json.dump({"segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in res["segments"]]},
          open(out / f"{song.stem}-{MODEL_SIZE}-segments.json", "w"), indent=2, ensure_ascii=False)
print("wrote", full)
