"""Stem extraction + transcription for the lyric loop. Runs in the m2v-pre image (Demucs + Whisper).

    python3 pipeline/vocals_transcribe.py SONG.wav OUT_DIR [--model medium]
Writes OUT_DIR/vocals.wav (Demucs htdemucs, two stems) and OUT_DIR/heard.json {"text": ..., "segments": [...]}.
"""
import argparse, json, os, shutil, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("song"); ap.add_argument("out"); ap.add_argument("--model", default="medium")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tmp = os.path.join(a.out, "demucs")
subprocess.run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-o", tmp, a.song], check=True)
stem = os.path.splitext(os.path.basename(a.song))[0]
voc = os.path.join(a.out, "vocals.wav")
shutil.move(os.path.join(tmp, "htdemucs", stem, "vocals.wav"), voc)
shutil.rmtree(tmp)
import whisper
m = whisper.load_model(a.model, download_root="/cache/whisper")
r = m.transcribe(voc, language="en")
json.dump({"text": r["text"], "segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in r["segments"]]},
          open(os.path.join(a.out, "heard.json"), "w"), indent=1)
print("heard", len(r["text"].split()), "words")
