"""Beat grid for a song: beats, downbeats (every 4th beat, phase = strongest onset sum), tempo -> ANALYSIS/beats.json
    python3 pipeline/beat_grid.py SONG.wav [ANALYSIS_DIR]"""
import json, os, sys
from pathlib import Path
import librosa, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import analysis_dir

song = Path(sys.argv[1]); ana = analysis_dir(song, sys.argv[2] if len(sys.argv) > 2 else None)
y, sr = librosa.load(song, sr=22050)
env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
tempo, bf = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=256, units="frames", tightness=100)
bt = librosa.frames_to_time(bf, sr=sr, hop_length=256)
strength = env[bf]
phase = int(np.argmax([strength[p::4].sum() for p in range(4)]))
down = bt[phase::4]
json.dump({"tempo": float(np.atleast_1d(tempo)[0]), "beats": bt.round(4).tolist(), "downbeats": down.round(4).tolist(),
           "downbeat_phase": phase}, open(ana / "beats.json", "w"), indent=1)
print(f"tempo {float(np.atleast_1d(tempo)[0]):.1f} BPM, {len(bt)} beats, {len(down)} downbeats, first beat {bt[0]:.2f}s")
