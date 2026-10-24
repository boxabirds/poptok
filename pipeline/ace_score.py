import glob, json, re, sys, whisper
from difflib import SequenceMatcher
norm = lambda s: re.findall(r"[a-z0-9']+", re.sub(r"\[[^\]]*\]", " ", s.lower().replace("’", "'")))
# python3 pipeline/ace_score.py DIR LYRICS.txt: Whisper each DIR/*.wav, in-order match with the lyrics -> *.score.json
ref = norm(open(sys.argv[2]).read())
m = whisper.load_model("medium", download_root="/cache/whisper")
for f in sorted(glob.glob(sys.argv[1] + "/*.wav")):
    hyp = norm(m.transcribe(f, language="en")["text"])
    sm = SequenceMatcher(None, ref, hyp, autojunk=False)
    hit = sum(b.size for b in sm.get_matching_blocks())
    json.dump({"file": f, "match": hit / len(ref), "heard": len(hyp)}, open(f[:-4] + ".score.json", "w"))
    print(f"{f.split('/')[-1]:28s} words heard {len(hyp):4d} / lyric words {len(ref)}  in-order lyric match {hit/len(ref)*100:5.1f}%", flush=True)
