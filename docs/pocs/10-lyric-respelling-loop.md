# POC 10: Respelling words the singer garbles (lyric loop)

**Date:** 2026-10-23 · **Status:** works in part; `pipeline/lyric_loop.py` · **Piece:** offline

**Hypothesis:** words ACE-Step garbles (product names, acronyms) become intelligible if the lyric sent to the model
is respelled phonetically, checked by a loop: generate, separate vocals, Whisper, compare, ask the LLM for a new
spelling.

**Test:** about 3 minutes per round on gruntus, at most 2 fresh spellings per word, then back to the original.

**Result:** 19 words flagged (numbers left alone). Good respellings: `TiDB → tie dee bee`, `Zeabur → zay burr`,
`LiteLLM → lite ell ell em`. Wrong: `Joomla → joo-ma`. Qwen 3.8 27B ignores "give 3 variants", so variants come
across rounds. Not measured: whether the final round beats the original take overall; Gemma 4 31B comparison not run.
