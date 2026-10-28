# poptok/lean

A small music-video pipeline (D37, 2026-10-28), forked in design from poptok after poptok's layers kept overriding
explicit briefs (cuckoo: "charcoal" came out as photographs of charcoal-coloured mud). Modelled on macvideo
(quintus ~/expts/macvideo) and music2vid, with local models only (MOSS-Image, LTX, InfiniteTalk, a local LLM).

    song + lyrics + brief
      1 timing        when each line is sung (poptok's alignment)
      2 directors     the storyboard: keyframes with time spans, the line(s) each covers, a beat, Amber none/seen/sings
                      (the directors decide one line per scene, several lines per scene, or several scenes per line)
      3 visualiser    per keyframe, music2vid's question asked of the directors' beat (never a raw lyric): the best
                      visual for this beat in the context of this song and story, the style first, word for word;
                      it adds visual detail only, never changes who, what or where
      4 render        MOSS keyframes, LTX clips (InfiniteTalk where she sings), cut to the spans, the song added

Nothing else until a real video shows it is needed. Proofs of concept first (D38): poc_medium.py (can MOSS draw
charcoal?), poc_sweep.py (Amber and lip-sync at 0-100% style), directors.py (the storyboard).
