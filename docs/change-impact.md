# Change impact: what a piece of feedback costs

The pipeline is a chain of steps; each step's output feeds the next. A change at one step invalidates everything
downstream of it and nothing upstream. This page lays the steps out end to end, says what each depends on, what it
costs to redo, and so what any given piece of feedback costs. Keep it current (see CLAUDE.md): whenever a step,
its inputs, its outputs or its cost changes, update this page.

Times are for a ~4 min song (~80–100 bars, ~40 scenes, ~6000 video frames, 4 choruses) on the current farm
([hardware-performance.md](hardware-performance.md)).

## The pipeline, end to end

```
inputs ──► 1 analysis ──► 2 lyrics aligned ──► 3 prompts + cues ──► 4 stills ──► 5 merge scenes ──► 6 scenes + transitions
 song        stems,          lyric lines on        one prompt per      one per      consecutive        scenes.json: per-scene
 lyrics      Whisper,        the song's clock      bar (LLM), who      bar (MOSS    near-duplicate     prompt, motion, A→B
 treatment   beat grid                             is in it (cues)     / Amber      bars → one         transition (LLM)
 cues                                                                  Qwen-Edit)   scene
 photo                                                                     │
                                                                           │ (sung scenes' stills)
            ┌──────────────────────────────────────────────────────────────┴───────────────┐
            ▼                                                                              ▼
 7 LTX chain (video) ──► 8 finish ──────────────────────────────────────────────────► 10 splice ──► NAME.mp4
   one continuous         retime to beats,                         9 lip-sync inserts ──┘
   chain of clips,        grade, deflare,                            one InfiniteTalk clip per
   each starting from     pulse (off), mux                           run of sung scenes, from that
   the previous frame                                                scene's still + the vocal stem
```

| # | Step | Depends on | Output | Cost to redo | Where |
|---|---|---|---|---|---|
| 0 | Directors' workshop (optional, `WORKSHOP=1`): conversation → treatment + line-by-line storyboard → plausibility review | lyrics, `inputs/brief.txt`, performer photo or not | `inputs/workshop.md`, `visual-treatment.txt`, `lyrics-cued.txt` | ~5–8 min (LLM) | gruntus |
| 1 | Analysis: stems, Whisper, beat grid | song | `analysis/` | ~5 min | gruntus |
| 2 | Lyrics aligned | lyrics, 1 | `*-true-segments.json` | ~1 min | gruntus |
| 3 | Prompts + actor cues (LLM) | 2, treatment / style word, cues, performer | `shots.json` (prompt + cast per bar) | ~5 min | gruntus |
| 4 | Stills | 3, performer photo, seeds (`reroll.json`) | `keys/shot_NNN.png` | 24 s (Amber) – 45 s (MOSS) per still on a Spark; whole song ~45–60 min over 3 boxes | Sparks, tritus |
| 5 | Merge near-duplicate scenes (LLM) | 3, 4 | `shots_merged.json` (scene boundaries = the clip plan) | ~2 min | gruntus |
| 6 | Scenes + transitions (LLM) | 5 | `scenes.json` | ~3 min | gruntus |
| 7 | LTX chain | 4, 5, 6 | `ltx_chain/*.png` | ~1 s/frame on a Spark: ~100 min whole song | Spark or quintus |
| 8 | Finish: retime, grade, deflare, pulse, the piece's effects (`--fx`, pipeline/fx/), mux | 7, 1, finish settings, `inputs/fx.json` | `NAME-nosync.mp4` | ~6 min | gruntus (CPU) |
| 9 | Lip-sync inserts | the sung scenes' stills (4), the vocal stem and timing (1), which scenes sing (3/6) | `sing/it_out_K.mp4` | ~11 min per chorus on gruntus, ~28 on a Spark | gruntus, Sparks |
| 10 | Splice | 8, 9 | `NAME.mp4` | seconds | gruntus |

**The chain rule (step 7).** Each clip starts from the previous clip's last frame, so a clip's look depends on every
clip before it. Re-rendering one clip in the middle and keeping the old clips after it leaves a jump where they meet
(oil Chronophobia v2: 9 of 39 joins jumped). So a change to any clip means **re-rendering from that clip to the end**
for a jump-free video. The cost of a change at scene *k* is therefore proportional to how far from the end it is:
a change in the last scene costs minutes, one in the first scene costs the whole ~100 min. (`scenes.py apply` can
re-render only the affected clips, which is fast but risks those jumps; use it for previews, not finals.)

**Lip-sync is off to the side.** Steps 9–10 depend on the stills and the audio, not on the video chain, so they can be
changed, re-rendered or swapped to another engine without touching steps 1–8, and they can run in parallel with the
chain.

## What a piece of feedback costs

| Feedback / change | Redo from | Reused | Cost (4 min song) | Examples |
|---|---|---|---|---|
| Grade, deflare, pulse on/off, retime settings, the piece's effects (fx.json: which events, light, shake) | 8 | everything else | ~6 min, CPU | Pulse removed from all Chronophobia renditions (2026-10-24) |
| Lip-sync engine or settings; a chorus redone | 9 (those choruses), 10 | 1–8 | 11–28 min per chorus | gruntus int8 vs Spark bf16 |
| Which lines are lip-synced | 9, 10 (+ the chain if those scenes' stills change) | 1–8 | per chorus | |
| One scene's still (a re-roll or a new prompt) | 4 for that still, then 7 from that scene to the end, 8, 10 (9 if it's a sung scene) | 1–3, 5, 6, all other stills, the chain before that scene | seconds for the still + up to ~100 min of chain (less the later the scene) | Slippery Road s023 three hands |
| One transition's wording | 7 from that change to the end, 8, 10 | 1–6, all stills, the chain before it | up to ~100 min | |
| All transitions' wording (e.g. "describe an event") | 6, 7 from the first change, 8, 10 | 1–5, all stills, lip-sync | ~3 min LLM + ~100 min chain | Oil v2/v3 |
| Motion within scenes | 7 from the first changed scene, 8, 10 | 1–6 | up to ~100 min | |
| Scene boundaries (merge) | 5, 6, 7 (all), 8, 10 | 1–4, stills, lip-sync | ~105 min | |
| Actor cues (who appears where) | 3 for those lines, 4 for those stills, then as "one scene's still" | analysis, other stills | stills + chain from the first change | |
| Re-run the workshop (new brief, new directors' take) | 0, then as "treatment" below | 1, 2 | ~8 min + ~3 h |  |
| Treatment or style word | 3 onward: all stills, all chain, lip-sync stills too | 1, 2 | ~3 h (stills ~1 h over 3 boxes, chain ~1.7 h, lip-sync ~1 h in parallel) | Chronophobia oil vs photographic |
| Lyrics text | 2 onward | 1 | ~3 h | |
| Song | everything | nothing | ~3.5 h | |
| Performer photo | Amber stills (4), then the chain from her first scene, 9 for all choruses | MOSS stills, 1–3, 5, 6 | ~2.5 h | |

**Rules of thumb**

- The later in the pipeline a change lands, the cheaper it is. Finishing and lip-sync are cheap; anything that changes
  a still or a transition costs chain time from that point to the end.
- Batch chain changes: collect all the scene and transition feedback for a video, then re-render the chain once from the
  earliest change.
- Feedback near the end of a song is cheap; feedback in the first verse costs a full chain.
- Fix problems at the still (step 4) before the chain is rendered: a re-rolled still costs seconds then, and up to
  ~100 min of chain afterwards. That is what the still check (`check_keys.py`) and any future plausibility check are for.
