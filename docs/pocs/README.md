# poptok proofs of concept

Each POC states a hypothesis, how it was tested, what we expected and what happened. The videos and stills they
mention live under `pieces/`, which git ignores, so they're on the build machine only.

**Reference method today:** C6 (POC 08): music2vid-style per-line prompts with the full lyrics as context, the
visual treatment as a style, MOSS stills, an LTX-2.5 chain with the LTX-2.3 Transition LoRA, scene changes on
downbeats, an occasional pulse on real bass hits.

| # | POC | Hypothesis in brief | Status |
|---|---|---|---|
| 00 | [Renderer search](00-renderer-search.md) | A modern image model in Deforum's per-frame loop gives stable morphs | Concluded: no; led to 03 |
| 01 | [Deforum baseline (A)](01-deforum-baseline.md) | A baseline to beat | Concluded |
| 02 | [Stills cut on the beat (B)](02-stills-and-cuts.md) | Stills plus cuts are enough | Dead end |
| 03 | [LTX Transition chain (C3)](03-ltx-transition-chain.md) | One chained shot, each scene transforming into the next | **Adopted** |
| 04 | [Downbeat retime + bass pulse (C3c)](04-beat-retime-and-pulse.md) | Changes on downbeats; pulse only on real low hits | **Adopted** |
| 05 | [One scene per prompt (C5)](05-one-scene-per-prompt.md) | Merging repeats removes duplicate scenes | **Adopted** ("absolutely incredible") |
| 06 | [Hard cuts per scene (D)](06-hard-cuts-and-artefacts.md) | Cuts beat the chain | Dead end; found the Whisper cause |
| 07 | [LTX settings permutations](07-ltx-permutations.md) | LoRA 0.6 / no even-light / 0.6 s lead beats C5 | Not compared; open |
| 08 | [Master lyrics (C6)](08-master-lyrics.md) | Master lyrics instead of Whisper's words fix the garbling | **Adopted** ("nearly perfect") |
| 09 | [ACE-Step song](09-ace-step-song.md) | ACE-Step sings the lyric intelligibly | **Adopted**; 77% with the LM |
| 10 | [Lyric respelling loop](10-lyric-respelling-loop.md) | Phonetic respelling fixes garbled names | Partial |
| 11 | [10-second test slices](11-test-slices.md) | Short tests from the full analysis catch problems early | **Adopted** |
| 12 | [Treatment "director" (offline-local v1)](12-treatment-director.md) | A whole-video plan improves coherence | **Rejected**: it overrode the lyrics |
| 13 | [Previous-line context (offline-local v2)](13-previous-line-context.md) | The previous line's visual makes scenes follow on | Rendered; 1 flare in 2:12; your verdict pending |
| 14 | [Strix Halo calibration](14-strix-halo-calibration.md) | Tritus is slow because of settings | Confirmed: 386 → 78 s per still |
| 15 | [ROCm 7.14](15-rocm-7-14.md) | Newer ROCm is faster | Blocked |
| 16 | [Qwen-Image on Strix Halo](16-qwen-image-on-strix.md) | Usable with the right settings | Ruled out for now (94–2996 s per frame) |
| 17 | [MacBook LTX renderer](17-quintus-ltx.md) | Quintus renders LTX-2.5 natively at Spark speed | **Passed**: 353 s vs 377 s |
| 18 | [Identity from a reference photo](18-identity-reference-chain.md) | Qwen-Image-Edit + her photo keeps the singer the same person | Identity **passes**; style and speed fail; round 2 next |
| 19 | [InfiniteTalk lip-sync](19-infinitetalk-lipsync.md) | Her photo can sing a chorus with believable lip-sync | **Passed** in fast mode (39 min for 16.6 s); your verdict pending |
| 20 | [One character, 40 trap scenes](20-identity-endurance-40.md) | Fixed anchors keep an invented character the same across macvideo's 40-scene fixture | Identity **passes** (0.86 mean, 40/40); poses copy the anchors; Lightning: 61 s a still |
| 21 | [Three characters kept apart](21-multicast-casting.md) | Cast-aware references keep Alan, Barry and Christy distinct | Queued after 20 |
| 22 | [Amber: placement, lip-sync, insertion](22-amber-investigation.md) | Place her in any scene and look, lip-sync it, insert it anywhere for any length | Q1–Q3 each **pass a first test**; style of her shots still photographic |
| 23 | [Can the LLM write actor cues?](23-actor-cues.md) | Qwen decides when and how Amber appears, from plain lyrics | **Works**: 18 s, words safe, directions obeyed; 69–81% agree on face-on-screen with the hand version |
| 24 | [One-word styles](24-one-word-styles.md) | A few words ("stained glass") reshape every frame via an LLM style brief | **Passed**: 5 of 6 styles strong on MOSS |
| 25 | [Still check before the video](25-still-check.md) | Blank stills and strangers can be caught and re-rolled automatically | **Built**; right on 4 runs |
| 26 | [Scenes as the unit of editing](26-scene-metadata.md) | Edit scenes, re-render only what changed | **Built**; Slippery Road v2 uses it |
| 27 | [Cuts vs morphs: the LoRA trigger](27-transition-trigger.md) | Morphs need an A→B description + "zhuanchang" | **Confirmed** (3 variants × 3 changes) |

## Road to the first renders (Chronophobia, Slippery Road)

| Step | What | Needs | Status |
|---|---|---|---|
| A | Analysis for Slippery Road; actor cues for both songs (POC 23), reviewed by the user | gruntus (LLM, analysis) | Slippery Road analysis running |
| B | Place Amber in any scene and look (POC 22 Q1) | Sparks (Qwen-Image-Edit); tritus for MOSS scene bases | next on the Sparks, after POC 20 |
| C | Lip-sync a placement (Q2): LongCat from a still; InfiniteTalk dubbing an LTX clip | Sparks | after B |
| D | Cast step in poptok: cues pick each line's route (MOSS / Qwen-Image-Edit with Amber / a sung shot) and sung shots are inserted on the song's clock (Q3); checked with 10-second tests | host code + one Spark | written alongside B and C |
| E | Full renders: Chronophobia on thing1, Slippery Road on thing2 | both Sparks | after D |

Off the critical path, run when a box would otherwise sit idle: POC 21 (three characters), POC 07 (LTX settings),
InfiniteTalk at full 40-step quality.

| 28 | [Plausibility critic (local Qwen vision)](28-plausibility-critic.md) | promising: + segmentation 4/4 caught, 0 false alarms (tuned on the same set) |
