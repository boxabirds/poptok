# POC 22: Amber in the video (placement, lip-sync, insertion)

**Date:** 2026-10-23 · **Status:** Q1, Q2, Q3 each pass a first test; casting step built; sung inserts next · **Piece:** chronophobia · **Performer photo:** `pieces/artists/r.amber.jpeg`

Three questions, in order; each gates the next.

## Q1. Can we place Amber in an arbitrary scene with arbitrary styling?

**Hypothesis:** from one photo, local models can put her in any scene (a lyric's setting, any pose or framing) in any
look (Chronophobia's hand-tinted pointillism, Offline's gritty black and white, and styles far from a photo, such as
oil paint or anime), and she stays recognisably her.

**Already known (POC 18 round 1):** Qwen-Image-Edit with her photo keeps her identity, but the look comes out
photographic, the style becomes props, rolling references copy the set forward, and it takes 230–580 s per still.

**Test:** 6 scenes × 4 looks, three approaches:

| # | Approach | Idea |
|---|---|---|
| A | Qwen-Image-Edit, her photo as the only reference, look in the prompt | Simplest; round 1 minus the rolling references |
| B | MOSS makes the scene in the look with a woman matching her description; Qwen-Image-Edit puts her face and hair in from the photo | The look comes from MOSS, which handles treatments well |
| C | Qwen-Image-Edit places her (as A); a light MOSS image-to-image pass applies the look | Identity first, look second |

**Pass if:** identity holds in every still (face, hair, age; clothing may change), the look matches the style's
non-Amber scenes, the scene follows the prompt, no props carried between stills; time per still noted.

## Q2. Can we lip-sync that placement?

**Hypothesis:** a placed still from Q1 can become a singing shot that keeps both her identity and the look.

**Already known (POC 19):** InfiniteTalk lip-syncs her photo well ("flying pass"), but on a plain grey wall in
photographic style.

**Test:** the best Q1 stills in two looks, the same chorus vocal, two routes:

| # | Route | Notes |
|---|---|---|
| A | Still → **LongCat-Video-Avatar 1.5** (image + audio → video) | Whisper-large-v3 audio, 8 steps, made for stylized looks |
| B | Still → LTX clip (her moving in the scene) → **InfiniteTalk dubs the clip** | Keeps LTX's camera and body motion; InfiniteTalk's video-to-video mode |

**Pass if:** lip-sync holds (by eye, plus mouth vs vocal correlation as in POC 19), identity holds to the last frame,
the look survives (no drift back to photographic), motion reads as a performance in the scene.

## Q3. Can it be inserted at any place in a video, for any duration?

**Hypothesis:** a performance shot can be cut into a poptok render at any lyric line, for any length, in sync with the
song and without a visible seam.

**Test:** in a Chronophobia render, insert performance shots of 4 s, 12 s and 30 s at three places (verse, chorus,
bridge):
1. Take the window from the aligned lyrics, slice the vocal stem to it, and generate the shot at exactly that length
   (both models stream in chunks for any length).
2. Conform it: 16:9 framing, upscale to the render's size, 24 fps, the piece's grade.
3. Join it: a cut on the downbeat, or an LTX Transition from the previous scene's last frame into the performance's
   first frame, and back out.

**Pass if:** her lips stay in sync with the song at every insert point (offset under about 40 ms, one frame),
duration exact, no seam or flash at the joins, the look matches the scenes around it.

## Order and machines

Q1 needs only stills: about 70 stills over the three approaches, about 2–3 h across both Sparks. Q2 needs the LongCat
image (built on thing1) and video input for InfiniteTalk (the decord stub needs replacing with an ffmpeg reader).
Q3 is mostly editing work on the host, plus generation at the lengths needed.

## Results

### Q1, first look (2026-10-24, Chronophobia 30 s test, cues from POC 23)

The casting step is built: `prompts_per_line.py --cues` writes each line for its cue, `keyframes.py --skip-cast`
leaves her shots, `amber_keys.py` renders them with Qwen-Image-Edit (Lightning) and her photo as the only reference:
**24 s a still**. 16 shots: 7 part, 6 sing, 3 without her.

- **Identity:** the sung shots are clearly her; face similarity to the photo 0.39–0.83 (one profile at 0.24, which
  SFace handles badly). **Poses vary** (profile, eyes closed, arm out): one photo reference doesn't lock the pose the
  way POC 20's three generated anchors did.
- **Part shots** work (hands, the sweater on its hanger) except one mirror shot that shows her full face.
- **Look: fails, as in POC 18.** Photographic, not hand-tinted sepia; "pointillist dots on skin" become dots on her
  cheeks, the pressed-flower collage becomes flower cards pinned behind her. MOSS's own shots have the look, so the
  two kinds of shot don't match.
- Next: approach C (`restyle.py`, a MOSS image-to-image pass at s0 0.35 / 0.5 / 0.65), scored for identity after.

### Q1, approach C (MOSS restyle) and the two test renditions

- **Restyle pass: no effect at s0 0.35 / 0.5 / 0.65.** The outputs are nearly identical to the Qwen-Image-Edit stills
  (13 s a pass: MOSS's schedule leaves few steps below 0.65). Identity unchanged (0.45 / 0.42 / 0.38 mean). It needs
  0.8 or more to change the look, which risks identity; strong styles are tested next (Qwen-Image-Edit with a POC 24
  style brief leading the prompt, `amber_style_test.py`).
- **Chronophobia 30 s test** (`pieces/chronophobia-test30/outputs/test/`): one woman throughout (the 3-women problem is
  gone); no flares; her shots barely move in LTX (near-still profile, mouth nearly static), and transitions between
  two shots of her sometimes pass through a double exposure.
- **Slippery Road 45 s test** (`pieces/slippery-road-test45/outputs/test/`): the cues' story plays out: the dogs, a
  hand on the pavement, the reveal of her lying on wet stones reflected in a puddle, her hand again at "It's my hand",
  then Chorus 2 at the London front door with the key, singing in the rain (the user's direction). The look matches
  the treatment (photographic winter blue), one flare at 7.5 s. Face scores are low (0.33) because she lies sideways,
  which SFace can't read; by eye it is her.

### Full Chronophobia v3: what the identity check flagged (106 stills)

59 stills have a face; mean similarity to her photo 0.58; 12 below 0.363. By eye:
- **Intro shots invent a stranger** (shots 3–6, before the first sung line, so no cue): MOSS draws a red-haired woman
  and portraits on the walls. Fixed for later runs: with cues, the intro prompt says "no people".
- **Her face leaks into a second person**: "While I'm spilling it with you" (shots 67–69) shows two women who both
  look like her. A second character needs its own reference or an explicit different description (POC 21).
- The rest are SFace misreads: empty rooms where it found portrait faces, and her own shots with closed eyes or in
  profile.

### Q2: lip-syncing a placement (LongCat-Video-Avatar 1.5)

Input: a Chronophobia casting still of her singing (test shot 13, Qwen-Image-Edit, in the scene's collage) and the
chorus vocal (44.2–60.8 s). `docker/longcat` on thing1 (built on the infinitetalk image; xformers in every attention
class via `LONGCAT_ATTN`), 8 distilled steps, 6 segments, 480p, no offload: **44 min for 16.6 s** (~160 s per second
of video; InfiniteTalk ~140 s).

| | InfiniteTalk (POC 19, from her photo) | LongCat 1.5 (from a placed still) |
|---|---|---|
| Lip-sync (`eval_lipsync.py`) | r = 0.31, mouth leads by 120 ms | r = 0.23, mouth leads by 120 ms (baseline 0.09) |
| Identity (SFace vs her photo) | 0.68–0.83 | 0.59–0.69 |
| Framing | portrait 512×768 | **landscape 832×480**, cuts straight into 16:9 |
| Look | the photo's grey wall | **keeps the placed still's look** (the collage behind her) |
| Expression | head tilts, eyes closing | more expressive: O and pursed shapes, frowns, small smiles |

Watch: `pieces/chronophobia/outputs/longcat_poc/longcat_chorus_mix.mp4`. Verdict pending the user's eye; LongCat is
the better fit for insertion because its output is already in the scene's framing and look.

### Q3: inserting it on the song's clock

The LongCat chorus cut into the Chronophobia 30 s test at the first downbeat after the vocal starts (14.93 s in the
clip, 44.93 s in the song), trimmed 0.73 s in so it stays on the song's clock, scaled to 1280×720 at 24 fps:
`pieces/chronophobia-test30/outputs/q3/chrono_test30_with_longcat.mp4`.

- Duration exact (30.00 s); a clean hard cut on the downbeat; no flare.
- **In sync after the cut:** against the song's own vocal stem the inserted part scores r = 0.32 with the mouth
  leading by 125 ms, the same as the standalone clip (−120 ms) to within a frame.
- Not yet tried: an LTX Transition into and out of the performance instead of a cut.

### User review, 2026-10-24 morning

| Output | Verdict | Action |
|---|---|---|
| Slippery Road 45 s test | Good. Her falling beside the puddle "truly amazing". But the pulse is random; her lighting doesn't match the scene; the key goes in where there is no lock | Pulses now snap to the beat grid (`finish.py --punch-snap`); her shots now ask for the scene's own light (`amber_keys.py`) |
| Chronophobia 30 s + LongCat insert | Promising, but LongCat can't say "r" ("fwames") | Sung inserts switched to InfiniteTalk from the same placed still |
| Chronophobia 30 s test | The "crease in the face" is strange; no stylistic treatment of her; transitions not coherent; a blouse turns into a cardigan | Her shots keep the sheer black dress from her photo unless the shot says otherwise |
| LongCat chorus | **Fail**: lips say "I'm fwightened of warm leaving soon" | LongCat dropped for lip-sync |
| InfiniteTalk chorus | **Success** | InfiniteTalk is the lip-sync engine |
| Amber in strong styles | Claymation and stained glass great; claymation the favourite | Oil impasto test queued (Chronophobia briefs) |

### Sung inserts in a full rendition (InfiniteTalk, 2026-10-24)

`pieces/chronophobia/outputs/v3/chronophobia-v3-sung.mp4`: Chronophobia v3 with every run of "sing" shots (the four
chorus stretches, 50 s) replaced by an InfiniteTalk performance started from that run's first casting still and driven
by the vocal stem for exactly that stretch (`pipeline/sing_inserts.py`, `bin/sing_inserts.sh`; fast 4-step mode, no
offload, ~33 min per 12 s insert on one Spark). Length unchanged (3:30.04); cuts on the shot boundaries (downbeats).

| Chorus | Lip-sync (r, lead) | Identity (SFace) |
|---|---|---|
| 1 (0:45) | 0.28, 160 ms | 0.61–0.69 |
| 2 (1:01) | 0.34, 120 ms | 0.53–0.62 |
| 3 (2:19) | 0.17, 200 ms (baseline 0.14): small mouth movements; golden wisps in the starting still | — |
| 4 (2:37) | 0.53, 80 ms | — |

The cut into chorus 1 is seamless: the LTX chain is already moving towards the same still the performance starts from.
