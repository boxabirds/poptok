# ACE-Step 1.5: you can't easily tell whether it will offload

**Written:** 8 November 2026. **Code examined:** ACE-Step 1.5, commit `ca1e85f` (29 August 2026), in `~/src/ACE-Step-1.5`;
our wrapper `pipeline/ace_gen.py`. **Runs examined:** the song tests of 23 October on gruntus (RTX 4090, 24 GB).

## Summary

ACE-Step 1.5 can move its models between the graphics card and ordinary computer memory, swapping each one in only
while it runs ("CPU offload"). This lets big setups run on small cards, but more slowly. We don't allow it: if a model
doesn't fit a machine, it should run on a bigger machine instead.

**ACE-Step gives no clear way to know in advance whether a given setup will need offloading, and its own launcher
switches offloading on without asking.** You find out afterwards, from its log. We learnt this the hard way: our
planner tests of 23 October were offloaded, and nothing in our results said so.

## What ACE-Step does

| Situation | What happens |
|---|---|
| **ACE-Step's launcher** (its command line and web app, `acestep_v15_pipeline.py`) | Decides for you, and only prints a line about it:<br>• cards under 20 GB: offloading on (l. 159–194);<br>• cards of 24 GB or less with the 4B planner: offloading on (l. 428–436). This includes the RTX 4090;<br>• cards under 20 GB: the 4B planner is swapped for the 1.7B (l. 438–448).<br>There is no option to fail instead. |
| **ACE-Step's Python interface** (what `pipeline/ace_gen.py` uses) | Offloads only if the caller asks (`offload_to_cpu=True`); it never switches it on by itself. |
| **Offloading off, and the models don't fit** | It fails: either loading stops with an out-of-memory error, or a memory check just before the music step stops it ("Insufficient free VRAM: need ~X GB, only Y GB available"). |
| **A check before loading** | None. Its memory check (`_vram_preflight_check`, `core/generation/handler/generate_music.py` l. 106) runs after the models are loaded. It only checks there is room left for the music step itself, using a rough table (`DIT_INFERENCE_VRAM_PER_BATCH` in `gpu_config.py`: 0.3–1.0 GB per song). When offloading is on, it skips the check entirely. |

### Its documentation contradicts its code

ACE-Step's GPU guide (`docs/en/GPU_COMPATIBILITY.md`) puts 20–24 GB cards (such as the 4090) in "Tier 6b": offloading
"None", 4B planner available. It also says that on 24 GB and up "all models fit without offloading … use XL DiT + 4B LM
for best quality". But the launcher's code switches offloading on for any card of 24 GB or less when the 4B planner is
chosen. The 4090 reports 23.5 GB, so the launcher would offload it, despite what the table says.

### The nearest thing to a check

ACE-Step ships a profiling tool, `profile_inference.py --mode tier-test --tier-boundary`, that tries each card-size
tier with and without offloading and reports the smallest tier where offloading can be switched off. It is a benchmark
you run once across all tiers, not a check before a particular job, and it simulates card sizes rather than testing the
card in front of you.

## How you can tell that offloading happened

Only afterwards:

1. **The log.** Lines like these mean it offloaded:
   - `Offloading LLM to CPU` / `Offloaded LLM to CPU in 5.3036s`
   - `Loading vae to cuda … Offloading vae to CPU`
   - `VRAM pre-flight: skipping check (offload_to_cpu=True, models loaded one-at-a-time)`
2. **The program's own settings, after loading.** The handlers keep `offload_to_cpu` as an attribute, and the music
   handler adds up the time spent swapping in `current_offload_cost` (`init_service_offload_context.py` l. 53).
   Neither is reported unless you print it.

Neither tells you *before* you run whether the setup fits.

## What this meant for our planner test (POC 09, 23 October)

We compared songs with and without ACE-Step's planner on the RTX 4090
([POC 09](../pocs/09-ace-step-song.md)):

| Music model | Without planner | With 4B planner | Time per 150 s song |
|---|---|---|---|
| XL, high quality, seed 2 | 73% | 77% | 23 s → 59 s |
| XL, high quality, seed 1 | 69% | 77% | 24 s → 64 s |
| XL, fast, seed 1 | 70% | 75% | 8 s → 46 s |
| 2B, fast, seed 2 | 44% | 70% | 6 s → 45 s |

(Percentage: share of the 353 lyric words that a speech recogniser, Whisper, heard in the right order.)

- **The planner runs were offloaded.** XL plus the 4B planner doesn't fit in 24 GB. The planner runs' own log wasn't
  kept, but the same script with the same 4B planner on the same card that morning
  (`pieces/offline/outputs/songs/loop/r01/ace.log`, 10:29) logs "Offloading LLM to CPU" and
  "offload_to_cpu=True". Our script then had offloading on; the no-offload rule came at 17:29 that day.
- **The quality results stand.** Offloading changes speed, not the sound.
- **The times with the planner don't.** They include the swapping (about 5 s each time the planner moves out), so
  they are slower than a machine where everything fits, and can't be repeated under our rules.
- **Two setups have never been measured without offloading:**
  - 2B plus the 4B planner on the 4090. On paper it fits: about 5 GB plus about 8 GB.
  - XL plus the 4B planner on a 128 GB machine. There's no ACE-Step container for the DGX Sparks' ARM processors yet.

## What we should do

1. **Check before loading, in our own code.** `pipeline/ace_gen.py` should:
   - add up the chosen models' file sizes, plus ACE-Step's own estimate for the music step;
   - compare the total with the card's free memory;
   - if it doesn't fit, refuse and say which machine it would fit on.
2. **Remove `ace_gen.py`'s `--offload` flag,** so offloading can't be switched on by accident.
3. **Never use ACE-Step's launcher for measurements.** It decides about offloading for you.
4. **Measure the two missing setups without offloading:** 2B plus the planner on the 4090, and XL plus the planner on a
   Spark once an ARM container exists.
5. **Generally:** any model that can offload by itself needs the same treatment: a fit check before the job, and a
   log line that says plainly "offloading: on/off".

## Sources

| What | Where |
|---|---|
| Launcher auto-offload and planner downgrade | `~/src/ACE-Step-1.5/acestep/acestep_v15_pipeline.py` l. 154–194, 428–448 |
| Card-size tiers | `~/src/ACE-Step-1.5/acestep/gpu_config.py` l. 34–36, 173–181, 313–473 |
| Memory check before the music step | `~/src/ACE-Step-1.5/acestep/core/generation/handler/generate_music.py` l. 106–175 |
| ACE-Step's GPU guide and profiling tool | `~/src/ACE-Step-1.5/docs/en/GPU_COMPATIBILITY.md` |
| Our wrapper | `pipeline/ace_gen.py` |
| The planner test | `docs/pocs/09-ace-step-song.md`; `docs/log.md`, 23 October 10:43–11:07 and 17:29 |
| Songs and run settings | `pieces/offline/outputs/songs/ace/` (with planner), `ace_nolm/` (without); `runs.json` in each |
| Log showing the offload | `pieces/offline/outputs/songs/loop/r01/ace.log` |
