# poptok: working notes for Claude

poptok makes music videos from lyrics, a visual treatment and a song (README.md). Current goal: videos for 25+ songs
by R.Amber, keeping every box usefully busy, calmly.

## Keep the log and the runbook current (every session)

- **[docs/log.md](docs/log.md): what was done.** Append dated entries (`## YYYY-MM-DD`, then `- **HH:MM** what was done
  → outcome`, with file paths, POC links and the user's verdict when given). Newest at the bottom. Write entries as work
  happens (each render started/finished, each test and its result, each pipeline change, each user decision), not in a
  sweep at the end. Times are local (Europe/London).
- **[docs/runbook-julian.md](docs/runbook-julian.md): how to reproduce on this setup.** Machines and their roles,
  paths, exact commands, settings, operating gotchas. Update it in the same session whenever any of those change
  (a new script or flag, a host's role, a model, a fix to a recurring problem).
- **[docs/hardware-performance.md](docs/hardware-performance.md): process step × box performance table** (gruntus,
  thing1/2, tritus, quintus). Every time a step is timed on a box (a render, a test, a benchmark), add or update that
  cell with the value, date and source; mark impossible combinations n/a with the reason (too big without offload,
  no port, not tried). The user advises others on hardware for local AI from this table, so measured numbers only.
- **[docs/change-impact.md](docs/change-impact.md): what a change costs.** The pipeline end to end, each step's
  inputs/outputs/cost, and a feedback → redo-from → reused → cost table. Update it whenever a step is added, removed or
  reordered, its dependencies change, or its measured cost changes; and use it to tell the user the cost of feedback
  before acting on it.
- Experiments with a hypothesis go in `docs/pocs/NN-name.md` (hypothesis, test, expected, result, status) and get a
  line in `docs/pocs/README.md` and an entry in the log.

## Rules

See the runbook's "Rules" section. In short: containers for everything (quintus native); never CPU offload; one GPU
job per box; local models only; never sudo; HF token only in env vars; outputs for the user on the share
(`pieces/...`, `pieces/_bench/eval/`); commit or push only when asked. Never edit a running bash script in place
(write a new file and `mv` it over).

## Compute stewardship (your responsibility)

Keep every box (gruntus, thing1, thing2, tritus, quintus) used efficiently and effectively: queue the next useful job
when one finishes, put each job on the box that suits it ([docs/hardware-performance.md](docs/hardware-performance.md)),
and tell the user when a box sits idle because there's nothing worth running. But respect other projects' work on the
same boxes: before starting a GPU job, check what's already running there (docker/podman ps, nvidia-smi, the lock).
If another project's job is in the way, **tell the user and let them decide**; don't stop it yourself. dbench
(the user's benchmark, 12–24 h per run) will usually be running somewhere; the user's default answer is "pause it so
this run can continue" (pause = hold the node and cancel the running job; resubmitting the same run-id resumes it),
but still say so and wait for the go-ahead.

## Reporting

When there are things for the user to look at, give a numbered table: no. / file path / description / self-eval.
Judge videos by consecutive frames, not sparse contact sheets.
