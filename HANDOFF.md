# GeoBench Copilot — Handoff

## Goal
An agentic web app where a non-ML domain expert (e.g. an energy analyst) uploads a dataset + describes a task
(example: "detecting wind farms in Sentinel-2 imagery") and an agent surfaces the most relevant experiments from
**torchgeo-bench** and tells them how to run their own. Built for the "Applied practitioner / non-ML" audience.

Grounded in the real torchgeo-bench repo (Chesweinfeld/Torchgeo-Bench_Copy_For_Experimentation) — it ships no UI,
so this is a new surface built on that repo's data model (datasets like m-pv4ger/m-brick-kiln/m-eurosat, backbones
like DINOv3-sat, DOFA, ResNet-MoCo, RCF; metrics accuracy/mIoU/F1; BandSpec/BenchDataset classes).

## The single deliverable
`GeoBench Copilot.dc.html` — a Design Component. Open directly in a browser. Supporting files: `image-slot.js`
(drag-drop chip slots), `fonts/` (see below).

Visual identity = **Ode Partners** (user works at ode.partners): black / bone / grayscale, `ode` wordmark in header,
CSS-variable theming with an animated **dark/light toggle** (sliding sun/moon knob). NO red / no color accents.
Display font is **RaptorText**, wired via `@font-face` at `./fonts/RaptorText.woff2`, `RaptorText600.woff2`,
`RaptorText700.woff2` with Schibsted Grotesk fallback. Body/mono: IBM Plex Mono for code/labels.

## What's built (all working)
Flow: dark left rail (drag-drop dataset + task, example chips, run button) → step-by-step agent reasoning →
right results canvas.
- **Agent reasoning timeline** (left rail): inspect → characterize → select metric → match datasets → rank backbones →
  check normalization → generate wrapper → recommend. Toggle step-by-step vs concise via the `reasoningStyle` prop.
- **Recommendation** card (frozen DINOv3 sat493m + linear probe, expected 0.96) with a **metric badge**.
- **Result visualizations**: linear-probe leaderboard (bars) + accuracy-vs-compute scatter. Toggle via `showCharts` prop.
- ~~**Predictions on your chips**~~ — REMOVED 2026-07-26 (see Resilience pass). The panel staged drop-in chips
  but could never score them (no in-app inference), so it promised something the app doesn't do. Its
  "not scored in-app" warning now lives inside the recommendation card, where the decision is made.
- **Closest benchmarks**: 3 cards with per-factor match bars (Bands / Label type / Task / Object scale).
- **Recommended backbones**: ranked list with exp. acc + GMACs.
- **Generated wrapper**: auto-written `datasets/windfarm_s2.py` (real BenchDataset subclass w/ BandSpec + per-band stats),
  copy button.
- **Build-your-run** panel: checkboxes to pick which experiments to run, live wall-clock **time estimate** that reacts to
  selection + hardware (A100/T4/CPU), an "emit HTML results dashboard" toggle (adds `report.format=html`) with a preview,
  and a live-updating `torchgeo-bench run` command (single `model=` vs `model=[...]`).
- **Caveats**: RGB-only, avoid model_native, imbalance/metric warning, transferred-scores disclaimer.
- **Auto glossary**: every technical term (models, datasets, metrics, bandspec_zscore, GMACs, macro-F1, etc.) is
  auto-underlined and opens a definition popover on click. Add terms to the `GLOSSARY` object and they're detected in prose
  automatically (see `tokenize()`).
- **Eval-metric selection knowledge** (latest): agent explains WHY it picked a metric — binary→accuracy, but imbalanced
  21/79 split → primary macro-F1; segmentation would use mIoU. Command emits `eval.metrics=[f1_macro,accuracy]`.

## Architecture notes (DC format)
- `renderVals()` returns all template inputs; template uses `{{ }}` dotted holes + `<sc-for>`/`<sc-if>`. Inline styles only.
- Theming: CSS custom props on `[data-theme]` in `<helmet><style>`; `--rail*` for dark panels, `--ink/--surface/--muted` for canvas.
- State machine: `phase` = upload → running → done. `run()` steps through `STEPS` on timers.
- Props (Tweaks): `showCharts` (bool), `reasoningStyle` (enum). Add more via `dc_set_props`.

## Backend (NEW — "make it real" is done)
**Pickup status (2026-07-14):** built + verified except the live agent E2E, which only needs
`ANTHROPIC_API_KEY` exported before starting the server (see backend/README.md pickup note).
`backend/` is a real agentic backend: FastAPI + SSE, with a Claude agent (Opus 4.8, SDK tool
runner) over deterministic tools — archive inspection (safe zip extraction, per-band Welford
stats, structure/sensor detection), facet match scoring against the real torchgeo-bench
registry (dumped to `backend/data/registry.json`), backbone ranking from the real
`results/all_results.csv` (linear/knn5 + gflops join), and template-filled BenchDataset
wrapper generation. Claude writes prose; tools compute numbers; the final payload is
assembled server-side from cached tool outputs so numbers can't be hallucinated.
See `backend/README.md` for setup/run. The frontend is wired: real file upload + drag-drop,
EventSource-driven agent steps, state-driven renderVals with the old literals kept as a
standalone demo fallback (opens with no backend / no file → canned walkthrough). The fake
`eval.metrics=`/`report.format=` CLI keys are gone — generated commands use only real
Hydra keys. Serve via `uv run uvicorn app:app --port 8000` in backend/ and open
http://localhost:8000/ (needs ANTHROPIC_API_KEY for live runs).

## Matcher calibration + external evals (2026-07-17)
The facet weights in `backend/tools/matching.py` are now **measured**, not guessed:
- `backend/scripts/calibrate_weights.py` — leave-one-dataset-out (LOO) harness scoring the
  matcher as **regret** (accuracy lost vs the truly-best backbone) + a sim↔transfer-utility
  rank correlation. CPU-only. `--search` = single-corpus grid search;
  `--source {tgb-linear,tgb-knn5,external}` picks the corpus; `--multi` = ROBUST search across
  all three corpora (minimax over baseline-normalized regret).
- `_WEIGHTS` = `{domain .25, task .20, bands .40, label .10, scale .05}`, the robust cross-corpus
  optimum. Single corpora overfit + disagree (tgb-linear wanted bands=.45/task=.15; GB2 wanted
  task=.55/bands=.20), so `--multi` picks weights that never fail badly on any protocol: regret
  0.0100 (tgb-linear) / 0.0148 (tgb-knn5) / 0.0184 (GB2 frozen). Old AHP prior kept as
  `_WEIGHTS_PRIOR`. CAVEAT: still ~11-15 folds/corpus — indicative, re-run `--multi` as folds grow.
- `backend/scripts/import_leaderboards.py` — pulls **frozen-encoder** evals from public
  leaderboards into `backend/data/external_results.csv`, protocol-tagged by `source`, **never**
  mixed into the canonical `all_results.csv` (different backbone namespaces / decoders):
  - `source=geobench2`: GEO-Bench-2 leaderboard (The-AI-Alliance/GEO-Bench-2-Leaderboard), frozen
    submissions only, 15 datasets. Fetched live via the GitHub API.
  - `source=pangaea`: PANGAEA (arXiv:2412.04204) Table 5 frozen results, 12 backbones × 11 datasets.
    Its leaderboard site is offline + repo has no CSV, so the verified paper table is committed as
    `backend/data/pangaea_raw.csv` (extracted from the arXiv HTML, cross-checked vs Table 7 anchors).
    4 datasets overlap the registry (burn_scars/pastis/dynamic_earthnet/spacenet7) → 4 usable folds.
  Corpora in calibration: `--source {tgb-linear,tgb-knn5,geobench2,pangaea}`. `--multi` searches all;
  corpora with <6 usable folds (pangaea today) are **validation-only** — reported, not in the objective.
- Independent validation: `calibrate_weights.py --source external` → matching halves regret vs
  no-matching (0.0267 vs 0.0520, 14 folds). The matcher's value generalizes to a separate
  protocol. (Caveat: sim↔utility rank corr ~0 on GB2 vs +0.33 on tgb.)
- `biomassters` is imported but not yet in the registry (add META/DOMAINS/DESCRIPTIONS to make
  it matchable).

## Registry expansion (2026-07-17)
Added 7 external (non-torchgeo-bench) datasets to `registry.json` + `matching.py` sidecars so their
imported evals become calibration folds: `biomassters, mados, sen1floods11, xview2, fbp, cropmap,
ai4smallfarms` (hand-authored facets from each dataset's paper; `source=external_handauthored`;
per-band stats omitted — matching reads only wavelength/sensor). Also generalized `_score_bands`
to a shared `_OPTICAL_SENSORS` set (adds maxar/gaofen2/spot). Effect: geobench2 → 15 folds,
pangaea → 11 folds (both fully unlocked).

**Findings (why weights stayed at CURRENT):**
- PANGAEA is now **validation-only by policy** (`_VALIDATION_ONLY` in calibrate_weights.py):
  matching can't beat its no-matching baseline at any sane weighting — the only vector that does
  (minimax) doubles tgb-linear regret. Its heterogeneous task mix doesn't follow facet-similarity.
- **`biomassters` (regression) poisoned segmentation pooling — FIXED 2026-07-17.** As a matchable
  candidate it looked facet-similar to flood/burn datasets and landed in their top-3, but its RMSE
  backbone ranking doesn't transfer → kuro_siwo regret 0.007→0.318, geobench2 0.0184→0.0542. Fix:
  a proper `regression` task family — biomassters is now `task="regression"` in the registry;
  `_score_task` isolates regression (score 20 vs categorical, 100 self); `matching.is_regression()`
  gates calibration pooling so regression and categorical benchmarks are never pooled together.
  Result: geobench2 back to 14 folds / 0.0184; biomassters no longer surfaces as a close match for
  a flood task (sim 59, task-facet 20, below the real flood benchmarks); it drops as a calibration
  fold (lone regression, no same-regime peers) so pangaea is 10 folds. The production agent
  (`query_results`) only reads all_results.csv (classification-only), so it never pooled regression
  anyway — no change needed there.
- `task .05` in the objective optima is a **measurement artifact** — each corpus is near
  single-task-family, so task has low variance and the optimizer starves it; task is a near-gate
  for real cross-family user queries, so we keep it at .20.

## Running the missing torchgeo-bench experiments (2026-07-17)
The 134 missing linear cells in all_results.csv are all **16 OlmoEarth backbones × the 11
result-bearing datasets** (the OlmoEarth family was added to the model list but never run).
`scripts/probe_datasets.py --mode missing` orders them **shortest-first** (cost = dataset
chip-count ÷ measured per-backbone throughput, both read from the CSV's profile rows), with
`--limit N`, `--device {auto,cuda,mps,cpu}`, and `--run`. Verified end-to-end on macOS/MPS:
`eurosat-spatial × olmoearth_v1_2_nano` produced a real row (linear acc 0.827, knn5 0.716).

Blockers resolved to make a run succeed (all now done in the local TGB worktree):
1. **Shell glob** — `dataset.names=[x]` must be quoted (zsh/sh glob it). Fixed in
   `tools/results.build_run_command` (`'dataset.names=[...]'`).
2. **CLI on PATH** — `torchgeo-bench` lives in `$TGB_REPO/.venv/bin`; the runner now injects it.
3. **Data staging** — `torchgeo-bench download eurosat` (+ a one-time `EuroSATSpatial(download=True)`
   for the spatial-split txts). Other datasets need `download geobench_v1` (m-*) / `geobench_v2`.
4. **Model dep** — `uv pip install "olmoearth-pretrain-minimal>=0.0.6"` in the TGB venv.
5. **Device** — TGB defaults `device=cuda:0`; on Apple silicon pass `device=mps eval.knn_device=cpu`
   (the runner's `--device auto` does this).
Each MPS run ≈ 3 min (nano) and up. eurosat-spatial's remaining 15 OlmoEarth cells are runnable now
(data staged); m-*/others need their geobench download first.

## Resilience + UX pass (2026-07-26)
- **SSE is now broadcast-safe**: `runs.py` replaced the per-run shared `asyncio.Queue`
  (which handed each event to only ONE consumer — reconnects/second tabs silently lost
  events) with an `asyncio.Event` signal + per-subscriber index into `run.events`.
  Verified with two concurrent curl consumers + `Last-Event-ID` replay.
- **Frontend stream recovery**: `es.onopen/onerror` handling — transient drops show a
  "reconnecting" note (EventSource auto-reconnect + server replay heals them); after 3
  failures it falls back to polling `GET /api/runs/{id}` (the agent keeps running
  server-side through a dropped stream). `done` without a result now errors instead of
  spinning forever.
- **Sample mode is explicit + labeled**: running with no file no longer silently plays
  the canned wind-farm demo as if it analyzed your task. The run button validates
  (file attached, task non-empty, not file://) with inline errors; a dedicated
  "watch a sample walkthrough" link runs the demo, which is labeled with a SAMPLE chip
  in the rail + a SAMPLE banner in the canvas.
- **Backend status in the header**: the UI probes `/api/health` at mount — bright dot +
  real counts ("116 backbones · 37 datasets · live") when connected, "backend offline ·
  sample mode only" otherwise. `/api/health` gained a `backbones` count.
- **Misc backend hardening**: startup sweep of orphaned `.runs/` dirs from dead
  processes; strong reference to the agent asyncio.Task (GC safety); non-529 Claude API
  errors now surface the API's own message (e.g. the billing error) instead of a
  misleading "please retry".
- **Test-chips panel removed**: it let you stage chips but never scored them (the app runs no
  inference), which read as a broken feature. Gone from the template, `renderVals`, the
  `image-slot.js` include, and `FinalPayload.chips`/`ChipPred` in schemas.py (both were dead —
  `_build_payload` never populated them). Its warning is now a footnote inside the recommendation
  card: nothing is scored here, run the generated wrapper to get predictions on your own chips.
- **Docs**: new top-level `README.md` (quick start, run anatomy, repo map);
  backend/README's stale AHP weights line replaced with the calibrated weights.
- **Metric labels now derive from the ranking metric** (2026-07-27). One `metricLabel(m, form)`
  helper maps raw CSV metric names (`accuracy`, `micro_mAP`, `f1_macro`, `mIoU`, …) to one
  consistent display label, used by the backbone-section label, leaderboard subtitle, per-model
  column, big-number caption, and the dashboard stat. Previously each site formatted it
  differently (`acc` / `micro_mAP` / `ACC`) and the section label was hardcoded to "acc" in
  sample mode. Crucially the labels derive from **rankMetric** (what the leaderboard numbers
  actually are), never from `rec.metric` (what's recommended for the user's task) — when the two
  differ, the section label says so: "ranked by linear-probe accuracy — these benchmarks don't
  report macro-F1". Both branches verified in-browser.
- **Live agent E2E VERIFIED (2026-07-27)** — credits added; a real run on
  `data/synthetic_windfarm.zip` streamed all 8 steps and produced a full payload (top pick
  `tgeo_dofa_large` 0.9715 acc on m-brick-kiln/m-pv4ger, wrapper, 5 caveats, 7 glossary terms).

## TODO / next ideas (not yet done)
0. Run the rest shortest-first: `python3 scripts/probe_datasets.py --mode missing --run` (stage
   geobench_v1 / geobench_v2 data first for the non-eurosat datasets). Then
   `calibrate_weights.py --multi` folds the new OlmoEarth rows in.
1. (DONE) Regression-pooling fix. Add more regression benchmarks so regression can be calibrated.
1. (DROPPED) Chip predictions — the staging panel was removed rather than made real. If in-app
   inference ever lands (running a frozen backbone + probe server-side on uploaded chips), rebuild
   the panel then; `image-slot.js` is still in the repo, just no longer loaded.
2. Support **segmentation tasks** end-to-end (mIoU leaderboard, real mask predictions from an FPN/DPT head) —
   the matcher handles them and the agent caveats "no results yet", but there are no accuracy rows in the CSV.
3. Drop the real **RaptorText .woff2** files into `fonts/` (user has them) — falls back to Schibsted Grotesk until then.
4. Wire the "generate HTML dashboard" to actually produce a downloadable report.
5. Persist runs (RunManager is in-memory, demo-grade).

## Design system
Bound design system project id: 5181c60b-e075-43d5-b9b5-459ff20be319 (currently empty — visual direction comes from Ode's
site + the user's screenshot, captured as the grayscale system above).
