# GeoBench Copilot — backend

> **Pickup note (updated 2026-07-27):** fully verified end to end, including the
> live agent. A real run on `data/synthetic_windfarm.zip` streams all 8 steps and
> produces a complete payload (matched m-brick-kiln / m-pv4ger / substations,
> top pick `tgeo_dofa_large` @ 0.9715 accuracy, generated wrapper, 5 caveats,
> 7 glossary terms). SSE replay, multi-consumer streaming, and the error paths
> are verified too. Nothing in the torchgeo-bench repo itself was modified.

Agentic backend for the GeoBench Copilot frontend (`../GeoBench Copilot.dc.html`).
A Claude agent (Sonnet 5, SDK tool runner) orchestrates deterministic tools that
inspect an uploaded chip archive, match it against the torchgeo-bench registry,
query real results from `results/all_results.csv`, and generate a `BenchDataset`
wrapper. Progress streams to the frontend over SSE.

## Setup

```bash
cd backend
uv sync

# one-time: dump registry metadata (runs under the torchgeo-bench venv, which has torch)
/Users/chesapeake/torchgeo-bench/.venv/bin/python scripts/extract_registry.py \
  --repo /Users/chesapeake/torchgeo-bench/.claude/worktrees/cranky-chaum-ec1924 --out data/

export ANTHROPIC_API_KEY=sk-ant-...   # required for the live agent
uv run uvicorn app:app --port 8000
# open http://localhost:8000/
```

Env overrides: `TGB_REPO`, `TGB_RESULTS_CSV`, `GEOBENCH_MODEL` (default
`claude-sonnet-5`), `FRONTEND_DIR`.

## API

| Endpoint | What |
|---|---|
| `POST /api/analyze` (multipart `file`, `task`) | start a run → `{run_id}` |
| `GET /api/runs/{id}/events` | SSE: `step_start`, `step_detail`, `inspection`, `tool_result`, `final`, `error`, `done` (supports `Last-Event-ID` replay) |
| `GET /api/runs/{id}` | poll fallback: status + final payload |
| `GET /api/health` | registry/CSV load status |

## Test fixture

```bash
uv run python scripts/make_synthetic_zip.py        # -> data/synthetic_windfarm.zip
curl -F "file=@data/synthetic_windfarm.zip" -F "task=detecting wind farms in Sentinel-2 imagery" \
  http://127.0.0.1:8000/api/analyze
curl -N http://127.0.0.1:8000/api/runs/<run_id>/events
```

## Architecture

- `agent.py` — system prompt + `@beta_async_tool` closures + tool-runner loop.
  Claude writes prose; tools compute numbers. `submit_recommendation` takes only
  the authored parts; the server assembles the final payload from cached tool
  outputs so numbers can't be invented.
- `tools/inspect.py` — safe zip extraction, structure detection, per-band Welford
  stats, sensor heuristic.
- `tools/matching.py` — facet match scores (Domain .25 / Task .20 / Bands .40 /
  Label .10 / Scale .05 — calibrated cross-corpus by
  `scripts/calibrate_weights.py --multi`, see HANDOFF.md) against
  `data/registry.json`.
- `tools/results.py` — pandas over `all_results.csv`: linear/knn5 means joined
  with profile gflops; real `torchgeo-bench run` command generation.
- `tools/wrapper.py` — deterministic BenchDataset template fill.
- `runs.py` / `app.py` — run lifecycle, SSE queue + replay, static frontend serving.

The backend never imports `torchgeo_bench` (torch-heavy); registry metadata is
dumped once by `scripts/extract_registry.py`. Regenerate after repo updates —
`/api/health` reports the dump timestamp.
