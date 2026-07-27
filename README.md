# GeoBench Copilot

An agentic assistant for geospatial ML practitioners: upload a dataset and describe
your task, and it surfaces the most relevant experiments from **torchgeo-bench**,
ranks candidate backbones on real frozen-probe results, and generates a runnable
BenchDataset wrapper.

The backend (FastAPI) serves both the API and the frontend. It's CPU-only — it
reads a bundled `all_results.csv` and calls the Claude API; no GPU or torch needed.

## Run locally
```bash
cd backend
cp .env.example .env      # then paste your ANTHROPIC_API_KEY
bash start.sh             # http://localhost:8000/
```

## Deploy
See **[DEPLOY.md](DEPLOY.md)** — one-click Render Blueprint (`render.yaml`), no
Docker required. The whole app is HTTP Basic-Auth gated via an `APP_PASSWORD`
secret so only people you share it with can run it (protecting the API bill).

## Data freshness
Backbone rankings come from a bundled snapshot of `all_results.csv`; the
"benchmark results as of …" badge in the UI shows its date. Refresh it with
`backend/scripts/bundle_results.sh` before redeploying.
