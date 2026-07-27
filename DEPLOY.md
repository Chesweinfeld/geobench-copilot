# Deploying GeoBench Copilot to Render (no Docker)

The backend serves the whole app (frontend + API). It's CPU-only — no GPU, no
torch — so it runs on Render's free tier. Render builds it natively from
`requirements.txt`; you never touch Docker.

## One-time setup

1. **Rotate your Anthropic key** first (the old one was pasted in chat). Get a
   fresh key at console.anthropic.com.

2. **Put the code on GitHub** (Render deploys from a git repo). From the project:
   ```bash
   cd "/Users/chesapeake/Desktop/Agent dataset task matcher"
   git init && git add -A && git commit -m "GeoBench Copilot"
   # create an empty repo on github.com, then:
   git remote add origin https://github.com/<you>/geobench-copilot.git
   git push -u origin main
   ```
   The `.gitignore` keeps out the venv, runs, uploads, and `.env`.

3. **Create the service on Render** (https://render.com):
   - **New → Blueprint**, connect the repo → Render reads `render.yaml` and
     configures everything (build + start command, Python version).
   - *(Or* New → Web Service, and set it manually:*
     - Build: `pip install -r requirements.txt`
     - Start: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`*)*

4. **Add the two secrets** in the Render dashboard → the service → *Environment*:
   - `ANTHROPIC_API_KEY` = your new key
   - `APP_PASSWORD` = a password you pick (anyone you share the link with needs it)
   - *(optional)* `APP_USERNAME` (defaults to `geobench`)

5. Render **builds and starts it** (~3–5 min; watch the deploy log). When it's
   live, open the `*.onrender.com` URL → your browser prompts for the password →
   you're in, with a real shareable link.

## Updating

- New experiment results: `bash backend/scripts/bundle_results.sh` to refresh the
  bundled `backend/data/all_results.csv`, then `git commit` + `git push`. Render
  auto-redeploys on push. The "benchmark results as of …" badge shows the date.
- Code changes: commit + push; Render redeploys automatically.

## Notes / limits

- **Free tier sleeps when idle** — the first request after a quiet spell takes
  ~30–60 s to wake. Fine for a demo; upgrade the plan for always-on.
- **Snapshot data.** The hosted rankings use the bundled `all_results.csv` at
  deploy time — it does NOT live-update from your Mac's running experiments.
  Re-bundle + push to refresh (the badge shows how stale it is).
- **In-memory runs.** `RunManager` state is lost on restart/sleep. Fine for
  demos, not for real concurrent traffic.
- **512 MB RAM** on free tier — plenty for the app, but very large uploads could
  strain it; the 500 MB upload cap still applies.

*(The `Dockerfile` and HF-Space files are left in the repo as an alternative
path — Render ignores them.)*
