#!/usr/bin/env bash
# Start the GeoBench Copilot backend with the API key from .env
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" || "${ANTHROPIC_API_KEY}" == "sk-ant-REPLACE_ME" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set. Edit backend/.env and paste your key." >&2
  exit 1
fi

echo "Key loaded (…${ANTHROPIC_API_KEY: -6}). Starting server on http://localhost:8000/ …"
exec uv run uvicorn app:app --port 8000
