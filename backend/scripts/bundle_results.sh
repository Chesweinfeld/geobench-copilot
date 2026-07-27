#!/usr/bin/env bash
# Refresh the bundled results snapshot (backend/data/all_results.csv) from the
# live torchgeo-bench repo, so a hosted deploy serves the latest experiments.
# Run this before committing/redeploying to the HF Space.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${TGB_RESULTS_CSV:-/Users/chesapeake/torchgeo-bench/.claude/worktrees/cranky-chaum-ec1924/results/all_results.csv}"
cp "$SRC" data/all_results.csv
echo "bundled $(wc -l < data/all_results.csv) rows -> backend/data/all_results.csv"
