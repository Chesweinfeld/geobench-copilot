"""Pull frozen-encoder evals from public geospatial leaderboards into a local,
protocol-tagged CSV so the matcher calibration can fold in many more datasets
than the 11 that have torchgeo-bench rows.

WHY a separate file (data/external_results.csv, never all_results.csv):
transfer numbers are only comparable *within one protocol*. torchgeo-bench is a
frozen backbone + linear/kNN probe; the GEO-Bench-2 leaderboard is a frozen (or
full-ft) backbone + a trained UNet/RCNN decoder, with its own backbone names.
Mixing the two backbone namespaces in one leave-one-out would be meaningless.
So we import each source under its own `source`/`protocol` tag and let
scripts/calibrate_weights.py run LOO *within* a source (--source external),
which validates the facet weights on an independent, real set of datasets.

WHAT it imports today: the GEO-Bench-2 leaderboard
(github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard). Only submissions whose
`frozen_or_full_ft == "frozen"` rows are kept — those are the protocol-honest,
frozen-feature evals. Full-fine-tune submissions are reported but skipped.

Scores are averaged over seeds per (dataset, backbone). RMSE datasets are
regression (lower is better) — we store `higher_is_better=False` and the
calibrator flips their sign so within-dataset backbone ranking stays consistent.

Usage:
    uv run python scripts/import_leaderboards.py            # fetch + write CSV
    uv run python scripts/import_leaderboards.py --dry-run  # summarize, no write
"""

from __future__ import annotations

import argparse
import io
import json
import ssl
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from tools import matching as M  # noqa: E402

# Same CA-bundle story as config.py: this Python/OpenSSL can't verify GitHub's
# chain with the system store, so pin certifi's bundle for our HTTPS reads.
try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = None

# GEO-Bench-2 CSV columns -> our normalized schema.
_GB2_METRIC_HIGHER_IS_BETTER = {
    "Multilabel_F1_Score": True,
    "Multiclass_Jaccard_Index": True,   # mIoU
    "Overall_Accuracy": True,
    "test_test_map": True,              # detection mAP
    "test_test_segm_map": True,
    "RMSE": False,                      # regression — lower is better
}
_UA = {"User-Agent": "geobench-copilot-importer"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as r:  # noqa: S310 (fixed https hosts)
        return r.read()


def _gh_api(path: str) -> list[dict]:
    owner, repo, ref = config.GB2_LEADERBOARD_OWNER, config.GB2_LEADERBOARD_REPO, config.GB2_LEADERBOARD_REF
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    return json.loads(_get(url))


def _fetch_gb2() -> pd.DataFrame:
    """Return normalized rows for every FROZEN GEO-Bench-2 submission."""
    entries = _gh_api("results")
    submissions = [e for e in entries if e["type"] == "dir"]
    print(f"GEO-Bench-2 leaderboard: {len(submissions)} submission(s) found")

    frames, n_frozen, n_ft = [], 0, 0
    for sub in submissions:
        raw = sub["url"].split("?")[0]  # contents url; children listed via api
        try:
            children = _gh_api(f"results/{sub['name']}")
        except Exception as e:  # noqa: BLE001
            print(f"  ! {sub['name']}: listing failed ({e}) — skipping")
            continue
        csv_entry = next((c for c in children if c["name"] == "results_and_parameters.csv"), None)
        if not csv_entry:
            continue
        df = pd.read_csv(io.BytesIO(_get(csv_entry["download_url"])))
        ft = set(df.get("frozen_or_full_ft", pd.Series(dtype=str)).dropna().unique())
        if "frozen" not in ft:
            n_ft += 1
            print(f"  - {sub['name'][:8]}: {sorted(ft) or ['unknown']} — not frozen, skipped")
            continue
        n_frozen += 1
        fz = df[df["frozen_or_full_ft"] == "frozen"].copy()
        # average over seeds per (dataset, backbone, metric)
        agg = (
            fz.groupby(["dataset", "backbone", "Metric"], as_index=False)["test metric"]
            .mean()
        )
        agg = agg.rename(columns={"backbone": "name", "Metric": "metric_name",
                                  "test metric": "metric_value"})
        agg["higher_is_better"] = agg["metric_name"].map(
            lambda m: _GB2_METRIC_HIGHER_IS_BETTER.get(m, True)
        )
        agg["source"] = "geobench2"
        agg["protocol"] = "frozen_unet"
        agg["method"] = "frozen"
        agg["bands"] = "default"
        agg["normalization"] = "gb2_default"
        agg["submission"] = sub["name"]
        frames.append(agg)
        print(f"  + {sub['name'][:8]}: frozen — {agg['dataset'].nunique()} datasets, "
              f"{agg['name'].nunique()} backbones, {len(agg)} rows")

    print(f"\nsummary: {n_frozen} frozen submission(s) kept, {n_ft} full-ft skipped")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_pangaea() -> pd.DataFrame:
    """PANGAEA (arXiv:2412.04204) Table 5 frozen-encoder main results, checked
    into data/pangaea_raw.csv (the leaderboard site is down and the repo ships
    no CSV, so the verified paper table is committed as static data — see the
    file header for the extraction/verification provenance). Its own backbone
    namespace + frozen-encoder-with-UNet-decoder protocol => a separate corpus."""
    if not config.PANGAEA_RAW_CSV.exists():
        print(f"  (no {config.PANGAEA_RAW_CSV.name} — skipping PANGAEA)")
        return pd.DataFrame()
    raw = pd.read_csv(config.PANGAEA_RAW_CSV, comment="#")
    raw = raw.rename(columns={"backbone": "name"})
    raw["higher_is_better"] = raw["higher_is_better"].astype(str).str.lower().eq("true")
    raw["source"] = "pangaea"
    raw["protocol"] = "frozen_unet"
    raw["method"] = "frozen"
    raw["bands"] = "default"
    raw["normalization"] = "pangaea_default"
    raw["submission"] = "arxiv_2412.04204_table5"
    print(f"PANGAEA: {raw['dataset'].nunique()} datasets, {raw['name'].nunique()} backbones, {len(raw)} rows")
    return raw


_COLS = ["source", "protocol", "method", "dataset", "name", "metric_name",
         "metric_value", "higher_is_better", "bands", "normalization", "submission"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="summarize, don't write the CSV")
    args = ap.parse_args()

    parts = [df for df in (_fetch_gb2(), _load_pangaea()) if not df.empty]
    if not parts:
        sys.exit("no rows imported — nothing written.")
    df = pd.concat(parts, ignore_index=True)

    # per-source overlap with the registry / usable-fold counts
    reg = set(M.registry())
    print()
    for src, g in df.groupby("source"):
        in_reg = sorted(set(g["dataset"]) & reg)
        not_in_reg = sorted(set(g["dataset"]) - reg)
        usable = g.groupby("dataset")["name"].nunique().loc[lambda s: s >= 5].index
        usable_in_reg = sorted(set(usable) & reg)
        print(f"[{src}] {g['dataset'].nunique()} datasets; "
              f"{len(usable_in_reg)} usable LOO folds (>=5 backbones AND in registry): "
              f"{', '.join(usable_in_reg)}")
        if not_in_reg:
            print(f"    not yet in registry: {', '.join(not_in_reg)}")
            print("    -> add to matching.py (META/DOMAINS/DESCRIPTIONS) to unlock as folds.")

    if args.dry_run:
        print("\n[dry-run] not writing.")
        return

    df[_COLS].to_csv(config.EXTERNAL_RESULTS_CSV, index=False)
    print(f"\nwrote {len(df)} rows -> {config.EXTERNAL_RESULTS_CSV}")
    print("next: uv run python scripts/calibrate_weights.py --source pangaea   (or --multi)")


if __name__ == "__main__":
    main()
