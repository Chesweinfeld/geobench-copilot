"""Query layer over torchgeo-bench's results/all_results.csv.

Loads the CSV once, joins accuracy rows (method in {linear, knn5}) with
profile rows (gflops, params_m, ...) pivoted wide — the same join
experiments/scripts/regen_results_explorer.py uses.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

import config

ACC_METHODS = ("linear", "knn5")
PROFILE_METRICS = ("gflops", "params_m", "throughput_samples_per_sec")


def _mtime(path: Path) -> int | None:
    """File modification time in ns, or None if it doesn't exist yet."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


# mtime-keyed cache: hold the parsed tables until all_results.csv changes on
# disk, so newly-appended experiment rows are picked up WITHOUT a server restart
# (the file is appended to by scripts/probe_datasets.py runs). {"key", "val"}.
_TABLES: dict = {}


def _tables() -> dict:
    key = _mtime(config.TGB_RESULTS_CSV)
    if _TABLES.get("key") != key:
        _TABLES["val"] = _build_tables()
        _TABLES["key"] = key
    return _TABLES["val"]


def _build_tables() -> dict:
    df = pd.read_csv(config.TGB_RESULTS_CSV)
    acc = df[df["method"].isin(ACC_METHODS)][
        ["dataset", "method", "metric_name", "metric_value", "ci_lower", "ci_upper",
         "name", "normalization", "bands", "image_size"]
    ].copy()
    prof = df[df["method"] == "profile"]
    prof_wide = (
        prof.pivot_table(
            index=["name", "dataset", "bands", "normalization"],
            columns="metric_name",
            values="metric_value",
            aggfunc="median",
        )
        .reset_index()
    )
    return {"acc": acc, "prof": prof_wide}


@lru_cache(maxsize=1)
def model_paths() -> dict[str, str]:
    return json.loads(config.MODELS_JSON.read_text())


def datasets_with_results() -> set[str]:
    return set(_tables()["acc"]["dataset"].unique())


def query_results(
    datasets: list[str],
    bands: str = "rgb",
    normalization: str = "bandspec_zscore",
    preferred_metric: str | None = None,
    top_n: int = 8,
) -> dict:
    """Rank models by mean linear-probe score over the matched datasets.

    If preferred_metric is given and at least one matched dataset actually has
    rows logged under that metric_name, ranking uses it — real numbers only,
    never synthesized. Otherwise falls back to whichever metric is most common
    among the requested datasets, and metric_note explains what happened so the
    caller can be honest about it rather than claiming an unavailable metric
    was used.
    """
    t = _tables()
    acc = t["acc"]
    known = datasets_with_results()
    requested = [d for d in datasets if d in known]
    skipped = [d for d in datasets if d not in known]
    if not requested:
        return {
            "metric": None,
            "ranked": [],
            "skipped_datasets": skipped,
            "note": "none of the requested datasets have accuracy results in all_results.csv",
        }

    sub_all = acc[
        acc["dataset"].isin(requested)
        & (acc["bands"] == bands)
        & (acc["normalization"] == normalization)
    ]
    available_metrics = sorted(sub_all["metric_name"].unique().tolist())
    metric_note: str | None = None

    if preferred_metric and preferred_metric in available_metrics:
        metric = preferred_metric
        used = sorted(sub_all.loc[sub_all["metric_name"] == metric, "dataset"].unique().tolist())
        dropped_metric = [d for d in requested if d not in used]
        if dropped_metric:
            metric_note = (
                f"Ranked by {metric} as requested; {len(dropped_metric)} matched "
                f"dataset(s) don't report it and were excluded: {dropped_metric}."
            )
    else:
        # enforce a single metric — never average accuracy with micro_mAP
        metric_by_ds = sub_all.groupby("dataset")["metric_name"].agg(lambda s: s.mode().iat[0])
        metric_counts = metric_by_ds.value_counts()
        metric = metric_counts.index[0]
        dropped_metric = [d for d, m in metric_by_ds.items() if m != metric]
        used = [d for d in requested if d not in dropped_metric]
        if preferred_metric and preferred_metric not in available_metrics:
            metric_note = (
                f"Requested metric {preferred_metric!r} isn't reported for any matched "
                f"dataset (available here: {available_metrics}) — ranked by {metric} instead."
            )

    sub = sub_all[sub_all["dataset"].isin(used) & (sub_all["metric_name"] == metric)]

    lin = sub[sub["method"] == "linear"]
    knn = sub[sub["method"] == "knn5"]

    lin_mean = lin.groupby("name")["metric_value"].mean()
    lin_cover = lin.groupby("name")["dataset"].nunique()
    knn_mean = knn.groupby("name")["metric_value"].mean()

    prof = t["prof"]
    prof_sub = prof[prof["dataset"].isin(used) & (prof["bands"] == bands)]
    prof_agg = prof_sub.groupby("name")[
        [c for c in PROFILE_METRICS if c in prof_sub.columns]
    ].median()

    min_cover = min(2, len(used))
    ranked = []
    for name, mean_v in lin_mean.sort_values(ascending=False).items():
        per_ds = {
            d: round(float(v), 4)
            for d, v in lin[lin["name"] == name].groupby("dataset")["metric_value"].mean().items()
        }
        gflops = params = thr = None
        if name in prof_agg.index:
            row = prof_agg.loc[name]
            gflops = None if pd.isna(row.get("gflops")) else round(float(row["gflops"]), 1)
            params = None if pd.isna(row.get("params_m")) else round(float(row["params_m"]), 1)
            thr = (
                None
                if pd.isna(row.get("throughput_samples_per_sec"))
                else round(float(row["throughput_samples_per_sec"]), 1)
            )
        ranked.append(
            {
                "name": name,
                "linear_mean": round(float(mean_v), 4),
                "linear_per_dataset": per_ds,
                "knn5_mean": round(float(knn_mean.get(name, float("nan"))), 4)
                if name in knn_mean
                else None,
                "coverage": int(lin_cover.get(name, 0)),
                "partial": int(lin_cover.get(name, 0)) < min_cover,
                "gflops": gflops,
                "params_m": params,
                "throughput": thr,
                "hydra_model_path": model_paths().get(name),
            }
        )

    full = [r for r in ranked if not r["partial"]]
    top = (full or ranked)[:top_n]
    leaderboard6 = [{"name": r["name"], "acc": r["linear_mean"]} for r in (full or ranked)[:6]]
    scatter6 = [
        {"name": r["name"], "gflops": r["gflops"], "acc": r["linear_mean"]}
        for r in (full or ranked)[:6]
        if r["gflops"] is not None
    ]
    budget = next((r["name"] for r in full if r["gflops"] is not None and r["gflops"] < 10), None)

    return {
        "metric": metric,
        "metric_note": metric_note,
        "available_metrics": available_metrics,
        "datasets_used": used,
        "datasets_dropped_wrong_metric": dropped_metric,
        "skipped_datasets": skipped,
        "bands": bands,
        "normalization": normalization,
        "ranked": top,
        "leaderboard6": leaderboard6,
        "scatter6": scatter6,
        "budget_pick": budget,
    }


def build_run_command(
    model_name: str,
    dataset_name: str,
    bands: str = "rgb",
    normalization: str = "bandspec_zscore",
    image_size: int = 224,
) -> str | None:
    """Emit a real torchgeo-bench command. Only real Hydra keys — there is
    no eval.metrics or report.format."""
    path = model_paths().get(model_name)
    if path is None:
        return None
    return (
        f"torchgeo-bench run model={path} \\\n"
        f"  'dataset.names=[{dataset_name}]' dataset.bands={bands} \\\n"
        f"  dataset.normalization={normalization} dataset.image_size={image_size} \\\n"
        f"  eval.knn_k=5"
    )


def estimate_minutes(n_chips: int, gflops: float | None, hw_factor: float = 1.0) -> int:
    """Illustrative wall-clock estimate for feature extraction + linear probe."""
    g = gflops if gflops else 20.0
    return max(1, round(n_chips / 1000 * g / 8 * hw_factor))
