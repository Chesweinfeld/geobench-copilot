"""Leave-one-dataset-out (LOO) calibration for the match_datasets facet weights.

Turns "are the fit weights good?" into a measured number. For each dataset that
has real backbone results, we hold it out as the "user task", let match_datasets
rank the *other* result-bearing datasets by sim, pick a backbone from the top-K
matched datasets, and measure REGRET — how much accuracy we lost versus the
backbone that was actually best on the held-out dataset.

Everything here reads all_results.csv + the registry; no torch, no GPU. It
evaluates whatever datasets have result rows, so it automatically covers
GeoBench-2 / PANGAEA datasets as soon as those rows are added to the CSV.

Usage:
    uv run python scripts/calibrate_weights.py            # eval current weights
    uv run python scripts/calibrate_weights.py --search   # + coarse weight search
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from tools import matching as M  # noqa: E402

TOP_K = 3  # how many matched datasets to pool a backbone recommendation from


# Each corpus is an independent, protocol-self-consistent eval set: its own
# backbone namespace + transfer regime. LOO runs WITHIN a corpus; the multi
# search asks which weights hold up ACROSS all of them (robustness), since any
# single small corpus overfits (11-15 folds). "tgb" == "tgb-linear" (kept as an
# alias for older invocations).
CORPORA = ("tgb-linear", "tgb-knn5", "geobench2", "pangaea")

# Corpora deliberately held OUT of the weight-setting objective (still evaluated
# and reported). PANGAEA: facet-matching cannot beat its no-matching baseline at
# any sane weighting — its task mix (marine debris, flood, building damage,
# biomass) doesn't follow "similar facets -> similar best backbone". Letting the
# minimax chase it wrecks the corpora where matching demonstrably works, so we
# treat it as a diagnostic, not a driver. (Verified 2026-07-17; revisit if its
# fold set changes.)
_VALIDATION_ONLY = {"pangaea"}
# imported-leaderboard corpora live together in EXTERNAL_RESULTS_CSV, one row per
# eval tagged with a `source` column; each is its own corpus here.
_EXTERNAL_SOURCES = {"geobench2", "pangaea"}
_ALIASES = {"tgb": "tgb-linear", "external": "geobench2"}


def _read_csv_stable(path, tries: int = 5) -> pd.DataFrame:
    """Read a CSV that may be appended to concurrently (e.g. all_results.csv
    while probe_datasets.py is running). If a read lands mid-append and hits a
    truncated final line, retry after a short pause — the write completes in
    milliseconds. Read-only, so it can never disturb the writer; this just makes
    calibrating on a live-growing file safe to run in parallel."""
    import time
    for i in range(tries):
        try:
            return pd.read_csv(path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            if i == tries - 1:
                raise
            time.sleep(0.3)


def _load_corpus_df(source: str) -> pd.DataFrame:
    source = _ALIASES.get(source, source)
    if source in _EXTERNAL_SOURCES:
        if not config.EXTERNAL_RESULTS_CSV.exists():
            sys.exit("no data/external_results.csv — run scripts/import_leaderboards.py first")
        df = _read_csv_stable(config.EXTERNAL_RESULTS_CSV)
        df = df[df["source"] == source]
        if df.empty:
            sys.exit(f"no rows for source={source!r} in {config.EXTERNAL_RESULTS_CSV.name}")
        if "higher_is_better" in df.columns:
            df["metric_value"] = df.apply(
                lambda r: r["metric_value"] if r["higher_is_better"] else -r["metric_value"], axis=1
            )
        return df
    if source in ("tgb-linear", "tgb-knn5"):
        method = "linear" if source == "tgb-linear" else "knn5"
        df = _read_csv_stable(config.TGB_RESULTS_CSV)
        return df[df["method"] == method]
    sys.exit(f"unknown corpus {source!r} — choose from {CORPORA}")


def _per_dataset_scores(source: str = "tgb-linear") -> dict[str, pd.Series]:
    """backbone -> score for each dataset, using the (bands, normalization,
    metric) slice with the widest backbone coverage. Within a dataset the
    metric is self-consistent, which is all we need for ranking backbones.

    Corpora (see CORPORA): tgb-linear / tgb-knn5 are the torchgeo-bench frozen
    linear- and kNN-5-probe protocols (all_results.csv); external is the
    imported public-leaderboard frozen evals (data/external_results.csv). Each
    lives in its OWN backbone namespace/protocol, so LOO runs within a corpus,
    never across. RMSE (regression) rows carry higher_is_better=False and are
    sign-flipped so a higher score always means a better backbone."""
    df = _load_corpus_df(source)
    out: dict[str, pd.Series] = {}
    for ds, g in df.groupby("dataset"):
        # pick the slice (bands, normalization, metric) with the most backbones
        best_key, best_n = None, -1
        for key, gg in g.groupby(["bands", "normalization", "metric_name"]):
            n = gg["name"].nunique()
            if n > best_n:
                best_key, best_n = key, n
        b, norm, metric = best_key
        sl = g[(g["bands"] == b) & (g["normalization"] == norm) & (g["metric_name"] == metric)]
        s = sl.groupby("name")["metric_value"].mean()
        if len(s) >= 5:
            out[ds] = s
    return out


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def _descriptors(ds: str) -> dict | None:
    """Build the match_datasets 'user task' descriptors for a dataset from the
    registry + the hand-authored scale/domain sidecars."""
    reg = M.registry().get(ds)
    if reg is None:
        return None
    gsd, scale = M.DATASET_META.get(ds, (None, "scene"))
    domains = sorted(M.DATASET_DOMAINS.get(ds, {"other"}))
    return dict(
        task_type=reg["task"],
        multilabel=reg["multilabel"],
        num_classes=reg["num_classes"],
        band_count=min(reg["num_channels"], 3) if reg.get("rgb_bands") else reg["num_channels"],
        sensor=reg["sensors"][0] if reg["sensors"] else "unknown",
        object_scale=scale,
        domain=domains[0],
        gsd_m=gsd,
    )


def _spearman(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return float("nan")
    ra = a[common].rank()
    rb = b[common].rank()
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def loo(scores: dict[str, pd.Series], weights: dict[str, float] | None = None) -> dict:
    """Run leave-one-out and return regret + diagnostics."""
    if weights is not None:
        M._WEIGHTS = weights  # matching reads this module global
    with_results = set(scores)
    regrets, base_regrets, sim_util_corr = [], [], []
    rows = []

    for held in scores:
        s_held = scores[held]
        desc = _descriptors(held)
        if desc is None:
            continue
        true_best = s_held.idxmax()

        scored = M.match_datasets(datasets_with_results=with_results, **desc)
        # Never pool across the regression / categorical divide: a regression
        # benchmark's backbone ranking doesn't transfer to a classification or
        # segmentation task (and vice versa), so exclude cross-regime candidates
        # before ranking. A lone regression dataset (e.g. biomassters) thus has
        # no same-regime peers and is skipped — you can't calibrate it here.
        held_reg = M.is_regression(held)
        cands = [d for d in scored if d["name"] in with_results and d["name"] != held
                 and M.is_regression(d["name"]) == held_reg]
        if len(cands) < TOP_K:
            continue
        pool_pool = [c for c in with_results if c != held and M.is_regression(c) == held_reg]

        # does higher sim track higher real transfer utility (rank-corr to held)?
        util = {c["name"]: _spearman(scores[c["name"]], s_held) for c in cands}
        sims = pd.Series({c["name"]: c["sim"] for c in cands})
        utils = pd.Series(util).dropna()
        if len(utils) >= 3:
            sim_util_corr.append(_spearman(sims, utils))

        # matched recommendation: pool top-K matched datasets' z-scored rankings
        topk = [c["name"] for c in cands[:TOP_K]]
        pooled = pd.concat([_zscore(scores[c]) for c in topk], axis=1).mean(axis=1)
        pred = pooled.reindex(s_held.index).dropna().idxmax()

        # baseline: pool ALL other same-regime datasets (no matching at all)
        allz = pd.concat([_zscore(scores[c]) for c in pool_pool], axis=1).mean(axis=1)
        base_pred = allz.reindex(s_held.index).dropna().idxmax()

        best = s_held.max()
        regrets.append(best - s_held[pred])
        base_regrets.append(best - s_held[base_pred])
        rows.append((held, round(best - s_held[pred], 4), round(best - s_held[base_pred], 4),
                     topk, pred, true_best))

    return dict(
        n=len(regrets),
        mean_regret=float(np.mean(regrets)) if regrets else float("nan"),
        baseline_regret=float(np.mean(base_regrets)) if base_regrets else float("nan"),
        sim_util_corr=float(np.nanmean(sim_util_corr)) if sim_util_corr else float("nan"),
        rows=rows,
    )


_KEYS = ["domain", "task", "bands", "label", "scale"]


def _grid(step: float = 0.05):
    """All weight vectors over _KEYS on a simplex (sum to 1) at the given step."""
    vals = [round(x * step, 2) for x in range(1, int(round(0.55 / step)) + 1)]
    for combo in itertools.product(vals, repeat=len(_KEYS)):
        if abs(sum(combo) - 1.0) <= 1e-9:
            yield dict(zip(_KEYS, combo))


def multi_search(step: float = 0.05, min_folds: int = 6) -> None:
    """Find weights that are robust ACROSS every corpus, not tuned to one.

    A single corpus (11-15 folds) overfits — its optimum contradicts the next
    corpus's. So we score each weight vector on all corpora and rank by the
    *worst* corpus, normalized by that corpus's own no-matching baseline
    (regret_c / baseline_c, scale-free: <1 means it still beats no matching
    there). Minimising the max ratio picks weights that never fail badly on any
    protocol, instead of winning big on one and losing on another.

    Corpora with < min_folds are VALIDATION-ONLY: too few folds to trust for
    setting weights (they'd let a handful of noisy datasets swing the objective),
    so they are reported but excluded from the min/mean the search optimises."""
    corpora = {c: _per_dataset_scores(c) for c in CORPORA}
    # actual usable folds = datasets that both have >=5 backbones AND resolve to
    # registry descriptors (loo skips the rest). Gate on THIS, not raw dataset
    # count, so a corpus with many out-of-registry datasets isn't over-credited.
    base_res = {c: loo(sc, dict(M._WEIGHTS)) for c, sc in corpora.items()}
    n_folds = {c: base_res[c]["n"] for c in CORPORA}
    obj = [c for c in CORPORA if n_folds[c] >= min_folds and c not in _VALIDATION_ONLY]
    for c in CORPORA:
        if c in obj:
            tag = "objective"
        elif c in _VALIDATION_ONLY:
            tag = "VALIDATION-ONLY (held out of objective by policy)"
        else:
            tag = f"VALIDATION-ONLY (<{min_folds} usable folds)"
        print(f"  {c:12s}: {n_folds[c]:2d} usable folds  [{tag}]")
    # baselines (no-matching) are weight-independent — computed once per corpus.
    base = {c: base_res[c]["baseline_regret"] for c in CORPORA}
    print("  baselines (no-matching regret): " +
          "  ".join(f"{c}={base[c]:.4f}" for c in CORPORA))
    print()

    def evaluate(w: dict) -> dict:
        per = {c: loo(sc, w)["mean_regret"] for c, sc in corpora.items()}
        ratio = {c: (per[c] / base[c] if base[c] > 0 else float("nan")) for c in CORPORA}
        # objective is driven only by well-powered corpora
        oclean = [ratio[c] for c in obj if ratio[c] == ratio[c]]
        return dict(per=per, ratio=ratio,
                    worst=max(oclean) if oclean else float("nan"),
                    mean=float(np.mean(oclean)) if oclean else float("nan"))

    named = {"CURRENT (_WEIGHTS)": dict(M._WEIGHTS)}
    if hasattr(M, "_WEIGHTS_PRIOR"):
        named["PRIOR (AHP)"] = dict(M._WEIGHTS_PRIOR)
    for label, w in named.items():
        e = evaluate(w)
        print(f"{label} {w}")
        print(f"  worst-corpus ratio={e['worst']:.3f}  mean ratio={e['mean']:.3f}  "
              + "  ".join(f"{c}={e['per'][c]:.4f}" for c in CORPORA))
    print()

    print(f"robust grid search over {len(CORPORA)} corpora (step {step})...")
    best_worst, best_mean = (float("inf"), None, None), (float("inf"), None, None)
    tried = 0
    for w in _grid(step):
        tried += 1
        e = evaluate(w)
        if e["worst"] < best_worst[0]:
            best_worst = (e["worst"], w, e)
        if e["mean"] < best_mean[0]:
            best_mean = (e["mean"], w, e)
    M._WEIGHTS = named["CURRENT (_WEIGHTS)"]  # restore (loo mutates the module global)
    print(f"  evaluated {tried} weight vectors\n")

    for tag, (score, w, e) in (("MINIMAX (best worst-corpus)", best_worst),
                               ("MEAN (best average)", best_mean)):
        print(f"{tag}: {w}")
        print(f"  worst-corpus ratio={e['worst']:.3f}  mean ratio={e['mean']:.3f}  "
              + "  ".join(f"{c}={e['per'][c]:.4f}" for c in CORPORA))
    print("\n  NOTE: still only ~11-15 folds per corpus — robust-across beats single-corpus")
    print("  overfitting, but treat as indicative. Set _WEIGHTS in tools/matching.py by hand.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="coarse grid search over weights")
    ap.add_argument("--multi", action="store_true",
                    help="robust search across ALL corpora (minimax over normalized regret)")
    ap.add_argument("--source", choices=[*CORPORA, "tgb", "external"], default="tgb-linear",
                    help="which corpus for single-source eval/search")
    args = ap.parse_args()

    if args.multi:
        multi_search()
        return

    scores = _per_dataset_scores(args.source)
    print(f"source: {args.source}")
    print(f"datasets with usable results: {len(scores)}")
    print("  " + ", ".join(sorted(scores)))
    print()

    current = dict(M._WEIGHTS)
    res = loo(scores, current)
    print(f"CURRENT weights {current}")
    print(f"  folds={res['n']}  mean regret={res['mean_regret']:.4f}  "
          f"(baseline no-matching={res['baseline_regret']:.4f})")
    print(f"  sim<->transfer-utility rank corr (want >0): {res['sim_util_corr']:+.3f}")
    print()
    print(f"  {'held-out':18s} {'regret':>7s} {'base':>7s}  matched(top3) -> pred")
    for held, reg, base, topk, pred, true in sorted(res["rows"], key=lambda r: -r[1]):
        print(f"  {held:18s} {reg:7.4f} {base:7.4f}  {'/'.join(topk):32s} -> {pred}  (true best {true})")

    if args.search:
        print("\ncoarse grid search (weights summing to 1, step 0.05)...")
        best = (res["mean_regret"], current)
        tried = 0
        for w in _grid():
            tried += 1
            r = loo(scores, w)
            if r["mean_regret"] < best[0]:
                best = (r["mean_regret"], w)
        M._WEIGHTS = current
        print(f"  evaluated {tried} weight vectors")
        print(f"  best mean regret={best[0]:.4f} at {best[1]}")
        print(f"  (current mean regret={res['mean_regret']:.4f})")
        print("  NOTE: only", len(scores), "LOO folds — treat as coarse/indicative, not a fit.")


if __name__ == "__main__":
    main()
