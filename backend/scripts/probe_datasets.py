"""Generate LOO-usable results for new datasets under the SAME protocol as
all_results.csv (frozen backbone + linear/kNN-5 probe).

WHY this exists: you cannot calibrate the fit weights against GeoBench-2 /
PANGAEA leaderboard numbers directly — those are fine-tuned with task-specific
decoders + HPO, a different transfer regime from your frozen linear probe. The
only apples-to-apples way to add a dataset to the LOO calibration is to run YOUR
protocol (these 59 backbones, linear/kNN-5) over it. This script orchestrates
exactly that by driving the real torchgeo-bench CLI, then the rows land in
all_results.csv and scripts/calibrate_weights.py picks them up automatically.

PREREQUISITES (this is a GPU compute job — it does NOT run in this sandbox):
  * run inside the torchgeo-bench repo env (config.TGB_REPO), torch available;
  * the target datasets downloaded/registered in torchgeo-bench;
  * a GPU (frozen-feature extraction over 59 backbones is the cost).

CAVEAT — segmentation: the current all_results.csv is entirely CLASSIFICATION
(linear/kNN on pooled features). The new GeoBench-2 datasets added to the
registry are SEGMENTATION, which needs a dense probe head that torchgeo-bench
may not expose the same way. Verify the harness supports a segmentation linear
probe before trusting these rows; otherwise start with classification datasets,
where the protocol is known-good.

Usage:
    uv run python scripts/probe_datasets.py                 # dry-run: print the plan
    uv run python scripts/probe_datasets.py --datasets cashew sa_crop_type
    uv run python scripts/probe_datasets.py --run           # actually execute (GPU)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from tools import results as R  # noqa: E402

# GeoBench-2 datasets added to the registry that have no result rows yet.
DEFAULT_TARGETS = ["cashew", "sa_crop_type", "nz_cattle", "neontree", "chesapeake", "pv4ger_seg"]

# GeoBench-v2 loaders return a local closure (_V2Dataset.get_dataset.<locals>.chained)
# that can't be pickled to DataLoader workers under the macOS/py3.13 'spawn' start
# method. Rather than kill parallelism with num_workers=0 (which makes the large
# .tortilla files pathologically slow — single-process, I/O-bound, effectively
# wedged), we run these under the 'fork' start method (scripts/_forkctx on
# PYTHONPATH): fork inherits memory instead of pickling, so workers start fine AND
# loading stays parallel/fast. The m-* (GeoBench-v1) datasets are unaffected and
# keep the platform default.
_V2_FORK = {"benv2", "forestnet", "so2sat", "treesatai", "cloudsen12",
            "dynamic_earthnet", "pastis", "caffe", "flair2", "fotw",
            "kuro_siwo", "spacenet2", "spacenet7", "burn_scars"}
_FORKCTX_DIR = Path(__file__).resolve().parent / "_forkctx"


def _benchmarked_backbones() -> list[str]:
    """The exact backbones already in all_results.csv — reproduce the protocol
    parity (same 59 models) rather than every model in models.json."""
    df = pd.read_csv(config.TGB_RESULTS_CSV, usecols=["method", "name"])
    names = sorted(df[df["method"] == "linear"]["name"].unique())
    paths = R.model_paths()
    return [n for n in names if n in paths]


def _cost_tables():
    """Per-backbone median throughput (samples/s) and per-dataset chip count,
    both read from the profile rows already in all_results.csv. Throughput is
    the best wall-clock predictor for a frozen-feature run: t ~= chips/thru."""
    df = pd.read_csv(config.TGB_RESULTS_CSV)
    prof = df[df["method"] == "profile"]
    thr = prof[prof["metric_name"] == "throughput_samples_per_sec"].groupby("name")["metric_value"].median()
    chips = prof.groupby("dataset")[["n_train", "n_val", "n_test"]].median().sum(axis=1)
    return thr.to_dict(), chips.to_dict()


def _est_seconds(dataset, backbone, thr, chips) -> float | None:
    n, t = chips.get(dataset), thr.get(backbone)
    if not n or not t:
        return None
    return n / t * 1.3  # +30% for probe fit/eval beyond feature extraction


def _missing_cells() -> list[tuple[str, str]]:
    """(dataset, backbone) linear cells absent from all_results.csv, over the
    datasets that already have SOME linear rows × the benchmarked backbones."""
    df = pd.read_csv(config.TGB_RESULTS_CSV, usecols=["method", "dataset", "name"])
    lin = df[df["method"] == "linear"]
    ds = sorted(lin["dataset"].unique())
    bk = _benchmarked_backbones()
    have = set(map(tuple, lin[["dataset", "name"]].drop_duplicates().values))
    return [(d, b) for d in ds for b in bk if (d, b) not in have]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["missing", "new"], default="missing",
                    help="missing = fill gaps in existing result-bearing datasets, "
                         "shortest-first; new = probe brand-new datasets (--datasets)")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_TARGETS,
                    help="target datasets for --mode new")
    ap.add_argument("--bands", default="rgb")
    ap.add_argument("--normalization", default="bandspec_zscore")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=None, help="run only the N shortest")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                    help="compute device; auto = mps on macOS else cuda")
    ap.add_argument("--exclude-backbone", nargs="+", default=[], metavar="SUBSTR",
                    help="skip backbones whose name contains any of these substrings, "
                         "e.g. --exclude-backbone base large (the heavy models that are "
                         "impractical on MPS)")
    ap.add_argument("--run", action="store_true", help="execute (needs GPU + torchgeo-bench env)")
    args = ap.parse_args()

    import sys as _sys
    device = ("mps" if _sys.platform == "darwin" else "cuda") if args.device == "auto" else args.device
    # torchgeo-bench defaults to device=cuda:0; on Apple silicon that hard-fails
    # ("Torch not compiled with CUDA enabled"), so override + keep KNN on CPU
    # (faiss-gpu isn't available for MPS).
    dev_flags = "" if device == "cuda" else f" device={device} eval.knn_device=cpu"

    thr, chips = _cost_tables()
    if args.mode == "missing":
        cells = _missing_cells()
        # optional --datasets filter: scope a run to specific (e.g. already-
        # staged) datasets, so you can skip ones whose data is still downloading.
        if args.datasets != DEFAULT_TARGETS:
            want = set(args.datasets)
            cells = [(d, b) for d, b in cells if d in want]
        # shortest-first; cells whose backbone was never profiled have unknown
        # cost and sort last (huge sentinel) so measured-cost runs go first.
        cells.sort(key=lambda db: _est_seconds(db[0], db[1], thr, chips) or 1e12)
    else:
        cells = [(d, b) for d in args.datasets for b in _benchmarked_backbones()]

    if args.exclude_backbone:
        subs = args.exclude_backbone
        before = len(cells)
        cells = [(d, b) for d, b in cells if not any(s in b for s in subs)]
        print(f"excluded {before - len(cells)} cell(s) whose backbone matches {subs}")

    plan = []  # (est_s, ds, bb, cmd)
    for ds, bb in cells:
        cmd = R.build_run_command(bb, ds, args.bands, args.normalization, args.image_size)
        if cmd:
            plan.append((_est_seconds(ds, bb, thr, chips), ds, bb, cmd + dev_flags))
    if args.limit:
        plan = plan[: args.limit]

    known = [p for p in plan if p[0] is not None]
    tot = sum(p[0] for p in known)
    print(f"mode={args.mode}: {len(plan)} runs "
          f"({len(known)} with measured cost, {len(plan) - len(known)} unknown-cost)")
    print(f"est. total on the reference GPU: ~{tot/60:.0f} min (~{tot/3600:.1f} h); "
          f"your hardware differs (MPS is slower).")
    print(f"torchgeo-bench repo: {config.TGB_REPO}\n")

    if not args.run:
        print("DRY RUN — shortest 5 (pass --run to execute; --limit N to cap):\n")
        for est, ds, bb, cmd in plan[:5]:
            tag = f"~{est:.0f}s" if est else "cost unknown"
            print(f"# [{tag}] {ds} x {bb}\n{cmd}\n")
        print(f"... and {max(0, len(plan) - 5)} more.")
        print("\nEach completed run appends a row to all_results.csv; then:")
        print("  uv run python scripts/calibrate_weights.py --multi")
        return

    if not config.TGB_REPO.exists():
        sys.exit(f"torchgeo-bench repo not found at {config.TGB_REPO} — set TGB_REPO")
    # the torchgeo-bench CLI lives in the repo's own venv; put it on PATH so this
    # works whether or not that venv is already activated in the caller's shell.
    import os
    env = dict(os.environ)
    venv_bin = config.TGB_REPO / ".venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    failures = 0
    for i, (est, ds, bb, cmd) in enumerate(plan, 1):
        tag = f"~{est:.0f}s" if est else "cost unknown"
        fork = ds in _V2_FORK
        print(f"[{i}/{len(plan)}] [{tag}] {ds} x {bb}{'  (fork)' if fork else ''}", flush=True)
        run_env = env
        if fork:
            # v2 datasets: run under the 'fork' start method so their unpicklable
            # loader closure works with parallel DataLoader workers (see _forkctx).
            run_env = dict(env)
            run_env["PYTHONPATH"] = f"{_FORKCTX_DIR}:{run_env.get('PYTHONPATH', '')}"
        proc = subprocess.run(cmd, shell=True, cwd=str(config.TGB_REPO), env=run_env)
        if proc.returncode != 0:
            failures += 1
            print(f"  !! exit {proc.returncode} — continuing")
    print(f"\ndone: {len(plan) - failures} ok, {failures} failed")
    print("re-run scripts/calibrate_weights.py --multi to fold the new rows into LOO.")


if __name__ == "__main__":
    main()
