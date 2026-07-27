"""One-shot registry extraction from torchgeo-bench.

Run this under the torchgeo-bench venv (which has torch installed) — the
backend itself never imports torchgeo_bench:

    /Users/chesapeake/torchgeo-bench/.venv/bin/python scripts/extract_registry.py \
        --repo /path/to/torchgeo-bench --out data/

Produces:
    data/registry.json  — per-dataset metadata + full BandSpec lists
    data/models.json    — CSV model name -> hydra `model=` config path
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path


def extract_registry(repo: Path) -> dict:
    sys.path.insert(0, str(repo / "src"))
    from torchgeo_bench.datasets.loading import _REGISTRY  # noqa: PLC0415

    datasets = {}
    for name, cls in _REGISTRY.items():
        bands = [dataclasses.asdict(b) for b in cls.bands]
        datasets[name] = {
            "name": cls.name,
            "task": cls.task,
            "num_classes": cls.num_classes,
            "multilabel": bool(getattr(cls, "multilabel", False)),
            "rgb_bands": list(cls.rgb_bands),
            "split_sizes": dict(cls.split_sizes),
            "num_channels": len(bands),
            "sensors": sorted({b["sensor"] for b in bands}),
            "bands": bands,
        }
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": str(repo),
        "datasets": datasets,
    }


def extract_models(repo: Path) -> dict:
    """Map each model config's `name:` to its hydra `model=` group path."""
    import yaml  # available in the torchgeo-bench venv (hydra dep)  # noqa: PLC0415

    conf_dir = repo / "src" / "torchgeo_bench" / "conf" / "model"
    mapping = {}
    for path in sorted(conf_dir.rglob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(cfg, dict):
            continue
        name = cfg.get("name")
        if not name:
            continue
        rel = path.relative_to(conf_dir).with_suffix("")
        mapping[str(name)] = str(rel)
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    registry = extract_registry(args.repo)
    (args.out / "registry.json").write_text(json.dumps(registry, indent=1))
    print(f"registry.json: {len(registry['datasets'])} datasets")

    models = extract_models(args.repo)
    (args.out / "models.json").write_text(json.dumps(models, indent=1, sort_keys=True))
    print(f"models.json: {len(models)} model configs")


if __name__ == "__main__":
    main()
