"""Environment-driven configuration for the GeoBench Copilot backend."""

import os
from pathlib import Path

# GDAL's bundled libcurl (common with conda/miniforge installs on macOS)
# often can't find a working CA bundle, so HTTPS reads of remote COG assets
# (the satellite auto-fetch path) fail with "unable to get local issuer
# certificate" — even when CURL_CA_BUNDLE/GDAL_CURL_CA_BUNDLE are already
# *set*, because conda's activate scripts often point them at a bundle that
# doesn't actually verify against these hosts. So: only trust an existing
# value if that file really exists on disk; otherwise (unset OR pointing at
# a missing/bad path) fall back to Python's certifi bundle, which is known
# to work. Best effort — skipped entirely if certifi isn't installed.
try:
    import certifi

    _CERTIFI_BUNDLE = certifi.where()
    for _var in ("CURL_CA_BUNDLE", "GDAL_CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        _current = os.environ.get(_var)
        if not _current or not Path(_current).is_file():
            os.environ[_var] = _CERTIFI_BUNDLE
except ImportError:
    pass

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"

TGB_REPO = Path(
    os.environ.get(
        "TGB_REPO",
        "/Users/chesapeake/torchgeo-bench/.claude/worktrees/cranky-chaum-ec1924",
    )
)
# Results source resolution, in priority order:
#   1. TGB_RESULTS_CSV env override (explicit)
#   2. the live torchgeo-bench repo, if present (local dev — hot-reloads as
#      experiments append rows)
#   3. a bundled snapshot at data/all_results.csv (hosted deploy, where the repo
#      isn't checked out — refresh it with scripts/bundle_results.sh)
_BUNDLED_RESULTS = DATA_DIR / "all_results.csv"
_LIVE_RESULTS = TGB_REPO / "results" / "all_results.csv"
TGB_RESULTS_CSV = Path(
    os.environ.get(
        "TGB_RESULTS_CSV",
        str(_LIVE_RESULTS if _LIVE_RESULTS.exists() else _BUNDLED_RESULTS),
    )
)

GEOBENCH_MODEL = os.environ.get("GEOBENCH_MODEL", "claude-sonnet-5")

# Optional HTTP Basic-Auth gate for hosted deploys. When APP_PASSWORD is set
# (a host secret), every request must supply it — this protects the Anthropic
# bill on a public link. Unset (local dev) = no gate. APP_USERNAME defaults to
# "geobench"; only the password really matters.
APP_PASSWORD = os.environ.get("APP_PASSWORD") or None
APP_USERNAME = os.environ.get("APP_USERNAME", "geobench")
# Hosts like Hugging Face Spaces inject the port to bind via $PORT (default 7860).
PORT = int(os.environ.get("PORT", "8000"))

# Directory holding the DC frontend (served statically, same-origin).
FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR", str(BACKEND_DIR.parent)))
FRONTEND_PAGE = "GeoBench Copilot.dc.html"

REGISTRY_JSON = DATA_DIR / "registry.json"
MODELS_JSON = DATA_DIR / "models.json"

# External evals imported from public leaderboards (scripts/import_leaderboards.py).
# Kept OUT of the canonical TGB_RESULTS_CSV so protocol namespaces never mix:
# each row carries `source` + `protocol` and calibration folds them in only when
# explicitly asked (--source external). Today's source is the GEO-Bench-2
# leaderboard's frozen-encoder submissions.
EXTERNAL_RESULTS_CSV = DATA_DIR / "external_results.csv"
# Static verified extract of PANGAEA (arXiv:2412.04204) Table 5 frozen results —
# committed because its leaderboard site is down and its repo ships no CSV.
PANGAEA_RAW_CSV = DATA_DIR / "pangaea_raw.csv"
GB2_LEADERBOARD_OWNER = os.environ.get("GB2_LEADERBOARD_OWNER", "The-AI-Alliance")
GB2_LEADERBOARD_REPO = os.environ.get("GB2_LEADERBOARD_REPO", "GEO-Bench-2-Leaderboard")
GB2_LEADERBOARD_REF = os.environ.get("GB2_LEADERBOARD_REF", "main")

# Upload limits
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB zip
MAX_ARCHIVE_ENTRIES = 20_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB extracted

# Vector sidecar files (GeoJSON/KML) are supplementary metadata only — cap
# how many we bother parsing so a huge or malformed batch can't stall a run.
MAX_VECTOR_FILES_PARSED = 5

# Satellite auto-fetch: when an archive has vector geometries but no raster
# images, pull real Sentinel-2 chips for those AOIs from Microsoft Planetary
# Computer's public STAC API instead of failing the run.
STAC_API_URL = os.environ.get(
    "STAC_API_URL", "https://planetarycomputer.microsoft.com/api/stac/v1"
)
STAC_COLLECTION = os.environ.get("STAC_COLLECTION", "sentinel-2-l2a")
STAC_DATETIME_RANGE = os.environ.get("STAC_DATETIME_RANGE", "2023-01-01/2025-12-31")
STAC_MAX_CLOUD_COVER = float(os.environ.get("STAC_MAX_CLOUD_COVER", "20"))
FETCH_AOI_HALF_SIDE_M = float(os.environ.get("FETCH_AOI_HALF_SIDE_M", "320"))  # ~640m box for point-like geometries
FETCH_AOI_PAD_FRAC = float(os.environ.get("FETCH_AOI_PAD_FRAC", "0.15"))  # extra margin around a polygon's own extent
FETCH_AOI_MAX_HALF_SIDE_M = float(os.environ.get("FETCH_AOI_MAX_HALF_SIDE_M", "5000"))  # cap: 10km box
FETCH_CHIP_PX = int(os.environ.get("FETCH_CHIP_PX", "64"))
MAX_FETCH_CHIPS = int(os.environ.get("MAX_FETCH_CHIPS", "100"))
# When no single scene is under STAC_MAX_CLOUD_COVER, blend up to this many
# of the least-cloudy scenes into a cloud-masked median composite instead.
STAC_COMPOSITE_MAX_ITEMS = int(os.environ.get("STAC_COMPOSITE_MAX_ITEMS", "6"))

# Grouped fetch: neighbouring geometries usually share a Sentinel-2 scene
# (favela/settlement layers cluster hard), so bin AOIs into a grid and do one
# catalog search + one windowed read per cell, cropping every chip locally,
# instead of a search + read per geometry. Turns O(chips) network ops into
# O(regions). Set FETCH_GROUP_ENABLED=0 to force the per-geometry path.
FETCH_GROUP_ENABLED = os.environ.get("FETCH_GROUP_ENABLED", "1") not in ("0", "false", "False", "")
FETCH_GROUP_CELL_KM = float(os.environ.get("FETCH_GROUP_CELL_KM", "5"))  # grid cell for grouping
FETCH_TARGET_GSD_M = float(os.environ.get("FETCH_TARGET_GSD_M", "10"))  # S2 RGB native resolution
FETCH_GROUP_MAX_READ_PX = int(os.environ.get("FETCH_GROUP_MAX_READ_PX", "2048"))  # cap union read dim (memory)
# Regions are I/O-bound (STAC search + COG reads over HTTP); fetch them on a
# small thread pool. GDAL releases the GIL during reads, so threads overlap
# network latency. Set to 1 to force sequential.
FETCH_MAX_WORKERS = int(os.environ.get("FETCH_MAX_WORKERS", "8"))
# Train/val/test split for auto-fetched chips is assigned by SPATIAL BLOCK, not
# per chip, so overlapping neighbours never straddle splits (fetched AOIs
# overlap heavily — a random split leaks). Whole blocks of this size go to one
# split, deterministically (stable hash), in these ratios.
FETCH_SPLIT_BLOCK_KM = float(os.environ.get("FETCH_SPLIT_BLOCK_KM", "5"))
FETCH_SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / val / test

# Agent limits
AGENT_TIMEOUT_S = 20 * 60
AGENT_MAX_ITERATIONS = 40

# Where per-run uploads/extractions live
RUNS_DIR = Path(
    os.environ.get("GEOBENCH_RUNS_DIR", str(BACKEND_DIR / ".runs"))
)
