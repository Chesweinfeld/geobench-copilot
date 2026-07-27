"""Dataset matching: weighted facet scores against the torchgeo-bench registry.

Registry metadata comes from data/registry.json (dumped by
scripts/extract_registry.py); the backend never imports torchgeo_bench.
"""

from __future__ import annotations

import json
import math

import config

# Facts the registry doesn't carry: (gsd_m, object_scale).
# object_scale: "object" = discrete objects in chips, "scene" = whole-scene
# category, "landcover" = wall-to-wall land cover / land use.
DATASET_META: dict[str, tuple[float, str]] = {
    "m-pv4ger": (0.3, "object"),
    "m-brick-kiln": (10, "object"),
    "m-eurosat": (10, "landcover"),
    "eurosat": (10, "landcover"),
    "eurosat-spatial": (10, "landcover"),
    "m-forestnet": (15, "landcover"),
    "m-so2sat": (10, "scene"),
    "so2sat": (10, "scene"),
    "m-bigearthnet": (10, "landcover"),
    "benv2": (10, "landcover"),
    "treesatai": (0.2, "landcover"),
    "forestnet": (15, "landcover"),
    # segmentation datasets (no accuracy rows yet, still matchable)
    "caffe": (7, "object"),
    "burn_scars": (30, "landcover"),
    "cloudsen12": (10, "scene"),
    "dynamic_earthnet": (3, "landcover"),
    "flair2": (0.2, "landcover"),
    "fotw": (10, "object"),
    "kuro_siwo": (10, "landcover"),
    "pastis": (10, "landcover"),
    "spacenet2": (0.3, "object"),
    "spacenet7": (4, "object"),
    # GeoBench-2 segmentation datasets (matchable; no linear-probe rows yet)
    "cashew": (10, "landcover"),
    "sa_crop_type": (10, "landcover"),
    "nz_cattle": (0.1, "object"),
    "neontree": (0.1, "object"),
    "chesapeake": (1, "landcover"),
    "pv4ger_seg": (0.1, "object"),
    "substations": (10, "object"),
    "everwatch": (0.1, "object"),
    # External benchmarks (PANGAEA / GEO-Bench-2) added to the registry so their
    # imported frozen evals become calibration folds (scripts/import_leaderboards.py).
    "biomassters": (10, "landcover"),
    "mados": (10, "landcover"),
    "sen1floods11": (10, "landcover"),
    "xview2": (0.5, "object"),
    "fbp": (4, "landcover"),
    "cropmap": (10, "landcover"),
    "ai4smallfarms": (10, "landcover"),
}

# Default GSD by sensor family when the upload carries no georeferencing.
SENSOR_DEFAULT_GSD = {"s2": 10.0, "aerial": 0.5, "sar": 10.0}

# Optical sensor families — any two of these are spectrally comparable enough to
# earn partial band credit (vs SAR/other, which are not).
_OPTICAL_SENSORS = {"s2", "aerial", "landsat", "planet", "worldview", "maxar", "gaofen2", "spot"}

# Best-effort thematic tag(s) per registry dataset — what the benchmark is
# actually *about*, independent of task/bands/scale. Lets "detecting wind
# farms" (energy_infrastructure) prefer m-pv4ger over an equally
# band/scale-matched but thematically unrelated dataset. Datasets can carry
# more than one tag when the benchmark spans themes (e.g. general land-cover
# sets that include both urban and agricultural classes).
DATASET_DOMAINS: dict[str, set[str]] = {
    "m-pv4ger": {"energy_infrastructure", "urban"},
    "m-brick-kiln": {"energy_infrastructure", "urban"},  # industrial/kiln facilities
    "m-eurosat": {"landcover_general", "urban", "agriculture"},
    "eurosat": {"landcover_general", "urban", "agriculture"},
    "eurosat-spatial": {"landcover_general", "urban", "agriculture"},
    "m-forestnet": {"forestry"},
    "forestnet": {"forestry"},
    "m-so2sat": {"urban"},
    "so2sat": {"urban"},
    "m-bigearthnet": {"landcover_general", "agriculture", "urban"},
    "benv2": {"landcover_general", "agriculture", "urban"},
    "treesatai": {"forestry"},
    "caffe": {"agriculture"},  # field/crop boundary segmentation
    "burn_scars": {"disaster_response", "forestry"},  # wildfire burn scars
    "cloudsen12": {"other"},  # cloud/atmospheric masking, not a thematic domain
    "dynamic_earthnet": {"landcover_general"},
    "flair2": {"landcover_general", "urban", "agriculture"},
    "fotw": {"agriculture"},  # Fields Of The World — crop field boundaries
    "kuro_siwo": {"disaster_response"},  # flood mapping
    "pastis": {"agriculture"},  # crop parcel segmentation
    "spacenet2": {"urban"},  # building footprints
    "spacenet7": {"urban"},  # building change detection
    "cashew": {"agriculture", "forestry"},  # cashew plantation extent
    "sa_crop_type": {"agriculture"},
    "nz_cattle": {"agriculture"},  # livestock detection
    "neontree": {"forestry"},  # tree-crown delineation
    "chesapeake": {"landcover_general", "urban", "agriculture"},
    "pv4ger_seg": {"energy_infrastructure", "urban"},  # rooftop-PV segmentation
    "substations": {"energy_infrastructure", "urban"},  # power substation detection
    "everwatch": {"other"},  # bird-species detection (wildlife — no taxonomy tag)
    "biomassters": {"forestry"},  # above-ground biomass regression
    "mados": {"marine"},  # marine debris / oil-spill segmentation
    "sen1floods11": {"disaster_response"},  # flood-water mapping
    "xview2": {"disaster_response", "urban"},  # building damage assessment (HADR)
    "fbp": {"landcover_general", "urban", "agriculture"},  # Five Billion Pixels land cover
    "cropmap": {"agriculture"},  # crop-type mapping
    "ai4smallfarms": {"agriculture"},  # smallholder field boundaries
}

# Domains that overlap enough to deserve partial credit rather than a hard miss.
_RELATED_DOMAINS: dict[str, set[str]] = {
    "urban": {"energy_infrastructure"},
    "energy_infrastructure": {"urban"},
    "agriculture": {"landcover_general", "forestry"},
    "forestry": {"landcover_general", "disaster_response", "agriculture"},
    # disaster_water was split (2026-07-27): disaster_response = acute-event
    # damage mapping (floods, fires, building damage); marine = water-surface
    # tasks (debris, oil, coastal). The two stay related to each other.
    "disaster_response": {"landcover_general", "forestry", "marine"},
    "marine": {"landcover_general", "disaster_response"},
    "landcover_general": set(),  # handled specially below — a broad-fit baseline
    "other": set(),
}

# Plain-English description of what each benchmark actually measures — this
# is the material the agent reasons over to judge real-world similarity to
# the user's task; the domain tag above is only a coarse pre-filter, not the
# final word on thematic fit.
DATASET_DESCRIPTIONS: dict[str, str] = {
    "m-pv4ger": "Rooftop solar photovoltaic (PV) panel presence in high-res aerial imagery (binary).",
    "m-brick-kiln": "Brick kiln facility presence in Sentinel-2 imagery — an industrial/pollution-source detector (binary).",
    "m-eurosat": "General land-use/land-cover classification (10 classes incl. residential, industrial, forest, river) — broad, not theme-specific.",
    "eurosat": "Same EuroSAT land-use/land-cover classification task as m-eurosat, different split convention.",
    "eurosat-spatial": "EuroSAT with a spatially-disjoint train/test split — same 10 land-use/land-cover classes.",
    "m-forestnet": "Forest-loss driver classification (12 classes: what caused deforestation at a site) from Landsat.",
    "forestnet": "Same forest-loss driver classification task as m-forestnet, from Sentinel-2.",
    "m-so2sat": "Local climate zone classification in urban areas (17 classes) from S2+SAR.",
    "so2sat": "Same local-climate-zone urban classification task as m-so2sat.",
    "m-bigearthnet": "Multi-label land-cover/land-use tagging (43 CORINE classes) from Sentinel-2.",
    "benv2": "BigEarthNet-v2: multi-label land-cover/land-use tagging (19 classes) from Sentinel-1+2.",
    "treesatai": "Tree species classification (15 species) from aerial + S1/S2 imagery.",
    "caffe": "Crop/agricultural field boundary segmentation from aerial imagery.",
    "burn_scars": "Wildfire burn-scar segmentation from Sentinel-2 — post-fire damage mapping.",
    "cloudsen12": "Cloud/shadow mask segmentation from Sentinel-2 — a pre-processing task, not a thematic one.",
    "dynamic_earthnet": "Multi-class land-cover change segmentation over time from Planet+S2.",
    "flair2": "Fine-grained land-cover semantic segmentation (13 classes) from French aerial imagery.",
    "fotw": "\"Fields Of The World\" — agricultural field boundary segmentation from Sentinel-2.",
    "kuro_siwo": "Flood extent segmentation from SAR + DEM — a disaster-response mapping task.",
    "pastis": "Agricultural parcel segmentation with crop-type labels from Sentinel-1/2 time series.",
    "spacenet2": "Building footprint segmentation from high-res satellite imagery — urban structure mapping.",
    "spacenet7": "Building footprint change detection over time from Planet imagery — urban growth tracking.",
    "cashew": "Cashew plantation land-cover segmentation (7 classes) from Sentinel-2 in Benin — tree-crop extent mapping.",
    "sa_crop_type": "South-African crop-type segmentation (10 crop classes) from Sentinel-2 — agricultural parcel mapping.",
    "nz_cattle": "Cattle segmentation in 0.1 m aerial RGB (binary) — discrete-object detection framed as segmentation.",
    "neontree": "Individual tree-crown segmentation from 0.1 m aerial RGB + hyperspectral + elevation (binary).",
    "chesapeake": "Chesapeake Bay land-cover segmentation (7 classes) from 1 m NAIP RGBN aerial imagery.",
    "pv4ger_seg": "Rooftop solar-PV panel segmentation from 0.1 m aerial RGB (binary) — the segmentation twin of m-pv4ger.",
    "substations": "Electrical power-substation detection from Sentinel-2 (binary) — energy-infrastructure localization, modeled as segmentation.",
    "everwatch": "Bird-species detection in 0.1 m aerial RGB (9 species) over the Everglades — wildlife localization, modeled as segmentation.",
    "biomassters": "Above-ground forest biomass estimation (pixel-wise regression) from Sentinel-1+2 time series in Finland.",
    "mados": "Marine debris and oil-spill segmentation (15 classes) from Sentinel-2 — marine pollution mapping.",
    "sen1floods11": "Flood-water extent segmentation from Sentinel-1+2 — a disaster-response mapping task.",
    "xview2": "Building damage assessment from pre/post-disaster VHR Maxar imagery (5 damage levels) — HADR change detection.",
    "fbp": "Five Billion Pixels: fine-grained land-cover segmentation (24 classes) from 4 m Gaofen-2 imagery over China.",
    "cropmap": "Crop-type mapping segmentation from Sentinel-1/2 + Planet time series in South Sudan — agricultural parcels.",
    "ai4smallfarms": "Smallholder agricultural field-boundary segmentation from Sentinel-2 in Southeast Asia (binary).",
}

# --- task taxonomy ---------------------------------------------------------
# The registry's benchmarks are only "classification" or "segmentation", but a
# user's task can be finer-grained. We accept a richer vocabulary and score it
# against the two benchmark families with a graded compatibility table, instead
# of a binary same/different. Values are how transferable a frozen-backbone
# probe on a benchmark of that family is to the user's task type.
#   image-level tasks predict one label per chip; dense tasks predict per pixel.
_DENSE_TASKS = {"segmentation", "instance_segmentation", "change_detection"}
TASK_TYPES = (
    "classification", "multilabel_classification", "regression",
    "segmentation", "instance_segmentation", "change_detection", "object_detection",
)
# user_task -> (score vs a classification benchmark, score vs a segmentation benchmark)
_TASK_COMPAT: dict[str, tuple[int, int]] = {
    "classification":            (100, 25),
    "multilabel_classification": (90, 25),   # same image-level probe, richer label
    "regression":                (60, 30),   # image-level, but continuous target
    "segmentation":              (25, 100),
    "instance_segmentation":     (30, 85),   # dense masks, plus instance separation
    "change_detection":          (35, 70),   # dense per-pixel map over a time pair
    "object_detection":          (45, 65),   # localized objects; masks are the closer proxy
}


def _task_family(task: str) -> str:
    return "dense" if task in _DENSE_TASKS else "image"


def _score_task(user_task: str, cand_task: str) -> int:
    """Graded task compatibility (0-100) between the user's task and a
    benchmark whose task is classification, segmentation, or regression."""
    if user_task == cand_task:
        return 100
    # Regression is its own transfer regime: a frozen-probe ranking of backbones
    # on a per-pixel regression benchmark does not carry over to a classification
    # or segmentation task (or vice versa), so keep it isolated — don't let a
    # regression benchmark rank as a "close" match for a categorical task.
    if "regression" in (user_task, cand_task):
        return 20
    compat = _TASK_COMPAT.get(user_task)
    if compat is None:  # unknown label — exact match or hard miss
        return 20
    if cand_task == "classification":
        return compat[0]
    if cand_task == "segmentation":
        return compat[1]
    return 20


# Datasets that actually have knn5/linear rows in all_results.csv (set at load).
#
# Weights (subjective / AHP-style expert priors — there's no labeled
# task->best-benchmark corpus to fit them objectively; Shannon-entropy
# calibration is the path if one is ever collected). DOMAIN is weighted
# highest: source<->target *thematic* similarity is one of the strongest
# empirical predictors of transfer gain (domain-similarity / EMD-transfer
# literature), and it's the axis a domain expert cares about most. Task stays a
# close second as a near-gate (a wrong-family benchmark's numbers barely
# transfer). Bands/label/scale refine. The LLM still re-reads descriptions and
# can override the ranking — the weights set the prior, not the verdict.
#
# CALIBRATED (2026-07-17): re-set from a ROBUST cross-corpus search
# (scripts/calibrate_weights.py --multi). A single corpus overfits — the
# 11-fold torchgeo-bench linear probe alone wanted bands=.45/task=.15, while the
# 15-fold GEO-Bench-2 frozen leaderboard wanted task=.55/bands=.20. These
# weights minimise the WORST corpus's regret (normalised by its no-matching
# baseline) across three well-powered protocols — tgb-linear, tgb-knn5, and
# GEO-Bench-2 frozen (imported by scripts/import_leaderboards.py); PANGAEA
# frozen is also wired but held validation-only (4 usable folds). They tie the
# best single-corpus fit on both torchgeo corpora (regret 0.0100 / 0.0148) AND
# cut GEO-Bench-2 regret 0.0267 -> 0.0184; worst-corpus ratio 0.50. The search
# surfaced that spectral/sensor fit (bands) predicts frozen-probe transfer
# strongly, but task can't be starved (dense-vs-image family is a near-gate).
# CAVEAT: still only ~11-15 folds per corpus — indicative, not a true fit;
# re-run --multi as more evals land. Prior AHP weights kept below for reference.
_WEIGHTS = {"domain": 0.25, "task": 0.20, "bands": 0.40, "label": 0.10, "scale": 0.05}
_WEIGHTS_PRIOR = {"domain": 0.30, "task": 0.26, "bands": 0.18, "label": 0.14, "scale": 0.12}


# mtime-keyed cache so edits/additions to registry.json (e.g. new benchmark
# datasets) are picked up without a server restart, mirroring results._tables.
_REGISTRY: dict = {}


def registry() -> dict:
    try:
        key = config.REGISTRY_JSON.stat().st_mtime_ns
    except OSError:
        key = None
    if _REGISTRY.get("key") != key:
        _REGISTRY["val"] = json.loads(config.REGISTRY_JSON.read_text())["datasets"]
        _REGISTRY["key"] = key
    return _REGISTRY["val"]


def is_regression(name: str) -> bool:
    """True if the benchmark is a (per-pixel) regression task — its own transfer
    regime, not poolable with classification/segmentation rankings."""
    return registry().get(name, {}).get("task") == "regression"


def _band_wavelength_range(bands: list[dict]) -> tuple[float, float] | None:
    wl = [b["wavelength_um"] for b in bands if b.get("wavelength_um")]
    if not wl:
        return None
    return min(wl), max(wl)


def _score_bands(band_count: int, sensor: str, cand: dict) -> int:
    # sensor family — optical families get partial credit for each other,
    # scored symmetrically (either side being optical qualifies).
    cand_sensors = set(cand["sensors"])
    if sensor in cand_sensors:
        s_sensor = 100
    elif sensor in _OPTICAL_SENSORS and cand_sensors & _OPTICAL_SENSORS:
        s_sensor = 70  # both optical
    elif sensor == "unknown":
        s_sensor = 50
    else:
        s_sensor = 20

    # channel-count ratio; multispectral candidates can run RGB-only, so
    # compare against 3 when the user's set is RGB-ish
    cand_ch = cand["num_channels"]
    if band_count <= 4 and cand.get("rgb_bands"):
        cand_ch = min(cand_ch, 3 if band_count == 3 else 4)
    s_channels = round(100 * min(band_count, cand_ch) / max(band_count, cand_ch))

    # spectral-range IoU when wavelengths known on the candidate side
    user_range = {3: (0.45, 0.67), 4: (0.45, 0.87)}.get(band_count)
    cand_range = _band_wavelength_range(cand["bands"])
    if user_range and cand_range:
        lo = max(user_range[0], cand_range[0])
        hi = min(user_range[1], cand_range[1])
        inter = max(0.0, hi - lo)
        union = max(user_range[1], cand_range[1]) - min(user_range[0], cand_range[0])
        s_spectral = round(100 * inter / union) if union else 50
        return round(0.5 * s_sensor + 0.3 * s_channels + 0.2 * s_spectral)
    # redistribute spectral weight
    return round(0.62 * s_sensor + 0.38 * s_channels)


def _score_label(multilabel: bool, num_classes: int, task_type: str, cand: dict) -> int:
    user_dense = _task_family(task_type) == "dense"
    cand_dense = cand["task"] == "segmentation"
    if user_dense != cand_dense:
        return 25  # image-level vs dense: label structures aren't comparable
    # class-count distance on a log2 scale (both same family)
    score = max(30, round(100 - 25 * abs(math.log2(max(num_classes, 1) / cand["num_classes"]))))
    if not user_dense:
        # image-level nuances: binary parity, and multilabel vs single-label
        if num_classes == 2 and cand["num_classes"] == 2:
            score = 100
        elif (num_classes == 2) != (cand["num_classes"] == 2):
            score = min(score, 45)
        if multilabel != cand["multilabel"]:
            score = max(10, score - 40)
    return score


def _score_scale(object_scale: str, gsd_m: float | None, cand_name: str) -> int:
    meta = DATASET_META.get(cand_name)
    if meta is None:
        return 50
    cand_gsd, cand_scale = meta
    if gsd_m:
        return max(0, round(100 - 60 * abs(math.log10(gsd_m / cand_gsd))))
    if object_scale == cand_scale:
        return 100
    if {object_scale, cand_scale} <= {"object", "scene"} or {object_scale, cand_scale} <= {"scene", "landcover"}:
        return 55
    return 30


def _score_domain(domain: str, cand_name: str) -> int:
    cand_domains = DATASET_DOMAINS.get(cand_name)
    if not cand_domains or domain == "other":
        return 50  # no thematic tag on either side — stay neutral, don't punish
    if domain in cand_domains:
        return 100
    if domain == "landcover_general" or "landcover_general" in cand_domains:
        # broad land-cover/land-use benchmarks are a reasonable partial fit
        # for almost any thematic task, and vice versa
        return 60
    if _RELATED_DOMAINS.get(domain, set()) & cand_domains:
        return 55
    return 20


def match_datasets(
    task_type: str,
    multilabel: bool,
    num_classes: int,
    band_count: int,
    sensor: str,
    object_scale: str,
    domain: str = "other",
    gsd_m: float | None = None,
    datasets_with_results: set[str] | None = None,
) -> list[dict]:
    """Score every registry dataset against the described upload.

    Returns all candidates sorted by sim desc, each with a per-facet
    breakdown and a has_results flag.
    """
    if gsd_m is None:
        gsd_m = SENSOR_DEFAULT_GSD.get(sensor)
    out = []
    for name, cand in registry().items():
        facets = {
            "Task": _score_task(task_type, cand["task"]),
            "Bands": _score_bands(band_count, sensor, cand),
            "Label type": _score_label(multilabel, num_classes, task_type, cand),
            "Object scale": _score_scale(object_scale, gsd_m, name),
            "Domain": _score_domain(domain, name),
        }
        sim = round(
            _WEIGHTS["task"] * facets["Task"]
            + _WEIGHTS["bands"] * facets["Bands"]
            + _WEIGHTS["label"] * facets["Label type"]
            + _WEIGHTS["scale"] * facets["Object scale"]
            + _WEIGHTS["domain"] * facets["Domain"]
        )
        out.append(
            {
                "name": name,
                "sim": sim,
                "task": cand["task"],
                "num_classes": cand["num_classes"],
                "multilabel": cand["multilabel"],
                "num_channels": cand["num_channels"],
                "sensors": cand["sensors"],
                "gsd_m": DATASET_META.get(name, (None, None))[0],
                "object_scale": DATASET_META.get(name, (None, None))[1],
                "has_results": name in datasets_with_results if datasets_with_results else None,
                "description": DATASET_DESCRIPTIONS.get(name, ""),
                "domain_tags": sorted(DATASET_DOMAINS.get(name, set())),
                "facets": [{"label": k, "pct": v} for k, v in facets.items()],
            }
        )
    out.sort(key=lambda d: (-d["sim"], d["name"]))
    return out
