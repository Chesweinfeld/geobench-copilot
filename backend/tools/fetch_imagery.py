"""Auto-fetch real Sentinel-2 imagery for uploaded vector geometries.

Used only as a fallback: if an archive has GeoJSON/KML but no raster chips
of its own, we pull real scenes for each geometry's AOI from Microsoft
Planetary Computer's public STAC API instead of failing the run. Requires
the optional `satellite` extra (rasterio, pystac-client, planetary-computer)
and outbound network access; both are expected to be present in the
deployed environment even though they may not be reachable from a sandbox.

Neighbouring geometries almost always share a Sentinel-2 scene (municipal
favela/settlement layers cluster tightly — e.g. 86% of Rio's 2016 favelas
sit within one AOI-width of another, and the whole set fits inside a single
S2 tile). So by default we bin AOIs into a spatial grid and fetch each cell
with ONE catalog search and ONE windowed read covering the cell's union,
then crop and rasterize every chip locally — turning O(chips) network ops
into O(regions). The per-geometry path is kept as an automatic fallback for
any cell the grouped read can't satisfy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("geobench.fetch_imagery")

_RGB_BANDS = ("B04", "B03", "B02")  # red, green, blue — matches the app's band-name guesser
_M_PER_DEG_LAT = 111_320

# Sentinel-2 L2A Scene Classification (SCL) codes that mean "not usable
# ground signal" — no-data, saturated/defective, cloud shadow, cloud
# (medium/high probability), thin cirrus. Used to mask bad pixels out of a
# composite rather than average clouds into it.
_SCL_INVALID = {0, 1, 3, 8, 9, 10}


class FetchImageryError(ValueError):
    """Raised when satellite imagery can't be fetched for the given AOIs.

    Message is user-safe — callers surface it directly in InspectError.
    """


@dataclass
class FetchedChip:
    path: Path
    item_id: str  # item id, or "composite(N)" when blended from several scenes
    datetime: str | None  # None for composites (spans a date range)
    cloud_cover: float | None  # mean input cloud cover for composites
    composite: bool = False
    source_items: list[str] = field(default_factory=list)
    aoi_side_m: float = 0.0
    aoi_clipped: bool = False  # true if the geometry was larger than the max AOI and got cropped
    mask_path: Path | None = None  # rasterized footprint mask, when the geometry had real area
    positive_frac: float | None = None  # fraction of mask pixels inside the polygon
    class_name: str | None = None  # this feature's category, when a class field was detected
    split: str = "train"  # spatial-block split assignment (train/val/test)


@dataclass
class ClassInfo:
    """Describes the categorical attribute field used to assign per-feature
    class IDs when rasterizing polygon labels — lets inspect_archive report
    exact numClasses/classBalance from the real feature attributes instead of
    guessing from a small sample of mask pixels."""
    field: str
    class_ids: dict[str, int]  # class name -> burned raster value (1..N; 0 is background)
    counts: dict[str, int]  # class name -> number of features with that class


@dataclass
class _AOI:
    """A geometry paired with its resolved AOI box + label metadata, computed
    once up front so grouping and both fetch paths can share it."""
    index: int
    geom: dict
    props: dict
    bbox: tuple[float, float, float, float]  # WGS84 (minx, miny, maxx, maxy)
    half_side_m: float
    clipped: bool
    class_name: str | None
    class_id: int
    split: str = "train"  # assigned by _assign_spatial_splits


# Common label-ish attribute names (English + a few Portuguese/Spanish terms,
# since open municipal datasets — e.g. Brazilian favela/settlement layers —
# are often not in English) checked first when guessing which property holds
# the class/category for each feature.
_CLASS_FIELD_HINTS = (
    "class", "type", "category", "label", "kind",
    "classe", "tipo", "categoria", "tipologia",
)


def _pick_class_field(props_list: list[dict]) -> str | None:
    """Best-effort guess at which attribute field is a categorical class
    label (2-20 distinct string values), preferring common label-ish names.
    Returns None if nothing qualifies — callers fall back to a binary mask."""
    if not props_list:
        return None
    keys: dict[str, None] = {}
    for p in props_list:
        for k in p:
            keys.setdefault(k, None)

    def qualifies(key: str) -> int | None:
        values = [p.get(key) for p in props_list if p.get(key) not in (None, "")]
        if len(values) < len(props_list) * 0.5:
            return None  # too many features missing this field
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return None  # looks numeric/continuous (e.g. area, population), not categorical
        distinct = len(set(str(v) for v in values))
        return distinct if 2 <= distinct <= 20 else None

    for hint in _CLASS_FIELD_HINTS:
        for key in keys:
            if hint == key.lower() and qualifies(key):
                return key
    for key in keys:
        if qualifies(key):
            return key
    return None


def _walk_coords(coords, fn) -> None:
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        fn(coords[0], coords[1])
    else:
        for c in coords:
            _walk_coords(c, fn)


def _geometry_extent(geometry: dict) -> tuple[float, float, float, float]:
    """Raw (minx, miny, maxx, maxy) spanning every coordinate in the geometry."""
    coords = geometry.get("coordinates")
    if coords is None:
        raise FetchImageryError("geometry has no coordinates")
    box = [float("inf"), float("inf"), float("-inf"), float("-inf")]

    def note(x, y):
        box[0] = min(box[0], x); box[1] = min(box[1], y)
        box[2] = max(box[2], x); box[3] = max(box[3], y)

    _walk_coords(coords, note)
    if box[0] == float("inf"):
        raise FetchImageryError("geometry has no usable coordinates")
    return tuple(box)


def _bbox_from_half_side(lon: float, lat: float, half_side_m: float) -> tuple[float, float, float, float]:
    """WGS84 bbox of a square AOI of the given half-side (meters) centered
    on (lon, lat). Equirectangular approximation — fine at chip scale."""
    dlat = half_side_m / _M_PER_DEG_LAT
    dlon = half_side_m / (_M_PER_DEG_LAT * max(math.cos(math.radians(lat)), 1e-6))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _aoi_for_geometry(geometry: dict) -> tuple[tuple[float, float, float, float], float, bool]:
    """Build an AOI box sized to the geometry's own extent, not a fixed box
    that might clip a large polygon or waste resolution on a tiny point.

    Returns (bbox, half_side_m, clipped) where clipped is True if the
    geometry's real extent exceeded FETCH_AOI_MAX_HALF_SIDE_M and the AOI
    had to be capped (centered on the geometry, but not covering all of it).
    """
    minx, miny, maxx, maxy = _geometry_extent(geometry)
    lon_c, lat_c = (minx + maxx) / 2, (miny + maxy) / 2

    m_per_deg_lon = _M_PER_DEG_LAT * max(math.cos(math.radians(lat_c)), 1e-6)
    width_m = (maxx - minx) * m_per_deg_lon
    height_m = (maxy - miny) * _M_PER_DEG_LAT
    span_m = max(width_m, height_m)

    if span_m < 1.0:
        # point-like geometry (or a degenerate one) — use the fixed default
        half_side = config.FETCH_AOI_HALF_SIDE_M
        clipped = False
    else:
        wanted = span_m * (1 + 2 * config.FETCH_AOI_PAD_FRAC) / 2
        half_side = max(wanted, config.FETCH_AOI_HALF_SIDE_M)
        clipped = half_side > config.FETCH_AOI_MAX_HALF_SIDE_M
        half_side = min(half_side, config.FETCH_AOI_MAX_HALF_SIDE_M)

    return _bbox_from_half_side(lon_c, lat_c, half_side), half_side, clipped


def _read_window(rasterio_mod, transform_bounds_fn, window_from_bounds_fn, asset_href: str,
                  bbox_4326: tuple[float, float, float, float], out_shape: tuple[int, int]):
    """Windowed, resampled read of one raster asset over an AOI. `out_shape`
    is (height, width) of the output grid. Returns (data, crs, win_bounds)
    where win_bounds is the AOI in the source CRS (describes the output grid)."""
    with rasterio_mod.open(asset_href) as src:
        win_bounds = transform_bounds_fn("EPSG:4326", src.crs, *bbox_4326)
        window = window_from_bounds_fn(*win_bounds, transform=src.transform)
        data = src.read(1, window=window, out_shape=out_shape)
        crs = src.crs
    return data, crs, win_bounds


def _read_item_rgb(rio_mods, item, bbox, out_shape: tuple[int, int], mask_clouds: bool):
    """Read one STAC item's RGB bands (+ optional SCL cloud mask) over an AOI
    at the given (height, width) output grid.

    Returns (stack[3,H,W] float32, bad_pixel_mask[H,W] bool | None, crs, win_bounds).
    """
    rasterio_mod, transform_bounds_fn, window_from_bounds_fn = rio_mods
    bands = []
    crs = win_bounds = None
    for b in _RGB_BANDS:
        asset = item.assets.get(b)
        if asset is None:
            raise FetchImageryError(f"item {item.id!r} is missing band {b}")
        data, c, wb = _read_window(rasterio_mod, transform_bounds_fn, window_from_bounds_fn,
                                    asset.href, bbox, out_shape)
        if crs is None:
            crs, win_bounds = c, wb
        bands.append(data.astype(np.float32))
    stack = np.stack(bands, axis=0)

    mask = None
    if mask_clouds and "SCL" in item.assets:
        try:
            scl, _, _ = _read_window(rasterio_mod, transform_bounds_fn, window_from_bounds_fn,
                                      item.assets["SCL"].href, bbox, out_shape)
            mask = np.isin(scl, list(_SCL_INVALID))
        except Exception as e:  # noqa: BLE001 — cloud masking is best-effort
            log.warning("SCL read failed for item %s: %s — compositing unmasked", item.id, e)
            mask = None
    return stack, mask, crs, win_bounds


_POLYGONAL_TYPES = {"Polygon", "MultiPolygon"}


def _rasterize_footprint(rasterio_mod, transform_geom_fn, geometry: dict, crs, transform,
                          chip_px: int, class_id: int = 1) -> tuple[np.ndarray, float] | None:
    """Rasterize a Polygon/MultiPolygon geometry onto the chip's own pixel
    grid, so a polygon label already present in the upload (a building
    footprint, a settlement boundary, ...) becomes a real segmentation mask
    for the fetched imagery instead of being discarded down to a centroid.

    Burns `class_id` (default 1, i.e. binary foreground/background) so
    multi-class labels — when the upload's attributes encode a category —
    come through as real per-pixel class values, not just presence/absence.

    Returns (mask[H,W] uint8, positive_fraction) or None if the geometry
    isn't polygonal or rasterization fails.
    """
    if geometry.get("type") not in _POLYGONAL_TYPES:
        return None
    try:
        from rasterio.features import rasterize

        geom_in_crs = transform_geom_fn("EPSG:4326", crs, geometry)
        mask = rasterize(
            [(geom_in_crs, class_id)], out_shape=(chip_px, chip_px), transform=transform,
            fill=0, dtype="uint8",
        )
    except Exception as e:  # noqa: BLE001 — best-effort; missing mask just means no label for this chip
        log.warning("mask rasterization failed: %s", e)
        return None
    return mask, float((mask > 0).mean())


def _write_mask_if_polygonal(rasterio_mod, transform_geom_fn, geometry, crs, transform,
                              index: int, mask_dir: Path, class_id: int = 1
                              ) -> tuple[Path | None, float | None]:
    """Rasterize + write a footprint mask for this chip if the source
    geometry is a Polygon/MultiPolygon. No-op (returns None, None) for
    Point/LineString geometries, which have no area to segment."""
    result = _rasterize_footprint(rasterio_mod, transform_geom_fn, geometry, crs, transform,
                                   config.FETCH_CHIP_PX, class_id=class_id)
    if result is None:
        return None, None
    mask, positive_frac = result
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"chip_{index:03d}.tif"
    with rasterio_mod.open(
        mask_path, "w", driver="GTiff", height=config.FETCH_CHIP_PX, width=config.FETCH_CHIP_PX,
        count=1, dtype="uint8", crs=crs, transform=transform,
    ) as dst:
        dst.write(mask[None, :, :])
    return mask_path, positive_frac


def _write_rgb_tif(rasterio_mod, out: Path, stack: np.ndarray, crs, transform) -> None:
    """Write a [C,H,W] float/int stack as a uint16 RGB GeoTIFF chip."""
    chip = np.round(stack).astype(np.uint16)
    with rasterio_mod.open(
        out, "w", driver="GTiff", height=chip.shape[1], width=chip.shape[2],
        count=chip.shape[0], dtype=chip.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(chip)


def _composite_rgb(rio_mods, items, bbox, out_shape: tuple[int, int]):
    """Cloud-masked per-pixel median composite across several scenes, over the
    given AOI at the given (height, width) output grid.

    Falls back to a plain median at any pixel where every input scene was
    masked out (so a run never produces a fully-blank chip)."""
    stacks, masks, used = [], [], []
    crs = win_bounds = None
    for item in items[: config.STAC_COMPOSITE_MAX_ITEMS]:
        try:
            stack, mask, c, wb = _read_item_rgb(rio_mods, item, bbox, out_shape, mask_clouds=True)
        except FetchImageryError:
            continue
        if crs is None:
            crs, win_bounds = c, wb
        stacks.append(stack)
        masks.append(mask)
        used.append(item)
    if not stacks:
        raise FetchImageryError("none of the candidate scenes could be read for compositing")

    arr = np.stack(stacks, axis=0)  # (N, 3, H, W)
    if any(m is not None for m in masks):
        mask_arr = np.stack(
            [m if m is not None else np.zeros(arr.shape[2:], dtype=bool) for m in masks], axis=0
        )
        arr_masked = np.where(mask_arr[:, None, :, :], np.nan, arr)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            composite = np.nanmedian(arr_masked, axis=0)
        all_bad = np.all(mask_arr, axis=0)
        if all_bad.any():
            plain = np.median(arr, axis=0)
            composite = np.where(all_bad[None, :, :], plain, composite)
    else:
        composite = np.median(arr, axis=0)

    return np.nan_to_num(composite, nan=0.0), crs, win_bounds, used


# ---------------------------------------------------------------------------
# Grouped fetch (default): one search + one union read per region
# ---------------------------------------------------------------------------

def _union_bbox(aois: list[_AOI]) -> tuple[float, float, float, float]:
    return (min(a.bbox[0] for a in aois), min(a.bbox[1] for a in aois),
            max(a.bbox[2] for a in aois), max(a.bbox[3] for a in aois))


def _group_aois(aois: list[_AOI], cell_m: float) -> list[list[_AOI]]:
    """Bin AOIs into a grid of ~cell_m cells (by AOI center) so neighbours
    that share a Sentinel-2 scene are fetched together. Cloud cover is a
    whole-scene property, so cell-mates already resolve to the same scene
    under the per-geometry path — grouping just dedupes the search + reads."""
    if not aois:
        return []
    lat_ref = sum((a.bbox[1] + a.bbox[3]) / 2 for a in aois) / len(aois)
    cell_lat = cell_m / _M_PER_DEG_LAT
    cell_lon = cell_m / (_M_PER_DEG_LAT * max(math.cos(math.radians(lat_ref)), 1e-6))
    groups: dict[tuple[int, int], list[_AOI]] = defaultdict(list)
    for a in aois:
        cx = (a.bbox[0] + a.bbox[2]) / 2
        cy = (a.bbox[1] + a.bbox[3]) / 2
        groups[(math.floor(cx / cell_lon), math.floor(cy / cell_lat))].append(a)
    return list(groups.values())


def _assign_spatial_splits(aois: list[_AOI]) -> None:
    """Assign each AOI a train/val/test split by SPATIAL BLOCK, in place.

    Fetched AOIs overlap heavily (co-located settlements), so a per-chip split
    leaks near-identical pixels across train and test. Instead we bin AOIs into
    coarse blocks (FETCH_SPLIT_BLOCK_KM) and keep each whole block in one split.

    Blocks are walked in a stable pseudo-random order (hash of the block key —
    deterministic across processes, unlike Python's salted hash()) and each is
    handed to whichever split is furthest below its target share. That keeps the
    80/10/10 ratios close even with few blocks, where independent per-block
    hashing would leave val nearly empty."""
    if not aois:
        return
    lat_ref = sum((a.bbox[1] + a.bbox[3]) / 2 for a in aois) / len(aois)
    block_m = config.FETCH_SPLIT_BLOCK_KM * 1000
    cell_lat = block_m / _M_PER_DEG_LAT
    cell_lon = block_m / (_M_PER_DEG_LAT * max(math.cos(math.radians(lat_ref)), 1e-6))

    blocks: dict[str, list[_AOI]] = defaultdict(list)
    for a in aois:
        cx = (a.bbox[0] + a.bbox[2]) / 2
        cy = (a.bbox[1] + a.bbox[3]) / 2
        blocks[f"{math.floor(cx / cell_lon)},{math.floor(cy / cell_lat)}"].append(a)

    n = len(aois)
    r_train, r_val, r_test = config.FETCH_SPLIT_RATIOS
    targets = {"train": r_train * n, "val": r_val * n, "test": r_test * n}
    counts = {"train": 0.0, "val": 0.0, "test": 0.0}
    ordered = sorted(blocks.items(), key=lambda kv: hashlib.sha1(kv[0].encode()).hexdigest())
    for _key, members in ordered:
        # largest relative deficit wins — spreads blocks to hit target shares
        split = max(("train", "val", "test"),
                    key=lambda k: (targets[k] - counts[k]) / max(targets[k], 1.0))
        counts[split] += len(members)
        for a in members:
            a.split = split


def _union_out_shape(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    """Output (height, width) for a union read: native ~10m GSD, floored at a
    single chip and capped at FETCH_GROUP_MAX_READ_PX to bound memory."""
    minx, miny, maxx, maxy = bbox
    lat = (miny + maxy) / 2
    w_m = (maxx - minx) * _M_PER_DEG_LAT * max(math.cos(math.radians(lat)), 1e-6)
    h_m = (maxy - miny) * _M_PER_DEG_LAT
    gsd = config.FETCH_TARGET_GSD_M
    lo, hi = config.FETCH_CHIP_PX, config.FETCH_GROUP_MAX_READ_PX
    w = min(hi, max(lo, round(w_m / gsd)))
    h = min(hi, max(lo, round(h_m / gsd)))
    return (h, w)


def _crop_chip(src, transform_bounds_fn, window_from_bounds_fn, bbox_4326, chip_px):
    """Crop + resample a chip_px square RGB chip from an already-open (in-memory)
    source dataset over an AOI. No network — the union scene is read once and
    every chip is cut from it locally. Returns (stack[3,H,W] float32, win_bounds)."""
    win_bounds = transform_bounds_fn("EPSG:4326", src.crs, *bbox_4326)
    window = window_from_bounds_fn(*win_bounds, transform=src.transform)
    data = src.read(indexes=[1, 2, 3], window=window, out_shape=(3, chip_px, chip_px))
    return data.astype(np.float32), win_bounds


def _fetch_group(rio_mods, rasterio_mod, memoryfile_cls, transform_from_bounds_fn,
                  transform_geom_fn, catalog, group: list[_AOI],
                  img_dir: Path, mask_dir: Path, warnings_out: list[str]
                  ) -> list[tuple[Path, FetchedChip]]:
    """Fetch a spatial group with ONE catalog search and ONE windowed read
    (per band, or per scene when compositing) covering the group's union, then
    crop + rasterize each chip locally. Raises FetchImageryError if the group's
    union can't be read at all, so the caller can fall back to per-geometry."""
    _, transform_bounds_fn, window_from_bounds_fn = rio_mods
    ubbox = _union_bbox(group)

    items = list(catalog.search(
        collections=[config.STAC_COLLECTION], bbox=ubbox,
        datetime=config.STAC_DATETIME_RANGE,
        max_items=max(config.STAC_COMPOSITE_MAX_ITEMS, 5),
    ).items())
    if not items:
        raise FetchImageryError(f"no {config.STAC_COLLECTION} scene found for region AOI")
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    best = items[0]
    best_cloud = best.properties.get("eo:cloud_cover", 100)
    out_shape = _union_out_shape(ubbox)

    if best_cloud <= config.STAC_MAX_CLOUD_COVER:
        union_rgb, _m, crs, win_bounds = _read_item_rgb(rio_mods, best, ubbox, out_shape, mask_clouds=False)
        item_id, dt, cloud, composite = best.id, best.properties.get("datetime"), best_cloud, False
        source_items = [best.id]
        region_note = None
    else:
        union_rgb, crs, win_bounds, used = _composite_rgb(rio_mods, items, ubbox, out_shape)
        mean_cloud = sum(it.properties.get("eo:cloud_cover", 100) for it in used) / len(used)
        item_id, dt, cloud, composite = f"composite({len(used)})", None, round(mean_cloud, 1), True
        source_items = [it.id for it in used]
        region_note = (
            f"region of {len(group)} chip(s): no scene under {config.STAC_MAX_CLOUD_COVER:.0f}% "
            f"cloud cover (best was {best_cloud:.0f}%) — used a {len(used)}-scene "
            "cloud-masked composite for the whole region"
        )

    uh, uw = union_rgb.shape[1], union_rgb.shape[2]
    union_transform = transform_from_bounds_fn(*win_bounds, uw, uh)
    union_u16 = np.round(union_rgb).astype(np.uint16)

    results: list[tuple[Path, FetchedChip]] = []
    with memoryfile_cls() as mf:
        with mf.open(driver="GTiff", height=uh, width=uw, count=3, dtype="uint16",
                     crs=crs, transform=union_transform) as ds:
            ds.write(union_u16)
        with mf.open() as src:
            for a in group:
                try:
                    data, cb = _crop_chip(src, transform_bounds_fn, window_from_bounds_fn,
                                          a.bbox, config.FETCH_CHIP_PX)
                    ctransform = transform_from_bounds_fn(*cb, config.FETCH_CHIP_PX, config.FETCH_CHIP_PX)
                    out = img_dir / f"chip_{a.index:03d}.tif"
                    _write_rgb_tif(rasterio_mod, out, data, crs, ctransform)
                    mask_path, positive_frac = _write_mask_if_polygonal(
                        rasterio_mod, transform_geom_fn, a.geom, crs, ctransform,
                        a.index, mask_dir, class_id=a.class_id)
                    results.append((out, FetchedChip(
                        path=out, item_id=item_id, datetime=dt, cloud_cover=cloud,
                        composite=composite, source_items=source_items,
                        aoi_side_m=a.half_side_m * 2, aoi_clipped=a.clipped,
                        mask_path=mask_path, positive_frac=positive_frac, class_name=a.class_name,
                        split=a.split,
                    )))
                except Exception as e:  # noqa: BLE001 — one bad crop shouldn't sink the region
                    log.warning("crop failed for geometry %d: %s", a.index, e)
                    warnings_out.append(f"geometry {a.index}: {e}")

    if not results:
        raise FetchImageryError("region produced no chips")
    if region_note:
        warnings_out.append(region_note)
    return results


# ---------------------------------------------------------------------------
# Per-geometry fetch (fallback): one search + read per geometry
# ---------------------------------------------------------------------------

def _fetch_one_geometry(rio_mods, rasterio_mod, transform_from_bounds_fn, transform_geom_fn,
                         catalog, a: _AOI, img_dir: Path, mask_dir: Path,
                         warnings_out: list[str]) -> tuple[Path, FetchedChip] | None:
    """Original path: search + read + write + rasterize for a single geometry.
    Returns (chip_path, meta) or None (and appends a warning) on failure."""
    i = a.index
    chip_shape = (config.FETCH_CHIP_PX, config.FETCH_CHIP_PX)
    try:
        all_items = list(catalog.search(
            collections=[config.STAC_COLLECTION], bbox=a.bbox,
            datetime=config.STAC_DATETIME_RANGE,
            max_items=max(config.STAC_COMPOSITE_MAX_ITEMS, 5),
        ).items())
        if not all_items:
            warnings_out.append(f"geometry {i}: no {config.STAC_COLLECTION} scene found for this AOI")
            return None
        all_items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
        best = all_items[0]
        best_cloud = best.properties.get("eo:cloud_cover", 100)
        out = img_dir / f"chip_{i:03d}.tif"

        if best_cloud <= config.STAC_MAX_CLOUD_COVER:
            stack, _mask, crs, win_bounds = _read_item_rgb(rio_mods, best, a.bbox, chip_shape, mask_clouds=False)
            transform = transform_from_bounds_fn(*win_bounds, config.FETCH_CHIP_PX, config.FETCH_CHIP_PX)
            _write_rgb_tif(rasterio_mod, out, stack, crs, transform)
            mask_path, positive_frac = _write_mask_if_polygonal(
                rasterio_mod, transform_geom_fn, a.geom, crs, transform, i, mask_dir, class_id=a.class_id)
            return out, FetchedChip(
                path=out, item_id=best.id, datetime=best.properties.get("datetime"),
                cloud_cover=best_cloud, composite=False, source_items=[best.id],
                aoi_side_m=a.half_side_m * 2, aoi_clipped=a.clipped,
                mask_path=mask_path, positive_frac=positive_frac, class_name=a.class_name,
                split=a.split,
            )

        # no clear single scene — blend the least-cloudy ones into a composite
        composite, crs, win_bounds, used = _composite_rgb(rio_mods, all_items, a.bbox, chip_shape)
        transform = transform_from_bounds_fn(*win_bounds, config.FETCH_CHIP_PX, config.FETCH_CHIP_PX)
        _write_rgb_tif(rasterio_mod, out, composite, crs, transform)
        mean_cloud = sum(it.properties.get("eo:cloud_cover", 100) for it in used) / len(used)
        warnings_out.append(
            f"geometry {i}: no scene under {config.STAC_MAX_CLOUD_COVER:.0f}% cloud cover "
            f"(best was {best_cloud:.0f}%) — used a {len(used)}-scene cloud-masked composite instead"
        )
        mask_path, positive_frac = _write_mask_if_polygonal(
            rasterio_mod, transform_geom_fn, a.geom, crs, transform, i, mask_dir, class_id=a.class_id)
        return out, FetchedChip(
            path=out, item_id=f"composite({len(used)})", datetime=None,
            cloud_cover=round(mean_cloud, 1), composite=True, source_items=[it.id for it in used],
            aoi_side_m=a.half_side_m * 2, aoi_clipped=a.clipped,
            mask_path=mask_path, positive_frac=positive_frac, class_name=a.class_name,
            split=a.split,
        )
    except FetchImageryError as e:
        warnings_out.append(f"geometry {i}: {e}")
    except Exception as e:  # noqa: BLE001 — one bad geometry shouldn't sink the run
        log.warning("satellite fetch failed for geometry %d: %s", i, e)
        warnings_out.append(f"geometry {i}: {e}")
    return None


def fetch_chips_for_geometries(
    geometries: list[tuple[dict, dict]], workdir: Path
) -> tuple[list[Path], list[FetchedChip], list[str], ClassInfo | None]:
    """Fetch one Sentinel-2 RGB chip per (geometry, properties) pair (capped
    at MAX_FETCH_CHIPS).

    Each AOI is sized to the geometry's own extent (padded, capped at
    FETCH_AOI_MAX_HALF_SIDE_M) rather than a fixed box that could clip a
    large polygon. By default neighbouring AOIs are grouped into a spatial
    grid and each cell is fetched with a single catalog search + windowed
    read, cropping every chip locally (set FETCH_GROUP_ENABLED=0 to force the
    per-geometry path). If no single scene is under STAC_MAX_CLOUD_COVER, the
    least-cloudy scenes are blended into an SCL-masked median composite.

    If a geometry is a Polygon/MultiPolygon, its footprint is rasterized into
    a real segmentation mask. When the features' properties carry a
    categorical attribute (auto-detected — see _pick_class_field), each
    feature's own class value is burned into its mask as a distinct class ID
    instead of a flat binary 1, and a ClassInfo describing the field/mapping/
    counts is returned so the caller can report exact numClasses/classBalance.

    Returns (chip_paths, fetched_meta, warnings, class_info). Raises
    FetchImageryError only if *no* chip could be fetched at all.
    """
    if not geometries:
        raise FetchImageryError("no vector geometries available to fetch imagery for")

    class_field = _pick_class_field([props for _, props in geometries])
    class_info: ClassInfo | None = None
    if class_field:
        distinct_values = sorted({str(props[class_field]) for _, props in geometries
                                   if props.get(class_field) not in (None, "")})
        class_ids = {name: idx + 1 for idx, name in enumerate(distinct_values)}
        counts = Counter(str(props[class_field]) for _, props in geometries
                          if props.get(class_field) not in (None, ""))
        class_info = ClassInfo(field=class_field, class_ids=class_ids, counts=dict(counts))

    try:
        import pystac_client
        import planetary_computer as pc
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds as transform_from_bounds
        from rasterio.warp import transform_bounds, transform_geom
        from rasterio.windows import from_bounds as window_from_bounds
    except ImportError as e:
        raise FetchImageryError(
            "satellite auto-fetch needs the 'satellite' extra "
            "(uv sync --extra satellite / pip install rasterio pystac-client "
            "planetary-computer)"
        ) from e
    rio_mods = (rasterio, transform_bounds, window_from_bounds)

    try:
        catalog = pystac_client.Client.open(config.STAC_API_URL, modifier=pc.sign_inplace)
    except Exception as e:  # noqa: BLE001 — surface network/DNS/etc. errors plainly
        raise FetchImageryError(f"could not reach the satellite catalog ({config.STAC_API_URL}): {e}") from e

    img_dir = workdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = workdir / "masks"

    warnings_out: list[str] = []

    # Resolve every AOI once (shared by grouping + both fetch paths).
    aois: list[_AOI] = []
    for i, (geom, props) in enumerate(geometries[: config.MAX_FETCH_CHIPS]):
        try:
            bbox, half_side_m, clipped = _aoi_for_geometry(geom)
        except FetchImageryError as e:
            warnings_out.append(f"geometry {i}: {e}")
            continue
        class_name = (str(props[class_field])
                      if class_field and props.get(class_field) not in (None, "") else None)
        class_id = class_info.class_ids.get(class_name, 1) if (class_info and class_name) else 1
        if clipped:
            warnings_out.append(
                f"geometry {i}: extent exceeds the {config.FETCH_AOI_MAX_HALF_SIDE_M:.0f}m "
                "max AOI half-side — fetched chip is centered on it but doesn't cover the whole shape"
            )
        aois.append(_AOI(i, geom, props, bbox, half_side_m, clipped, class_name, class_id))

    dropped = len(geometries) - len(geometries[: config.MAX_FETCH_CHIPS])
    if dropped > 0:
        warnings_out.append(
            f"only the first {config.MAX_FETCH_CHIPS} of {len(geometries)} geometries were "
            f"fetched (MAX_FETCH_CHIPS cap) — {dropped} skipped"
        )

    # Spatial-block train/val/test assignment (before any fetch) so overlapping
    # neighbours never straddle splits. Independent of the fetch grouping below.
    _assign_spatial_splits(aois)

    # Each work unit is fetched independently; regions/geometries are network
    # I/O-bound, so run them on a thread pool. Every worker collects its own
    # warnings and returns them, so no shared-list races.
    def _run_grouped(group: list[_AOI]):
        w: list[str] = []
        try:
            return _fetch_group(rio_mods, rasterio, MemoryFile, transform_from_bounds,
                                transform_geom, catalog, group, img_dir, mask_dir, w), w
        except Exception as e:  # noqa: BLE001 — fall back to per-geometry for this region
            log.warning("grouped fetch failed for a %d-AOI region, falling back "
                        "per-geometry: %s", len(group), e)
            w.append(f"region of {len(group)} chip(s) fell back to per-geometry fetch: {e}")
            res = []
            for a in group:
                r = _fetch_one_geometry(rio_mods, rasterio, transform_from_bounds,
                                        transform_geom, catalog, a, img_dir, mask_dir, w)
                if r:
                    res.append(r)
            return res, w

    def _run_single(a: _AOI):
        w: list[str] = []
        r = _fetch_one_geometry(rio_mods, rasterio, transform_from_bounds, transform_geom,
                                catalog, a, img_dir, mask_dir, w)
        return ([r] if r else []), w

    if config.FETCH_GROUP_ENABLED and len(aois) > 1:
        units = _group_aois(aois, config.FETCH_GROUP_CELL_KM * 1000)
        worker = _run_grouped
        log.info("grouped %d geometries into %d region(s) (~%.0fkm cells), fetching on %d worker(s)",
                 len(aois), len(units), config.FETCH_GROUP_CELL_KM, max(1, config.FETCH_MAX_WORKERS))
    else:
        units = aois
        worker = _run_single

    workers = max(1, min(config.FETCH_MAX_WORKERS, len(units) or 1))
    if workers == 1:
        outs = [worker(u) for u in units]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(worker, units))  # map preserves unit order → deterministic warnings

    results: list[tuple[Path, FetchedChip]] = []
    for res, w in outs:
        results.extend(res)
        warnings_out.extend(w)

    # Keep output ordered by geometry index (chip_000, chip_001, ...) regardless
    # of the region grouping, so callers see a stable, source-aligned sequence.
    results.sort(key=lambda pm: pm[0].name)
    chip_paths = [p for p, _ in results]
    meta = [m for _, m in results]

    # Spatial-block split manifest, honored by the generated wrapper's loader
    # (maps chip filename -> split) so training never leaks overlapping chips.
    if meta:
        manifest = {m.path.name: m.split for m in meta}
        try:
            (workdir / "splits.json").write_text(json.dumps(manifest, indent=0))
        except OSError as e:
            log.warning("could not write splits.json manifest: %s", e)

    if not chip_paths:
        raise FetchImageryError(
            "could not fetch satellite imagery for any of the archive's geometries: "
            + "; ".join(warnings_out[:3])
        )
    return chip_paths, meta, warnings_out, class_info
