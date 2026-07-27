"""Archive inspection: safe extraction, structure detection, per-band stats.

Deterministic — no LLM involvement. The agent tool wrapper closes over the
run's archive path; Claude never supplies filesystem paths.
"""

from __future__ import annotations

import csv
import json
import random
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config
from schemas import BandStat, Inspection, VectorSummary

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
# Common geospatial vector formats — parsed as supplementary context only
# (footprints/AOI, geometry types, attribute schema). Never drives
# structure/labelType detection, which stays keyed off the raster chips.
VECTOR_EXTS = {".geojson", ".kml", ".json"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}
MASK_DIR_NAMES = {"masks", "labels", "mask", "label", "annotations", "gt"}
IMG_DIR_NAMES = {"images", "image", "imgs", "img", "chips", "data"}

_MAX_SAMPLE = 128
_MAX_RATIO = 200  # per-entry compression-bomb ratio guard


class InspectError(ValueError):
    """Raised for malformed / hostile archives; message is user-safe."""


def _safe_extract(archive_path: Path, workdir: Path) -> list[Path]:
    """Extract image/CSV entries with zip-slip + bomb guards. Returns files."""
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        if len(infos) > config.MAX_ARCHIVE_ENTRIES:
            raise InspectError(f"archive has {len(infos)} entries (max {config.MAX_ARCHIVE_ENTRIES})")
        total = sum(i.file_size for i in infos)
        if total > config.MAX_UNCOMPRESSED_BYTES:
            raise InspectError("archive expands beyond the 2 GB limit")
        for info in infos:
            if info.is_dir():
                continue
            name = info.filename
            ext = Path(name).suffix.lower()
            if ext not in IMAGE_EXTS and ext != ".csv" and ext not in VECTOR_EXTS:
                continue
            if info.compress_size and info.file_size / max(info.compress_size, 1) > _MAX_RATIO:
                raise InspectError(f"suspicious compression ratio on {name}")
            target = (workdir / name).resolve()
            if not target.is_relative_to(workdir.resolve()):
                raise InspectError(f"unsafe path in archive: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    if not extracted:
        raise InspectError("no image, CSV, or vector (GeoJSON/KML) files found in the archive")
    return extracted


def _read_image(path: Path) -> np.ndarray:
    """Read as float64 array shaped (H, W, C)."""
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(str(path))
    else:
        from PIL import Image

        arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3 and arr.shape[0] <= 32 and arr.shape[0] < arr.shape[2]:
        # channels-first (C, H, W) -> (H, W, C)
        arr = np.moveaxis(arr, 0, 2)
    return arr.astype(np.float64)


def _bbox_of(coords, box: list[float]) -> None:
    """Recursively fold a GeoJSON coordinates array into [minx, miny, maxx, maxy]."""
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        x, y = coords[0], coords[1]
        box[0] = min(box[0], x); box[1] = min(box[1], y)
        box[2] = max(box[2], x); box[3] = max(box[3], y)
    else:
        for c in coords:
            _bbox_of(c, box)


def _parse_geojson(path: Path) -> tuple[VectorSummary, list[tuple[dict, dict]]] | None:
    """Parse a .geojson/.json file into (VectorSummary, [(geometry, properties)]).
    Returns None if the file isn't actually GeoJSON (so plain .json config
    files are skipped)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    typ = data.get("type")
    if typ == "FeatureCollection":
        features = data.get("features") or []
    elif typ == "Feature":
        features = [data]
    elif typ in {"Point", "MultiPoint", "LineString", "MultiLineString",
                 "Polygon", "MultiPolygon", "GeometryCollection"}:
        features = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
        return None  # not recognizable GeoJSON — likely an unrelated .json

    geom_types: Counter = Counter()
    attr_fields: set[str] = set()
    box = [float("inf"), float("inf"), float("-inf"), float("-inf")]
    geometries: list[tuple[dict, dict]] = []
    for feat in features:
        geom = (feat or {}).get("geometry") or {}
        props = feat.get("properties") or {}
        if geom.get("type"):
            geom_types[geom["type"]] += 1
            geometries.append((geom, props))
        coords = geom.get("coordinates")
        if coords is not None:
            try:
                _bbox_of(coords, box)
            except (TypeError, IndexError):
                pass
        attr_fields.update(props.keys())

    crs = None
    crs_block = data.get("crs")
    if isinstance(crs_block, dict):
        crs = ((crs_block.get("properties") or {}).get("name"))

    summary = VectorSummary(
        filename=path.name,
        format="geojson",
        geometryTypes=sorted(geom_types),
        featureCount=len(features),
        attributeFields=sorted(attr_fields)[:20],
        bbox=[round(v, 6) for v in box] if box[0] != float("inf") else None,
        crs=crs,
    )
    return summary, geometries


def _local_tag(elem: ET.Element) -> str:
    tag = elem.tag
    return tag.split("}", 1)[1] if "}" in tag else tag


_KML_GEOM_TAGS = {"Point", "LineString", "Polygon"}  # top-level geometries only


def _parse_kml(path: Path) -> tuple[VectorSummary, list[tuple[dict, dict]]] | None:
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError):
        return None
    root = tree.getroot()

    geom_types: Counter = Counter()
    attr_fields: set[str] = set()
    feature_count = 0
    box = [float("inf"), float("inf"), float("-inf"), float("-inf")]
    geometries: list[tuple[dict, dict]] = []

    for placemark in root.iter():
        if _local_tag(placemark) != "Placemark":
            continue
        feature_count += 1
        placemark_pts: list[list[float]] = []
        placemark_geom_types: set[str] = set()
        placemark_props: dict[str, str] = {}
        for child in placemark.iter():
            tag = _local_tag(child)
            if tag in _KML_GEOM_TAGS:
                geom_types[tag] += 1
                placemark_geom_types.add(tag)
            if tag == "coordinates" and child.text:
                for triplet in child.text.split():
                    parts = triplet.split(",")
                    if len(parts) >= 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                        except ValueError:
                            continue
                        box[0] = min(box[0], x); box[1] = min(box[1], y)
                        box[2] = max(box[2], x); box[3] = max(box[3], y)
                        placemark_pts.append([x, y])
            if tag == "SimpleField" and "name" in child.attrib:
                attr_fields.add(child.attrib["name"])
            if tag == "Data" and "name" in child.attrib:
                attr_fields.add(child.attrib["name"])
                val_elem = next((c for c in child if _local_tag(c) == "value"), None)
                if val_elem is not None and val_elem.text:
                    placemark_props[child.attrib["name"]] = val_elem.text.strip()
        if placemark_pts:
            if placemark_geom_types == {"Polygon"}:
                # Preserve real ring structure (not just a point cloud) so a
                # polygon label already in the KML — a footprint, a boundary —
                # can be rasterized into a real segmentation mask later,
                # instead of being flattened down to a centroid.
                geometries.append(({"type": "Polygon", "coordinates": [placemark_pts]}, placemark_props))
            elif placemark_geom_types == {"LineString"}:
                geometries.append(({"type": "LineString", "coordinates": placemark_pts}, placemark_props))
            else:
                # Point, or mixed/MultiGeometry placemarks — a synthetic
                # MultiPoint is still enough to derive a centroid/AOI from.
                geometries.append(({"type": "MultiPoint", "coordinates": placemark_pts}, placemark_props))

    if feature_count == 0 and not geom_types:
        return None

    summary = VectorSummary(
        filename=path.name,
        format="kml",
        geometryTypes=sorted(geom_types),
        featureCount=feature_count,
        attributeFields=sorted(attr_fields)[:20],
        bbox=[round(v, 6) for v in box] if box[0] != float("inf") else None,
        crs="EPSG:4326",  # KML coordinates are always lon/lat WGS84 per spec
    )
    return summary, geometries


def _collect_vector_context(files: list[Path]) -> tuple[list[VectorSummary], list[tuple[dict, dict]]]:
    """Returns (summaries, geometries) where each geometry entry is
    (geometry_dict, properties_dict) — properties travel with the geometry
    so the satellite-fetch path can pick a class/label attribute per feature."""
    vector_files = [f for f in files if f.suffix.lower() in VECTOR_EXTS]
    summaries: list[VectorSummary] = []
    all_geometries: list[tuple[dict, dict]] = []
    for f in vector_files[: config.MAX_VECTOR_FILES_PARSED]:
        ext = f.suffix.lower()
        parsed = _parse_kml(f) if ext == ".kml" else _parse_geojson(f)
        if parsed is not None:
            summary, geometries = parsed
            summaries.append(summary)
            all_geometries.extend(geometries)
    return summaries, all_geometries


def _detect_structure(files: list[Path], workdir: Path) -> tuple[str, dict]:
    """Classify archive layout. Returns (structure, extras)."""
    images = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
    csvs = [f for f in files if f.suffix.lower() == ".csv"]

    rel = {f: f.relative_to(workdir) for f in images}

    # image/mask sibling dirs (possibly under split dirs)
    by_parent: dict[str, list[Path]] = defaultdict(list)
    for f, r in rel.items():
        by_parent[r.parent.name.lower()].append(f)
    mask_files = [f for d, fs in by_parent.items() if d in MASK_DIR_NAMES for f in fs]
    img_files = [f for d, fs in by_parent.items() if d in IMG_DIR_NAMES for f in fs]
    if mask_files and img_files:
        mask_stems = {f.stem.replace("_mask", "").replace("_label", "") for f in mask_files}
        img_stems = {f.stem for f in img_files}
        if len(mask_stems & img_stems) >= max(1, len(img_stems) // 2):
            return "image-mask-pairs", {"images": img_files, "masks": mask_files}

    # folders-per-class: {split}/{class}/img or {class}/img
    class_of: dict[Path, str] = {}
    split_of: dict[Path, str] = {}
    ok = 0
    for f, r in rel.items():
        parts = [p.lower() for p in r.parts[:-1]]
        if not parts:
            continue
        cls = r.parts[-2]
        if cls.lower() in SPLIT_NAMES or cls.lower() in IMG_DIR_NAMES:
            continue
        class_of[f] = cls
        for p in parts:
            if p in SPLIT_NAMES:
                split_of[f] = "val" if p.startswith("val") else p
                break
        ok += 1
    if ok >= len(images) * 0.8 and len(set(class_of.values())) >= 2:
        return "folders-per-class", {"class_of": class_of, "split_of": split_of, "images": images}

    # CSV labels: a csv with a filename-ish column
    for c in csvs:
        try:
            with open(c, newline="") as fh:
                rows = list(csv.DictReader(fh))
        except (OSError, csv.Error):
            continue
        if not rows:
            continue
        cols = {k.lower(): k for k in rows[0] if k}
        fname_col = next((cols[k] for k in ("filename", "file", "image", "path", "id") if k in cols), None)
        label_col = next((cols[k] for k in ("label", "labels", "class", "category", "y") if k in cols), None)
        if fname_col and label_col:
            labels = [r[label_col] for r in rows if r.get(label_col)]
            multilabel = any(("," in l or ";" in l) for l in labels)
            return "csv-labels", {
                "images": images,
                "labels": labels,
                "multilabel": multilabel,
                "csv": c,
            }

    return "flat", {"images": images}


def inspect_archive(archive_path: Path, workdir: Path, *, filename: str, size_bytes: int) -> Inspection:
    workdir = workdir.resolve()
    files = _safe_extract(archive_path, workdir)
    vector_context, vector_geometries = _collect_vector_context(files)
    structure, extras = _detect_structure(files, workdir)
    images: list[Path] = extras["images"]

    imported_satellite = False
    import_note: str | None = None
    fetch_class_info = None
    # Only the satellite-fetch path below produces per-chip fetch metadata; for
    # uploaded archives it stays empty and the split counters fall back to
    # {"all": len(images)}. Must be bound here so the image-mask-pairs / csv
    # branches don't hit an UnboundLocalError on uploaded (non-fetched) data.
    fetch_meta: list = []
    if not images:
        if not vector_geometries:
            raise InspectError(
                "archive contains no images, and no GeoJSON/KML geometries to "
                "fetch satellite imagery for"
            )
        from tools.fetch_imagery import FetchImageryError, fetch_chips_for_geometries
        try:
            images, fetch_meta, fetch_warnings, fetch_class_info = fetch_chips_for_geometries(
                vector_geometries, workdir)
        except FetchImageryError as e:
            raise InspectError(
                f"archive has no images — tried auto-fetching satellite imagery "
                f"for its {len(vector_geometries)} geometries but that failed: {e}"
            ) from e
        imported_satellite = True
        extras = {"images": images}

        mask_paths = [m.mask_path for m in fetch_meta if m.mask_path is not None]
        if mask_paths:
            # the vector file's polygons already ARE labels (footprints,
            # boundaries) — rasterized into real masks, this is a
            # segmentation dataset, not a bag of unlabeled thumbnails.
            # Reuse the existing image-mask-pairs path verbatim.
            structure = "image-mask-pairs"
            extras["masks"] = mask_paths
        else:
            structure = "fetched-satellite"

        clouds = [m.cloud_cover for m in fetch_meta if m.cloud_cover is not None]
        cloud_note = f", median input cloud cover {sorted(clouds)[len(clouds) // 2]:.0f}%" if clouds else ""
        n_composite = sum(1 for m in fetch_meta if m.composite)
        n_clipped = sum(1 for m in fetch_meta if m.aoi_clipped)
        import_note = (
            f"No chips were uploaded — auto-fetched {len(images)} Sentinel-2 RGB chip"
            f"{'s' if len(images) != 1 else ''} from Microsoft Planetary Computer "
            f"around the archive's {len(vector_geometries)} vector geometr"
            f"{'ies' if len(vector_geometries) != 1 else 'y'}{cloud_note}."
        )
        if mask_paths:
            import_note += (
                f" {len(mask_paths)} of the geometries were polygons, so their labeled "
                "footprints were rasterized into segmentation masks aligned to the fetched "
                "imagery — this is a segmentation task, not unlabeled classification chips."
            )
            if fetch_class_info:
                import_note += (
                    f" Classes came from the '{fetch_class_info.field}' attribute "
                    f"({len(fetch_class_info.class_ids)} classes + background)."
                )
        if n_composite:
            import_note += (
                f" {n_composite} chip{'s' if n_composite != 1 else ''} had no single "
                "cloud-free scene, so they're cloud-masked medians of several dates instead."
            )
        if n_clipped:
            import_note += (
                f" {n_clipped} geometr{'ies' if n_clipped != 1 else 'y'} were larger than "
                "the max fetch AOI, so the chip is centered on them but doesn't cover the whole shape."
            )
        if fetch_warnings:
            n_skipped = len(vector_geometries) - len(images)
            if n_skipped > 0:
                import_note += f" ({n_skipped} geometr{'ies' if n_skipped != 1 else 'y'} skipped — see warnings.)"

    # --- labels / class balance ---
    class_balance: dict[str, float] = {}
    splits: dict[str, int] = {}
    num_classes: int | None = None
    label_type = "unlabeled"

    if structure == "folders-per-class":
        counts = Counter(extras["class_of"].values())
        total = sum(counts.values())
        class_balance = {k: round(v / total, 3) for k, v in counts.most_common()}
        num_classes = len(counts)
        label_type = "binary" if num_classes == 2 else f"multiclass ({num_classes})"
        splits = dict(Counter(extras["split_of"].values())) or {"all": len(images)}
    elif structure == "csv-labels":
        if extras["multilabel"]:
            label_type = "multilabel"
            all_labels = Counter(
                t.strip() for l in extras["labels"] for t in l.replace(";", ",").split(",") if t.strip()
            )
        else:
            all_labels = Counter(extras["labels"])
        total = sum(all_labels.values())
        class_balance = {k: round(v / total, 3) for k, v in all_labels.most_common(12)}
        num_classes = len(all_labels)
        if not extras["multilabel"]:
            label_type = "binary" if num_classes == 2 else f"multiclass ({num_classes})"
        splits = dict(Counter(m.split for m in fetch_meta)) or {"all": len(images)}
    elif structure == "image-mask-pairs":
        label_type = "masks"
        if fetch_class_info:
            # exact counts from the real feature attributes — more accurate
            # than sampling a handful of mask files, and gives real per-class
            # proportions for free.
            num_classes = len(fetch_class_info.class_ids) + 1  # + background
            total = sum(fetch_class_info.counts.values())
            class_balance = {name: round(n / total, 3) for name, n in
                              sorted(fetch_class_info.counts.items(), key=lambda kv: -kv[1])}
        else:
            rng = random.Random(0)
            vals: set[int] = set()
            for m in rng.sample(extras["masks"], min(8, len(extras["masks"]))):
                try:
                    vals.update(np.unique(_read_image(m)).astype(int).tolist())
                except (OSError, ValueError):
                    continue
            num_classes = len(vals) if vals else None
        splits = dict(Counter(m.split for m in fetch_meta)) or {"all": len(images)}
    else:
        splits = dict(Counter(m.split for m in fetch_meta)) or {"all": len(images)}

    # --- sampled per-band stats (Welford) ---
    rng = random.Random(0)
    sample = rng.sample(images, min(_MAX_SAMPLE, len(images)))
    count = 0
    mean = m2 = None
    mins = maxs = None
    shapes: Counter = Counter()
    dtype = "unknown"
    for p in sample:
        try:
            arr = _read_image(p)
        except (OSError, ValueError):
            continue
        if dtype == "unknown":
            ext = p.suffix.lower()
            if ext in {".tif", ".tiff"}:
                import tifffile

                dtype = str(tifffile.imread(str(p)).dtype)
            else:
                from PIL import Image

                dtype = str(np.asarray(Image.open(p)).dtype)
        shapes[(arr.shape[0], arr.shape[1], arr.shape[2])] += 1
        px = arr.reshape(-1, arr.shape[2])
        if mean is None:
            nb = arr.shape[2]
            mean = np.zeros(nb)
            m2 = np.zeros(nb)
            mins = np.full(nb, np.inf)
            maxs = np.full(nb, -np.inf)
        if arr.shape[2] != mean.shape[0]:
            continue  # inconsistent band count; skip outliers
        # batched Welford update over all pixels of this chip
        n_new = px.shape[0]
        batch_mean = px.mean(axis=0)
        batch_var = px.var(axis=0)
        delta = batch_mean - mean
        tot = count + n_new
        mean = mean + delta * n_new / tot
        m2 = m2 + batch_var * n_new + delta**2 * count * n_new / tot
        count = tot
        mins = np.minimum(mins, px.min(axis=0))
        maxs = np.maximum(maxs, px.max(axis=0))

    if mean is None:
        raise InspectError("could not read any image in the archive")

    std = np.sqrt(m2 / max(count, 1))
    n_bands = mean.shape[0]
    (h, w, _), _ = shapes.most_common(1)[0]
    image_size = h if h == w else None

    # canonical band-name guess for common layouts
    band_names: list[str | None] = [None] * n_bands
    if n_bands == 3:
        band_names = ["red", "green", "blue"]
    elif n_bands == 4:
        band_names = ["red", "green", "blue", "nir"]
    elif n_bands == 13:
        band_names = [
            "coastal_aerosol", "blue", "green", "red", "red_edge_1", "red_edge_2",
            "red_edge_3", "nir", "red_edge_4", "water_vapor", "cirrus", "swir_1", "swir_2",
        ]

    band_stats = [
        BandStat(
            index=i, name=band_names[i],
            mean=round(float(mean[i]), 3), std=round(float(std[i]), 3),
            min=round(float(mins[i]), 3), max=round(float(maxs[i]), 3),
        )
        for i in range(n_bands)
    ]

    # --- sensor heuristic ---
    vmax = float(maxs.max())
    resolution = None
    try:  # optional rasterio pass for GSD/CRS on a couple of samples
        import rasterio

        for p in sample[:3]:
            if p.suffix.lower() in {".tif", ".tiff"}:
                with rasterio.open(p) as src:
                    if src.crs and src.res and src.res[0] not in (0, 1):
                        resolution = f"{src.res[0]:g} m/px"
                        break
    except ImportError:
        pass

    if n_bands >= 10 and "int16" in dtype and vmax <= 20000:
        sensor, evidence = "s2", f"{n_bands} bands, {dtype}, DN range ≤ {vmax:.0f} — Sentinel-2 MSI signature"
    elif n_bands <= 4 and dtype == "uint8" and vmax <= 255:
        sensor, evidence = "aerial", f"{n_bands}-band uint8 0–255 — aerial/web RGB imagery"
    elif n_bands <= 4 and "int16" in dtype and 1000 <= vmax <= 20000:
        sensor, evidence = "s2", f"{n_bands}-band {dtype} with Sentinel-2-like DN range (max {vmax:.0f})"
    elif n_bands <= 2 and "float" in dtype:
        sensor, evidence = "sar", f"{n_bands} float band(s) — SAR-like backscatter"
    else:
        sensor, evidence = "unknown", f"{n_bands} bands, {dtype}, max value {vmax:.0f}"

    return Inspection(
        filename=filename,
        chips=len(images),
        sizeMB=round(size_bytes / 1048576, 1),
        bands=n_bands,
        dtype=dtype,
        imageSize=image_size,
        resolution=resolution,
        sensorGuess=sensor,
        sensorEvidence=evidence,
        labelType=label_type,
        numClasses=num_classes,
        classBalance=class_balance,
        splits=splits,
        structure=structure,
        bandStats=band_stats,
        vectorContext=vector_context,
        importedSatellite=imported_satellite,
        importNote=import_note,
    )
