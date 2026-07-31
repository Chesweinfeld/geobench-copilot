"""The GeoBench Copilot agent: Claude (tool runner) over deterministic tools.

Claude writes all prose; deterministic tools compute every number. The final
payload is assembled server-side from cached tool outputs — Claude submits
only the parts it authored (prose, picks, notes), so numbers can't drift.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import anthropic
from anthropic import beta_async_tool
from pydantic import BaseModel, Field, ValidationError

import config
from runs import Run, RunManager
from schemas import (
    FinalPayload,
    LeaderboardBar,
    MatchedDataset,
    RankedModel,
    Recommendation,
    RunModel,
    ScatterPt,
)
from tools import matching, results
from tools.inspect import InspectError, inspect_archive as _inspect_impl
from tools.wrapper import generate_wrapper as _wrapper_impl

log = logging.getLogger("geobench.agent")

STEP_LABELS = [
    "Inspecting dataset",
    "Characterizing task",
    "Selecting eval metric",
    "Matching GeoBench datasets",
    "Ranking frozen backbones",
    "Checking bands & normalization",
    "Generating torchgeo-bench wrapper",
    "Composing recommendation",
]

# server backstop: domain tool -> the step it implies
_TOOL_STEP = {
    "inspect_archive": 0,
    "match_datasets": 3,
    "query_results": 4,
    "generate_wrapper": 6,
    "submit_recommendation": 7,
}

SYSTEM_PROMPT = """You are GeoBench Copilot. You help a non-ML geospatial domain \
expert benchmark their uploaded dataset against torchgeo-bench (frozen foundation-model \
backbones + linear probes on GeoBench-style datasets). Your tools compute all facts; \
you never invent numbers — every numeric claim in your step details, notes, and final \
submission must come from a tool result.

Work through exactly these 8 steps, in order. Before or right after the relevant tool \
call, call emit_step(step, detail) with a one-sentence detail (<=160 chars) containing \
concrete numbers from tool results:
 0 Inspecting dataset — call inspect_archive first, then emit with its facts
 1 Characterizing task — your reasoning: task type, sensor, object scale, and the \
task's real-world domain (urban / agriculture / forestry / energy_infrastructure / \
disaster_water / landcover_general / other — see match_datasets' domain arg) from \
the user's task description, not just the pixels. Pick the most specific task type \
from classification / multilabel_classification / regression / segmentation / \
instance_segmentation / change_detection / object_detection. If the inspection returned \
vectorContext (GeoJSON/KML), use its geometry types / bbox / attribute fields as \
extra evidence (e.g. AOI footprint, feature schema) but never let it override the \
chip-based label/structure detection.
 2 Selecting eval metric — apply the metric rules below, say why
 3 Matching GeoBench datasets — call match_datasets, passing the domain you settled \
on in step 1. Its `sim` score and Domain facet are a coarse first pass, not the \
final answer: read every candidate's `description` and reason for yourself about \
which 3 are the closest real-world match to the user's task, even if that means \
picking a lower-`sim` candidate over a higher-`sim` one that only superficially \
matches (same bands/scale, unrelated subject).
 4 Ranking frozen backbones — call query_results on the matched dataset names, \
passing preferred_metric = the metric you picked in step 2 (e.g. "macro_f1" for an \
imbalanced binary/multiclass task, "accuracy" otherwise). query_results will use it \
if any matched dataset actually reports it, and tells you via metric_note if none \
do — in that case rank by whatever it fell back to and say so plainly rather than \
claiming a metric was used that the data doesn't have.
 5 Checking bands & normalization — compare bandspec_zscore vs model_native; \
recommend keeping the user's band set
 6 Generating torchgeo-bench wrapper — call generate_wrapper. Always call it, even \
when inspect_archive's importedSatellite is true: the fetched chips are real raster \
imagery and inspect_archive already measured real per-band stats on them (see \
bandStats/chips/dtype in its result) — pass those straight through exactly like you \
would for an uploaded archive. Never skip or refuse this step reasoning that a \
vector-only upload "has no raster chips to measure" — by the time you see the \
inspection result, it already does.
 7 Composing recommendation — call submit_recommendation

Metric rules: single-label classification -> accuracy; if the class balance is more \
skewed than 60/40, name macro-F1 as the primary metric with accuracy alongside (and \
say why); multilabel -> micro_mAP; segmentation -> mIoU. The benchmark leaderboards \
report accuracy or micro_mAP only — never claim macro-F1 numbers exist in the results.

The run commands and all result numbers are produced by tools. Valid torchgeo-bench \
config keys are model=, dataset.names, dataset.bands, dataset.normalization, \
dataset.image_size, eval.knn_k. There is no eval.metrics or report.format key — never \
mention them.

recommendation.expected is NOT a prediction of how the model will perform on the \
user's own data — it is the real score that model got in query_results on the \
closest-matched benchmark dataset(s). Never call it "expected accuracy" or imply it's \
a forecast. Always phrase it as what it is: e.g. "<model> scored <expected> <metric> \
on <matched dataset name(s)>, the closest public benchmark to your task" — name the \
actual matched dataset(s) it came from, every time you state the number, in prose and \
in each model's `note`. The transferred-scores caveat exists precisely because this is \
someone else's result on a similar dataset, not a forecast for this user's data.

Audience: plain, concrete, jargon-light prose. In submit_recommendation, provide \
glossary definitions (1-2 sentences each) for every technical term you used that a \
non-ML reader wouldn't know. Always include these caveats (reworded for the user's \
case) plus any case-specific ones: transferred-scores disclaimer (benchmark numbers \
come from similar datasets, not the user's own); avoid model_native normalization; \
metric/imbalance warning when the classes are imbalanced. If the task is segmentation \
or no matched dataset has results, say clearly that no leaderboard numbers exist yet \
and the ranking is indicative only. If inspect_archive's importedSatellite is true, \
add a caveat that the chips are freshly fetched Sentinel-2 scenes around the uploaded \
geometries, not the user's own imagery, so the benchmark-match accuracy expectations \
are illustrative only. If labelType is "masks" (the geometries were polygons — a real \
footprint/boundary label rasterized per chip), do NOT call this data unlabeled: say \
plainly that the polygons themselves are the segmentation ground truth and this is a \
real, usable labeled dataset, just not benchmarked against this exact imagery yet. \
Only call it unlabeled when labelType genuinely is "unlabeled" (point/line geometries \
with no area to rasterize). Either way, the generated wrapper and its band stats are \
real measurements on those fetched chips and are fully usable as-is; that part is \
never "illustrative only".

After submit_recommendation returns "ok", reply with one short sentence and stop."""


class _SubmitMatched(BaseModel):
    name: str
    desc: str = Field(min_length=10)


class _SubmitModel(BaseModel):
    name: str
    note: str = Field(min_length=10)
    chips: list[str] = Field(default_factory=list, max_length=4)
    top: bool = False
    budget: bool = False


class SubmitInput(BaseModel):
    recommendation: Recommendation
    matched: list[_SubmitMatched] = Field(min_length=1, max_length=3)
    models: list[_SubmitModel] = Field(min_length=3, max_length=5)
    wrapper_note: str = Field(min_length=10)
    caveats: list[str] = Field(min_length=3)
    glossary: dict[str, str] = Field(default_factory=dict)


def _tolerant_match(value: float, allowed: set[float], tol: float = 0.005) -> bool:
    return any(abs(value - a) <= tol for a in allowed)


# Crude keyword fallback for match_datasets' domain arg, used only when the
# fallback path has to guess without Claude's actual reasoning over the task
# description (see _fallback_submit below).
_DOMAIN_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("energy_infrastructure", ("solar", "wind farm", "wind turbine", "kiln", "power plant",
                                "factory", "industrial", "pipeline", "substation")),
    ("urban", ("urban", "building", "city", "rooftop", "road", "infrastructure")),
    ("agriculture", ("crop", "farm", "field", "agricult", "orchard", "vineyard")),
    ("forestry", ("forest", "tree", "deforest", "logging", "canopy")),
    ("disaster_water", ("flood", "fire", "burn", "wildfire", "disaster", "landslide", "drought")),
    ("landcover_general", ("land cover", "land-cover", "landcover", "land use")),
]


def _guess_domain(task: str) -> str:
    t = (task or "").lower()
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(kw in t for kw in keywords):
            return domain
    return "other"


def _guess_preferred_metric(insp) -> str:
    """Mirror the system prompt's metric rules deterministically, for the
    fallback path only (Claude normally makes this call itself in step 2)."""
    if insp.labelType == "masks":
        return "mIoU"
    if insp.labelType == "multilabel":
        return "micro_mAP"
    if insp.classBalance and max(insp.classBalance.values()) > 0.6:
        return "macro_f1"  # skewed class balance — accuracy alone would mislead
    return "accuracy"


def _build_payload(run: Run, sub: SubmitInput) -> FinalPayload:
    """Assemble the FinalPayload from cached tool outputs + the authored
    (or synthesized) parts in `sub`. Shared by submit_recommendation and the
    no-submission fallback so both paths produce an identical shape."""
    ranked_by_name = {r["name"]: r for r in run.query_result.get("ranked", [])}
    metric = run.query_result.get("metric") or sub.recommendation.metric
    matched = [
        MatchedDataset(
            name=m.name,
            sim=run.match_result[m.name]["sim"],
            desc=m.desc,
            facets=run.match_result[m.name]["facets"],
        )
        for m in sub.matched
    ]
    models = [
        RankedModel(
            rank=i + 1,
            name=mod.name,
            top=mod.top,
            budget=mod.budget,
            expected=ranked_by_name[mod.name]["linear_mean"],
            metric=metric,
            gmacs=ranked_by_name[mod.name]["gflops"],
            note=mod.note,
            chips=mod.chips,
        )
        for i, mod in enumerate(sub.models)
    ]
    leaderboard = [
        LeaderboardBar(name=r["name"], acc=r["acc"],
                       top=r["name"] == sub.recommendation.headlineModel
                       or any(m.top and m.name == r["name"] for m in sub.models))
        for r in run.query_result.get("leaderboard6", [])
    ]
    scatter = [
        ScatterPt(name=r["name"], label=r["name"].replace("_", " "),
                  gmacs=r["gflops"], acc=r["acc"],
                  top=any(m.top and m.name == r["name"] for m in sub.models))
        for r in run.query_result.get("scatter6", [])
        if r["gflops"] is not None
    ]
    wrapper = run.wrapper.model_copy(update={"note": sub.wrapper_note})

    return FinalPayload(
        recommendation=sub.recommendation,
        matched=matched,
        models=models,
        leaderboard=leaderboard,
        scatter=scatter,
        wrapper=wrapper,
        runModels=getattr(run, "run_models", []),
        caveats=sub.caveats,
        glossary=sub.glossary,
    )


def run_error_message(exc: BaseException, run_id: str) -> str:
    """Build the user-facing message for an otherwise unhandled run failure.

    Never returns an empty string. Plenty of exceptions stringify to "" —
    MemoryError being the usual one on a small instance — and the UI falls back
    to a bare "analysis failed" on an empty message, which tells the viewer
    nothing and hides the fact that a full traceback is already in the server
    log. Always name the exception type, and tag the run id so a user's report
    can be matched to its "agent run failed" traceback.
    """
    if isinstance(exc, MemoryError):
        detail = ("the server ran out of memory on this dataset — it may be "
                  "too large for this instance")
    else:
        detail = str(exc).strip() or f"unexpected {type(exc).__name__} on the server"
    return f"{detail} (run {run_id})"


async def run_agent(run: Run, manager: RunManager) -> None:
    def emit(event_type: str, data: dict) -> None:
        manager.emit(run, event_type, data)

    def ensure_step(step: int) -> None:
        if step not in run.emitted_steps:
            run.emitted_steps.add(step)
            emit("step_start", {"step": step, "label": STEP_LABELS[step]})

    # ---- tools (closures over `run`; Claude never passes paths) ----

    @beta_async_tool
    async def emit_step(step: int, detail: str) -> str:
        """Report progress on one of the 8 canonical steps.

        Args:
            step: Step index 0-7 (0 Inspecting dataset ... 7 Composing recommendation).
            detail: One sentence (<=160 chars) with concrete numbers from tool results.
        """
        if not 0 <= step <= 7:
            return "error: step must be 0-7"
        ensure_step(step)
        emit("step_detail", {"step": step, "detail": detail[:200]})
        return "ok"

    @beta_async_tool
    async def inspect_archive() -> str:
        """Inspect the user's uploaded archive: chip count, bands, dtype, image size,
        per-band statistics, label structure, class balance, and a sensor guess with
        evidence. If the archive also contains GeoJSON/KML files, their geometry
        types, feature counts, attribute fields, and bbox are returned as
        supplementary vectorContext — informational only, it never changes the
        label/structure detection which is keyed off the raster chips.

        If the archive has vector geometries but no raster chips at all, real
        Sentinel-2 imagery is auto-fetched around each geometry from Microsoft
        Planetary Computer; when that happens importedSatellite=true and
        importNote explains what was pulled and from where. The returned
        chips/bands/dtype/bandStats describe those fetched images exactly like
        they would any uploaded archive — they are real measured chips, not a
        placeholder, so generate_wrapper works normally on them.

        If the vector geometries were Polygons/MultiPolygons (a footprint, a
        boundary — e.g. building or settlement outlines), each one is
        rasterized into a real segmentation mask aligned to its fetched chip:
        labelType will be "masks" and structure "image-mask-pairs". This is
        NOT unlabeled data — the polygon *is* the label, so treat it as a
        genuine segmentation task (task_type="segmentation" in match_datasets,
        mIoU as the metric) with real per-chip ground truth. Only say the
        chips are "unlabeled" when labelType is actually "unlabeled" (i.e. the
        geometries were points/lines with no area to rasterize) — always mention
        the auto-fetch plainly either way, but don't undersell real mask labels
        as illustrative-only; the caveat about accuracy expectations being
        illustrative applies to the benchmark-match numbers (they're from
        similar public datasets, not measured on this exact imagery), not to
        whether the labels themselves are real. Call this first."""
        ensure_step(0)
        insp = await asyncio.to_thread(
            _inspect_impl,
            run.archive_path,
            run.workdir,
            filename=run.filename,
            size_bytes=run.size_bytes,
        )
        run.inspection = insp
        emit("inspection", insp.model_dump())
        return insp.model_dump_json()

    @beta_async_tool
    async def match_datasets(
        task_type: str,
        multilabel: bool,
        num_classes: int,
        band_count: int,
        sensor: str,
        object_scale: str,
        domain: str = "other",
    ) -> str:
        """Score all torchgeo-bench datasets against the described upload.

        Args:
            task_type: The user's task. One of "classification",
                "multilabel_classification", "regression", "segmentation",
                "instance_segmentation", "change_detection", "object_detection".
                The benchmarks themselves are only classification or
                segmentation; finer task types are scored against those two
                families with graded compatibility (e.g. object_detection is a
                partial match to segmentation). Pick the most specific one that
                fits the user's described task.
            multilabel: Whether each chip can carry multiple labels.
            num_classes: Number of classes in the user's data.
            band_count: Number of spectral bands in the user's chips.
            sensor: Sensor family: "s2", "aerial", "sar", or "unknown".
            object_scale: "object" (discrete objects), "scene" (whole-scene category),
                or "landcover" (wall-to-wall cover types).
            domain: The task's real-world theme, one of "urban", "agriculture",
                "forestry", "energy_infrastructure" (solar/wind/kilns/industrial
                facilities), "disaster_water" (floods/fires/burn scars), or
                "landcover_general" for broad wall-to-wall land-cover mapping with
                no single theme. Use "other" only if truly none of these fit.
                This is a semantic match on top of task/bands/label/scale — e.g.
                so a wind-farm task prefers m-pv4ger over an equally band/scale
                -matched but thematically unrelated benchmark.

        Returns all candidates sorted by similarity with per-facet breakdowns
        (Task / Bands / Label type / Object scale / Domain), a has_results flag,
        a one-line `description` of what the benchmark actually depicts, and its
        `domain_tags`. The `sim` ranking and Domain facet are a coarse heuristic —
        read each candidate's `description` yourself and use your own judgment
        about real-world similarity to the user's stated task before picking the
        top 3 for submit_recommendation's `matched`. You are not bound to the
        highest-`sim` order: a lower-`sim` candidate whose description is
        obviously the closer real-world match (e.g. a named benchmark for the
        exact object/scene the user described) should be preferred over a
        higher-`sim` one that's only superficially similar (same bands/scale but
        a different subject). Say in each match's `desc` (in submit_recommendation)
        why it's a good real-world fit, not just that it scored well.
        """
        ensure_step(3)
        scored = matching.match_datasets(
            task_type, multilabel, num_classes, band_count, sensor, object_scale,
            domain=domain, datasets_with_results=results.datasets_with_results(),
        )
        run.match_result = {d["name"]: d for d in scored}
        emit("tool_result", {"step": 3, "tool": "match_datasets",
                             "summary": [{"name": d["name"], "sim": d["sim"]} for d in scored[:5]]})
        return json.dumps(scored[:10])

    @beta_async_tool
    async def query_results(
        datasets: list[str],
        bands: str = "rgb",
        normalization: str = "bandspec_zscore",
        preferred_metric: str | None = None,
    ) -> str:
        """Rank frozen backbones by mean score over the given benchmark
        datasets, joined with compute cost (gflops, params).

        Args:
            datasets: Benchmark dataset names to aggregate over (use the matched names).
            bands: "rgb" or "all".
            normalization: "bandspec_zscore" (recommended) or "model_native".
            preferred_metric: The metric you selected in step 2 (e.g. "accuracy",
                "macro_f1", "micro_mAP", "mIoU"). Used to rank models only if at
                least one matched dataset actually reports it — real numbers
                only, nothing is ever computed that wasn't logged. Check the
                returned metric_note: if it says your preferred metric wasn't
                available, the ranking fell back to whatever was and you must
                say so plainly rather than claiming the preferred metric was used.
        """
        ensure_step(4)
        q = await asyncio.to_thread(
            results.query_results, datasets, bands, normalization, preferred_metric
        )
        run.query_result = q
        emit("tool_result", {
            "step": 4, "tool": "query_results",
            "summary": [{"name": r["name"], "acc": r["linear_mean"], "gflops": r["gflops"]}
                        for r in q["ranked"][:5]],
        })
        return json.dumps(q)

    @beta_async_tool
    async def generate_wrapper(
        dataset_name: str,
        class_name: str,
        sensor: str,
        band_names: list[str],
        wavelengths: list[float | None],
    ) -> str:
        """Generate the torchgeo-bench BenchDataset wrapper file from the measured
        band statistics, and ready-to-run benchmark commands for the top backbones.

        Args:
            dataset_name: Kebab-case name for the user's dataset, e.g. "windfarm-s2".
            class_name: Python class name, e.g. "WindfarmS2".
            sensor: Sensor family string for the BandSpec entries, e.g. "s2".
            band_names: Canonical name per measured band, e.g. ["red","green","blue"].
                Must have exactly one entry per band reported by inspect_archive,
                INCLUDING any all-zero padding bands — name those "unused" rather
                than dropping them, or this call fails. Which bands are worth
                training on is a separate decision; say it in the normalization
                note, not by shortening this list.
            wavelengths: Center wavelength in um per band (null when unknown).
                Same length rule as band_names; use null for padding bands.
        """
        if run.inspection is None:
            return "error: call inspect_archive first"
        ensure_step(6)
        wrapper = _wrapper_impl(
            dataset_name, class_name, run.inspection, sensor, band_names, wavelengths
        )
        run.wrapper = wrapper

        run_models = []
        if run.query_result and run.query_result.get("ranked"):
            for i, r in enumerate(run.query_result["ranked"][:5]):
                cmd = results.build_run_command(r["name"], dataset_name)
                if cmd is None:
                    continue
                run_models.append(RunModel(
                    id=f"m{i}",
                    model=r["hydra_model_path"] or r["name"],
                    short=r["name"],
                    gmacs=r["gflops"],
                    mins=results.estimate_minutes(run.inspection.chips, r["gflops"]),
                    command=cmd,
                ))
        run.run_models = run_models

        emit("tool_result", {"step": 6, "tool": "generate_wrapper",
                             "summary": {"filename": wrapper.filename}})
        return json.dumps({
            "wrapper_filename": wrapper.filename,
            "wrapper_code": wrapper.code,
            "runModels": [m.model_dump() for m in run_models],
        })

    @beta_async_tool
    async def submit_recommendation(payload_json: str) -> str:
        """Submit the final recommendation. Pass a JSON object with keys:
        recommendation {headlineModel, head, expected, metric, prose},
        matched [{name, desc}] (1-3, the tool's scores are attached server-side),
        models [{name, note, chips, top, budget}] (3-5, numbers attached server-side),
        wrapper_note (caption for the generated file), caveats [>=3 strings],
        glossary {term: definition}.

        recommendation.expected is the real score from query_results for the
        closest-matched benchmark dataset(s) — not a forecast for the user's own
        data. Phrase `prose` and every model's `note` accordingly (name the
        matched dataset(s) the number came from), and never call it "expected
        accuracy" as if it were a prediction.

        Args:
            payload_json: The JSON object as a string.
        """
        ensure_step(7)
        try:
            sub = SubmitInput.model_validate_json(payload_json)
        except ValidationError as e:
            return f"VALIDATION ERROR — fix and resubmit:\n{e}"

        errors: list[str] = []
        if run.match_result is None or run.query_result is None or run.wrapper is None:
            return ("VALIDATION ERROR: call match_datasets, query_results and "
                    "generate_wrapper before submitting")

        ranked_by_name = {r["name"]: r for r in run.query_result.get("ranked", [])}
        allowed_scores = {r["linear_mean"] for r in ranked_by_name.values()}
        allowed_scores |= {r["knn5_mean"] for r in ranked_by_name.values() if r["knn5_mean"]}

        for m in sub.matched:
            if m.name not in run.match_result:
                errors.append(f"matched dataset {m.name!r} was not scored by match_datasets")
        for mod in sub.models:
            if mod.name not in ranked_by_name:
                errors.append(f"model {mod.name!r} is not in query_results output")
        if ranked_by_name and not _tolerant_match(sub.recommendation.expected, allowed_scores):
            errors.append(
                f"recommendation.expected={sub.recommendation.expected} does not match any "
                f"tool-returned score (allowed: {sorted(allowed_scores, reverse=True)[:8]})"
            )
        if sum(m.top for m in sub.models) != 1:
            errors.append("exactly one model must have top=true")
        if sum(m.budget for m in sub.models) > 1:
            errors.append("at most one model may have budget=true")
        if errors:
            return "VALIDATION ERROR — fix and resubmit:\n- " + "\n- ".join(errors)

        payload = _build_payload(run, sub)
        run.final_payload = payload
        emit("final", payload.model_dump())
        return "ok"

    async def _fallback_submit() -> None:
        """Safety net: if Claude's loop ends without ever calling
        submit_recommendation (validation loops, hit max_iterations, timed
        out mid-reasoning, etc.), synthesize a minimal recommendation
        straight from whatever deterministic tool state exists — including,
        at minimum, the generated wrapper + run commands — rather than
        dead-ending the run with nothing to show. Only gives up if there
        isn't even enough to run the matcher/ranker/wrapper tools on.
        """
        if run.inspection is None:
            return  # never got past step 0 — genuinely nothing to submit
        insp = run.inspection

        if run.match_result is None:
            task_type = "segmentation" if insp.labelType == "masks" else "classification"
            multilabel = insp.labelType == "multilabel"
            await match_datasets(
                task_type=task_type, multilabel=multilabel,
                num_classes=insp.numClasses or 2, band_count=insp.bands,
                sensor=insp.sensorGuess, object_scale="scene",
                domain=_guess_domain(run.task),
            )
        if not run.match_result:
            return

        if run.query_result is None:
            top_names = [d["name"] for d in
                         sorted(run.match_result.values(), key=lambda d: -d["sim"])[:3]]
            await query_results(datasets=top_names, preferred_metric=_guess_preferred_metric(insp))
        if not run.query_result or not run.query_result.get("ranked"):
            return

        if run.wrapper is None:
            band_names = [b.name or f"band_{b.index}" for b in insp.bandStats] or ["band_0"]
            await generate_wrapper(
                dataset_name="uploaded-dataset", class_name="UploadedDataset",
                sensor=insp.sensorGuess, band_names=band_names,
                wavelengths=[None] * len(band_names),
            )
        if run.wrapper is None:
            return

        ranked = run.query_result["ranked"]
        models_n = ranked[: min(5, len(ranked))]
        if len(models_n) < 3:
            return  # too few ranked models to satisfy a real submission

        top = models_n[0]
        metric = run.query_result.get("metric") or "accuracy"
        matched_names = list(run.match_result.keys())[:3]
        matched_label = " / ".join(matched_names)
        fallback_note = (
            "Automated fallback summary — the agent's step-by-step reasoning "
            "did not finish, so this was assembled directly from the "
            "deterministic tool results rather than authored by Claude. "
            "Treat it as a provisional starting point and verify before use; "
            "the generated wrapper and run commands below are real and usable. "
            f"{top['name']} scored {top['linear_mean']:.3f} {metric} on {matched_label} "
            "— the closest public benchmark(s) to your task, not a forecast for your own data."
        )
        try:
            sub = SubmitInput(
                recommendation=Recommendation(
                    headlineModel=top["name"], head="linear probe",
                    expected=top["linear_mean"], metric=metric, prose=fallback_note,
                ),
                matched=[
                    _SubmitMatched(
                        name=n,
                        desc=f"Matched on task/band/label-type similarity ({run.match_result[n]['sim']}%).",
                    )
                    for n in matched_names
                ],
                models=[
                    _SubmitModel(
                        name=m["name"],
                        note=(f"Scored {m['linear_mean']:.3f} {metric} on {matched_label} "
                              f"(rank #{i + 1} of the matched benchmarks) — not measured on your data."),
                        chips=[], top=(i == 0), budget=False,
                    )
                    for i, m in enumerate(models_n)
                ],
                wrapper_note="Auto-generated from your archive's measured band statistics.",
                caveats=[
                    fallback_note,
                    "Benchmark numbers are transferred from similar public datasets, not measured on your own data.",
                    "Avoid model_native normalization for satellite imagery — bandspec_zscore is the safer default.",
                ] + ([run.query_result["metric_note"]] if run.query_result.get("metric_note") else []),
                glossary={},
            )
        except ValidationError as e:
            log.warning("fallback submission failed validation: %s", e)
            return

        payload = _build_payload(run, sub)
        run.final_payload = payload
        emit("final", payload.model_dump())

    # ---- runner loop ----

    try:
        # max_retries above the SDK default of 2: 529 "overloaded" spikes are
        # transient, and the SDK retries them with exponential backoff, so a few
        # more attempts let a run ride out a busy period instead of failing.
        client = anthropic.AsyncAnthropic(max_retries=8)
    except anthropic.AnthropicError:
        run.status = "error"
        emit("error", {"message": "the server has no Anthropic credentials — "
                                  "export ANTHROPIC_API_KEY and restart the backend"})
        emit("done", {})
        return
    user_msg = (
        f"The user uploaded {run.filename!r} ({run.size_bytes / 1048576:.1f} MB) and "
        f"described their task as:\n\n{run.task!r}\n\nAnalyze it end to end."
    )

    # Bracket every run in the log. A successful run used to log nothing at
    # all, so a log showing agent activity then "Shutting down" was unreadable:
    # no way to tell whether the run finished or the instance killed it. These
    # two lines are what make that answerable after the fact.
    started = time.time()
    log.info("run %s: start — %s (%.1f MB), task=%r",
             run.id, run.filename, run.size_bytes / 1048576, run.task[:120])
    run.status = "running"
    try:
        async with asyncio.timeout(config.AGENT_TIMEOUT_S):
            runner = client.beta.messages.tool_runner(
                model=config.GEOBENCH_MODEL,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                tools=[emit_step, inspect_archive, match_datasets, query_results,
                       generate_wrapper, submit_recommendation],
                messages=[{"role": "user", "content": user_msg}],
                max_iterations=config.AGENT_MAX_ITERATIONS,
            )
            async for message in runner:
                if message.stop_reason == "refusal":
                    raise RuntimeError("the model declined this request")
        if run.final_payload is None:
            log.warning("run %s: agent never submitted — synthesizing a fallback", run.id)
            await _fallback_submit()
        if run.final_payload is None:
            raise RuntimeError(
                "agent finished without submitting a recommendation, and there wasn't "
                "enough tool state (inspection/match/results) to synthesize a fallback"
            )
        run.status = "done"
    except InspectError as e:
        run.status = "error"
        emit("error", {"message": f"Could not read the archive: {e}"})
    except (TimeoutError, asyncio.TimeoutError):
        run.status = "error"
        emit("error", {"message": "analysis timed out — please retry"})
    except anthropic.APIStatusError as e:
        run.status = "error"
        log.exception("API error")
        if e.status_code == 529:
            emit("error", {"message": "the Claude API is overloaded right now (529) and "
                                      "retries didn't clear it — please try again shortly"})
        else:
            # Surface the API's own message when it has one — 400s are often
            # actionable (billing, invalid model) and "please retry" would lie.
            detail = ""
            try:
                detail = e.body["error"]["message"]
            except (TypeError, KeyError):
                pass
            emit("error", {"message": f"Claude API error ({e.status_code})"
                                      + (f": {detail}" if detail else " — please retry")})
    except anthropic.APIConnectionError:
        run.status = "error"
        emit("error", {"message": "could not reach the Claude API — check network/key"})
    except Exception as e:  # noqa: BLE001 — surface anything else as a run error
        run.status = "error"
        log.exception("agent run failed")
        emit("error", {"message": run_error_message(e, run.id)})
    finally:
        log.info("run %s: %s after %.1fs", run.id, run.status, time.time() - started)
        emit("done", {})
