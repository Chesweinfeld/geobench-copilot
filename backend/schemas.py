"""Pydantic models mirroring the frontend data contract.

Field names match the DC component's renderVals() expectations verbatim —
this file is the single source of truth for the SSE `final` payload and for
validating the agent's submit_recommendation call.
"""

from typing import Literal

from pydantic import BaseModel, Field


class BandStat(BaseModel):
    index: int
    name: str | None = None  # canonical guess, e.g. "red" — may be None
    mean: float
    std: float
    min: float
    max: float


class VectorSummary(BaseModel):
    """Supplementary metadata parsed from a GeoJSON/KML file in the archive.

    Informational only — never drives structure/labelType detection, which
    stays keyed off the raster chips (and CSV/mask siblings) as before.
    """
    filename: str
    format: str  # "geojson" | "kml"
    geometryTypes: list[str] = Field(default_factory=list)
    featureCount: int
    attributeFields: list[str] = Field(default_factory=list)
    bbox: list[float] | None = None  # [minx, miny, maxx, maxy] when derivable
    crs: str | None = None  # e.g. "EPSG:4326"; None when unknown/unspecified


class Inspection(BaseModel):
    filename: str
    chips: int
    sizeMB: float
    bands: int
    dtype: str
    imageSize: int | None = None
    resolution: str | None = None  # e.g. "10 m/px" when known
    sensorGuess: str  # "s2" | "aerial" | "sar" | "unknown" — evidence-based guess
    sensorEvidence: str  # one-line why
    labelType: str  # "binary" | "multiclass (N)" | "multilabel" | "masks" | "unlabeled"
    numClasses: int | None = None
    classBalance: dict[str, float] = Field(default_factory=dict)  # class -> fraction
    splits: dict[str, int] = Field(default_factory=dict)  # split -> chip count
    structure: str  # "folders-per-class" | "image-mask-pairs" | "csv-labels" | "flat" | "fetched-satellite"
    bandStats: list[BandStat] = Field(default_factory=list)
    vectorContext: list[VectorSummary] = Field(default_factory=list)
    importedSatellite: bool = False  # true if chips were auto-fetched (no images were uploaded)
    importNote: str | None = None  # one-line summary of what was fetched and from where


class Facet(BaseModel):
    label: Literal["Bands", "Label type", "Task", "Object scale", "Domain"]
    pct: int = Field(ge=0, le=100)


class MatchedDataset(BaseModel):
    name: str
    sim: int = Field(ge=0, le=100)
    desc: str
    facets: list[Facet]


class RankedModel(BaseModel):
    rank: int
    name: str
    top: bool = False
    budget: bool = False
    expected: float
    metric: str
    gmacs: float | None = None
    note: str
    chips: list[str] = Field(default_factory=list)


class LeaderboardBar(BaseModel):
    name: str
    acc: float
    top: bool = False


class ScatterPt(BaseModel):
    name: str
    label: str
    gmacs: float
    acc: float
    top: bool = False


class Wrapper(BaseModel):
    filename: str
    code: str
    note: str


class RunModel(BaseModel):
    id: str
    model: str  # hydra path, e.g. "timm/vit_large_patch16_dinov3sat"
    short: str  # display name (CSV name)
    gmacs: float | None = None
    mins: int
    command: str  # full torchgeo-bench run command for this model


class Recommendation(BaseModel):
    headlineModel: str
    head: str
    expected: float
    metric: str
    prose: str


class FinalPayload(BaseModel):
    recommendation: Recommendation
    matched: list[MatchedDataset] = Field(min_length=1, max_length=3)
    models: list[RankedModel] = Field(min_length=1, max_length=5)
    leaderboard: list[LeaderboardBar] = Field(default_factory=list, max_length=6)
    scatter: list[ScatterPt] = Field(default_factory=list, max_length=6)
    wrapper: Wrapper
    runModels: list[RunModel] = Field(default_factory=list)
    caveats: list[str] = Field(min_length=1)
    glossary: dict[str, str] = Field(default_factory=dict)
