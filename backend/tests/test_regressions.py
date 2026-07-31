"""Regression tests for the failure modes that reached production.

Every test here corresponds to a bug that shipped, reached a user, and gave no
useful signal when it broke. They share one shape: the failure was silent,
misattributed, or only reproducible somewhere other than a developer laptop.
That is the class this file exists to guard.

Run:  cd backend && uv run --extra dev pytest -q
"""

from __future__ import annotations

import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
REPO = BACKEND.parent

import config  # noqa: E402
from agent import run_error_message  # noqa: E402


# --------------------------------------------------------------------------
# 1. Static file serving must not hand out secrets or source.
#
# FRONTEND_DIR is the repo root, so mounting it with StaticFiles served
# backend/.env (the Anthropic key), .git/, and every source file to anyone who
# could reach the app. Nothing errored and nothing logged.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


@pytest.mark.parametrize("path", [
    "/backend/.env",
    "/backend/agent.py",
    "/backend/config.py",
    "/.git/config",
    "/render.yaml",
    "/HANDOFF.md",
    "/backend/data/registry.json",
])
def test_static_never_serves_secrets_or_source(client, path):
    assert client.get(path).status_code == 404, (
        f"{path} is reachable over HTTP — the static mount is too broad again"
    )


@pytest.mark.parametrize("path", ["/", "/support.js"])
def test_static_still_serves_the_frontend(client, path):
    r = client.get(path)
    assert r.status_code == 200 and r.content, f"{path} should still be served"


def test_react_is_served_locally_not_from_a_cdn(client):
    """The blank-white-screen bug: React was fetched from unpkg.com at runtime,
    and the runtime hides the page before that resolves. On a network that
    blocks the CDN the app was an unexplained white void."""
    vendored = sorted((REPO / "vendor").glob("react*.js"))
    assert vendored, "vendored React builds are missing from vendor/"
    for f in vendored:
        assert client.get(f"/vendor/{f.name}").status_code == 200

    # The invariant is about what the browser *fetches*, so inspect script
    # sources rather than page text — prose mentioning a CDN is harmless.
    head = (REPO / config.FRONTEND_PAGE).read_text(encoding="utf-8").split("</head>")[0]
    srcs = re.findall(r"<script[^>]+src=[\"']([^\"']+)", head, flags=re.I)
    assert srcs, "the page head loads no scripts at all — did the boot chain change?"
    external = [s for s in srcs if re.match(r"https?://|//", s)]
    assert not external, (
        f"the page boots from third-party script(s) {external} — if that host is "
        "blocked or down the page renders as a blank white screen"
    )


# --------------------------------------------------------------------------
# 2. A run failure must never produce an empty message.
#
# The catch-all emitted str(e), which is "" for MemoryError among others. The
# UI then showed a bare "analysis failed" with no detail and no hint that a
# traceback was already in the server log.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    MemoryError(),
    RuntimeError(""),
    ValueError("   "),
    OSError(),
    KeyboardInterrupt(),
    Exception(),
    RuntimeError("a real detail"),
])
def test_run_error_message_is_never_empty(exc):
    msg = run_error_message(exc, "abc123")
    assert msg.strip(), f"{type(exc).__name__} produced an empty message"
    assert "abc123" in msg, "the run id is needed to find the matching traceback"


def test_run_error_message_names_the_type_when_there_is_no_text():
    assert "RuntimeError" in run_error_message(RuntimeError(""), "r1")


def test_run_error_message_keeps_a_real_detail():
    assert "a real detail" in run_error_message(RuntimeError("a real detail"), "r1")


def test_memory_error_is_explained_in_plain_language():
    msg = run_error_message(MemoryError(), "r1").lower()
    assert "memory" in msg


# --------------------------------------------------------------------------
# 3. Ground sample distance must be reported in metres.
#
# rasterio's src.res is in the CRS's own units. Formatting it as "m/px"
# unconditionally reported 10 m/px Sentinel-2 chips in EPSG:4326 as
# "8.98e-05 m/px" — five orders of magnitude off, and resolution is a matching
# facet, so every match against such a dataset was junk.
# --------------------------------------------------------------------------

rasterio = pytest.importorskip("rasterio", reason="resolution needs the geo extra")


def _chip_zip(tmp_path: Path, crs: str, res: float, origin: tuple[float, float]) -> Path:
    from rasterio.transform import from_origin

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir(parents=True), masks.mkdir(parents=True)
    for i in range(4):
        arr = np.random.randint(0, 4000, (3, 64, 64)).astype("uint16")
        kw = dict(driver="GTiff", height=64, width=64, crs=crs,
                  transform=from_origin(origin[0], origin[1], res, res))
        with rasterio.open(images / f"c{i}.tif", "w", count=3, dtype="uint16", **kw) as ds:
            ds.write(arr)
        with rasterio.open(masks / f"c{i}.tif", "w", count=1, dtype="uint8", **kw) as ds:
            ds.write((arr[0] > 2000).astype("uint8"), 1)

    z = tmp_path / "chips.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in list(images.iterdir()) + list(masks.iterdir()):
            zf.write(p, p.relative_to(tmp_path))
    return z


def _resolution_of(zip_path: Path, tmp_path: Path) -> str | None:
    from tools.inspect import inspect_archive

    wd = tmp_path / "wd"
    wd.mkdir(exist_ok=True)
    insp = inspect_archive(zip_path, wd, filename=zip_path.name,
                           size_bytes=zip_path.stat().st_size)
    return insp.model_dump()["resolution"]


def test_geographic_crs_reports_metres_not_degrees(tmp_path):
    # 8.9827e-05 degrees of latitude is ~10 m — Sentinel-2's native GSD.
    z = _chip_zip(tmp_path, "EPSG:4326", 8.9827e-05, (72.88, 19.10))
    res = _resolution_of(z, tmp_path)
    assert res is not None
    metres = float(res.split()[0])
    assert 9.0 < metres < 11.0, f"expected ~10 m/px, got {res!r}"


def test_projected_crs_resolution_is_unchanged(tmp_path):
    z = _chip_zip(tmp_path, "EPSG:32643", 10.0, (300000, 2110000))
    assert _resolution_of(z, tmp_path).startswith("10 ")


def test_sub_metre_projected_resolution_survives_formatting(tmp_path):
    z = _chip_zip(tmp_path, "EPSG:32643", 0.5, (300000, 2110000))
    assert _resolution_of(z, tmp_path).startswith("0.5 ")


# --------------------------------------------------------------------------
# 4. generate_wrapper's length contract must be stated where the model reads it.
#
# It requires one band name per *raw* band. On a dataset with all-zero padding
# bands the agent passed only the informative ones and burned iterations
# recovering from the resulting ValueError.
# --------------------------------------------------------------------------

def test_generate_wrapper_documents_the_band_length_rule():
    import agent as agent_module

    src = Path(agent_module.__file__).read_text(encoding="utf-8")
    start = src.index("def generate_wrapper")
    doc = src[start:start + 1600]
    assert "exactly one entry" in doc.lower() or "one entry per" in doc.lower(), (
        "generate_wrapper's docstring must state the band_names length rule — "
        "the model only sees the docstring"
    )


# --------------------------------------------------------------------------
# 5. Stranded runs must be reclaimed.
#
# A run only leaves "running" if the agent reaches its own try/finally. An
# OOM-kill or restart strands it, and cleanup previously only reaped
# done/error runs — so the uploaded archive leaked forever.
# --------------------------------------------------------------------------

def test_cleanup_reaps_runs_stranded_in_running(tmp_path, monkeypatch):
    import runs as runs_module

    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    mgr = runs_module.RunManager()

    stranded_dir = config.RUNS_DIR / "stranded"
    stranded_dir.mkdir()
    (stranded_dir / "archive.zip").write_bytes(b"x" * 1024)
    stranded = runs_module.Run(id="stranded", task="t", filename="f.zip",
                               size_bytes=1024, dir=stranded_dir)
    stranded.status = "running"
    stranded.created_at = time.time() - runs_module.STALE_RUN_S - 60
    mgr.runs["stranded"] = stranded

    mgr._cleanup_old()

    assert "stranded" not in mgr.runs, "a stranded run was never reclaimed"
    assert not stranded_dir.exists(), "its uploaded archive leaked"


def test_cleanup_leaves_a_genuinely_running_run_alone(tmp_path, monkeypatch):
    import runs as runs_module

    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs2")
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    mgr = runs_module.RunManager()

    live_dir = config.RUNS_DIR / "live"
    live_dir.mkdir()
    live = runs_module.Run(id="live", task="t", filename="f.zip",
                           size_bytes=1, dir=live_dir)
    live.status = "running"
    live.created_at = time.time() - 30
    mgr.runs["live"] = live

    mgr._cleanup_old()

    assert "live" in mgr.runs, "an in-flight run was reaped out from under the agent"


# --------------------------------------------------------------------------
# 6. The polling window must outlast the server's own agent timeout.
#
# The UI polled for 5 minutes after a dropped stream while the server allowed
# the agent 20, so a long run was declared dead while it was still working.
# --------------------------------------------------------------------------

def test_client_poll_window_outlasts_the_agent_timeout():
    page = (REPO / config.FRONTEND_PAGE).read_text(encoding="utf-8")
    line = next(ln for ln in page.splitlines() if "POLL_WINDOW_MS" in ln and "=" in ln)
    expr = line.split("=", 1)[1].strip().rstrip(";")
    window_s = eval(expr, {"__builtins__": {}}) / 1000  # noqa: S307 — literal arithmetic
    assert window_s > config.AGENT_TIMEOUT_S, (
        f"client gives up after {window_s:.0f}s but the agent may run for "
        f"{config.AGENT_TIMEOUT_S}s — it would report a false failure"
    )


# --------------------------------------------------------------------------
# 7. A dropped SSE stream is not a failed analysis.
#
# EventSource fires "error" for transport failures as well as for a
# server-sent `event: error`. Both dispatch to the same listener, so a dropped
# connection surfaced a bare "analysis failed" and pre-empted the reconnect and
# polling recovery — the exact bug a user hit in production.
#
# Behaviour is covered by tests/sse_contract.test.mjs (real EventSource against
# a server that drops mid-stream). This is the cheap static guard.
# --------------------------------------------------------------------------

def test_transport_errors_are_not_treated_as_backend_errors():
    page = (REPO / config.FRONTEND_PAGE).read_text(encoding="utf-8")
    start = page.index('es.addEventListener("error"')
    handler = page[start:start + 900]
    assert "typeof e.data !== \"string\"" in handler, (
        "the SSE error listener no longer distinguishes a transport failure "
        "(no data) from a server-sent error event — a dropped stream will "
        "again read as 'analysis failed'"
    )


# --------------------------------------------------------------------------
# 8. Operational visibility and platform fit.
#
# Production logs showed agent activity followed by "Shutting down" and there
# was no way to tell whether the run had finished — a successful run logged
# nothing at all. The platform health check was also being 401'd by the
# password gate, and long runs make no inbound requests, so a free-tier
# instance could be spun down mid-run.
# --------------------------------------------------------------------------

def test_every_run_logs_its_start_and_outcome():
    import agent as agent_module

    src = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert 'log.info("run %s: start' in src, "runs must announce themselves"
    assert 'log.info("run %s: %s after' in src, (
        "a run must log its outcome in a finally, or a log showing activity "
        "then a shutdown is unreadable after the fact"
    )


def test_health_check_path_is_not_behind_the_password_gate(monkeypatch):
    """The platform's health check is unauthenticated; gating it made the
    service read as failing. /healthz is exempt — and must stay trivial."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "APP_PASSWORD", "a-secret")
    import app as app_module

    with TestClient(app_module.app) as c:
        assert c.get("/healthz").status_code == 200
        # ...and the exemption must not have opened anything else up
        for path in ("/", "/api/health", "/support.js", "/backend/.env"):
            assert c.get(path).status_code == 401, f"{path} escaped the gate"


def test_healthz_leaks_no_configuration():
    from fastapi.testclient import TestClient
    import app as app_module

    with TestClient(app_module.app) as c:
        body = c.get("/healthz").json()
    assert body == {"ok": True}, "liveness must not report config, data or model"


def test_band_count_error_tells_the_model_how_to_fix_it():
    """The model is the caller, so the exception text is the only instruction
    it receives. 'must have exactly 19 entries' was true but not actionable."""
    import tools.wrapper as wrapper_module

    src = Path(wrapper_module.__file__).read_text(encoding="utf-8")
    start = src.index("def generate_wrapper")
    body = src[start:start + 2000]
    assert "unused" in body, "the error must say what to do with padding bands"
    assert "len(band_names)" in body, "the error must report what was received"


def test_client_keeps_the_instance_awake_during_a_run():
    """A long run makes no inbound requests, so a free-tier instance can be
    stopped mid-run — which destroys the in-memory run."""
    page = (REPO / config.FRONTEND_PAGE).read_text(encoding="utf-8")
    assert "_startKeepalive" in page and "/api/health" in page
    # every path that ends a run must stop the timer, or it outlives the run
    assert page.count("this._stopKeepalive()") >= 6, (
        "a run-ending path is missing _stopKeepalive — the timer would leak"
    )


# --------------------------------------------------------------------------
# 9. A damaged archive must fail as an archive problem.
#
# Found by end-to-end testing: zipfile raises BadZipFile, which is not an
# InspectError, so a truncated upload skipped the "Could not read the archive"
# handler and surfaced a bare "File is not a zip file" — no cause, no remedy.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,data", [
    ("garbage", b"PK\x03\x04not-really-a-zip"),
    ("empty", b""),
    ("truncated", None),  # filled in below from a real archive
])
def test_damaged_archives_raise_a_clean_inspect_error(tmp_path, name, data):
    from tools.inspect import InspectError, inspect_archive

    if data is None:
        good = tmp_path / "good.zip"
        with zipfile.ZipFile(good, "w") as zf:
            zf.writestr("images/a.txt", "x" * 4096)
        raw = good.read_bytes()
        data = raw[: len(raw) // 2]

    bad = tmp_path / f"{name}.zip"
    bad.write_bytes(data)
    wd = tmp_path / f"wd_{name}"
    wd.mkdir()

    with pytest.raises(InspectError) as ei:
        inspect_archive(bad, wd, filename=bad.name, size_bytes=bad.stat().st_size)
    assert "zip" in str(ei.value).lower(), "the message should name the real problem"


def test_server_sent_error_events_always_carry_data():
    """The guard above is only sound because every server-sent event has a
    data: line. If _sse ever omits one, real errors would be silently ignored."""
    import runs as runs_module

    frame = runs_module._sse({"id": 1, "type": "error", "data": {"message": "boom"}})
    assert "\ndata: " in frame and "boom" in frame
