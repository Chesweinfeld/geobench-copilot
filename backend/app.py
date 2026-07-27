"""FastAPI app: analyze endpoint, SSE event stream, static frontend serving."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from agent import run_agent
from runs import RunManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("geobench.app")

manager = RunManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [p for p in (config.REGISTRY_JSON, config.MODELS_JSON) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"missing {[str(p) for p in missing]} — run scripts/extract_registry.py "
            "under the torchgeo-bench venv first (see backend README)"
        )
    if not config.TGB_RESULTS_CSV.exists():
        raise RuntimeError(f"results CSV not found at {config.TGB_RESULTS_CSV} — set TGB_RESULTS_CSV")
    # warm the pandas/registry caches
    from tools import matching, results

    n_results = len(results.datasets_with_results())
    n_datasets = len(matching.registry())
    log.info("loaded %d datasets (%d with results), model=%s",
             n_datasets, n_results, config.GEOBENCH_MODEL)
    yield


app = FastAPI(title="GeoBench Copilot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _password_gate(request: Request, call_next):
    """HTTP Basic-Auth gate for hosted deploys. Active only when APP_PASSWORD is
    set (a host secret); browsers show a native login prompt. No-op locally."""
    if config.APP_PASSWORD:
        import base64
        import secrets

        ok = False
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
                # constant-time compare to avoid timing leaks
                ok = secrets.compare_digest(user, config.APP_USERNAME) and \
                    secrets.compare_digest(pw, config.APP_PASSWORD)
            except (ValueError, UnicodeDecodeError):
                ok = False
        if not ok:
            return JSONResponse(
                {"detail": "authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="GeoBench Copilot"'},
            )
    return await call_next(request)


@app.post("/api/analyze", status_code=202)
async def analyze(file: UploadFile, task: str = Form(...)):
    if not task.strip():
        raise HTTPException(400, "task description is empty")
    try:
        run = await manager.create_run(file, task.strip())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    # keep a strong reference on the run — a bare create_task can be GC'd
    run.agent_task = asyncio.create_task(run_agent(run, manager))
    return {"run_id": run.id}


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str, request: Request):
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    last_id = None
    header = request.headers.get("last-event-id")
    if header and header.isdigit():
        last_id = int(header)
    return StreamingResponse(
        manager.events(run, last_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/runs/{run_id}")
async def run_status(run_id: str):
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    body = {"run_id": run.id, "status": run.status, "task": run.task}
    if run.inspection is not None:
        body["inspection"] = run.inspection.model_dump()
    if run.final_payload is not None:
        body["final"] = run.final_payload.model_dump()
    return JSONResponse(body)


def _results_as_of() -> str:
    """Date the backbone-ranking data (all_results.csv) was last updated —
    important on a hosted deploy, where it's a bundled snapshot, so viewers know
    how fresh the numbers are."""
    import datetime

    try:
        ts = config.TGB_RESULTS_CSV.stat().st_mtime
        return datetime.date.fromtimestamp(ts).isoformat()
    except OSError:
        return "unknown"


@app.get("/api/health")
async def health():
    from tools import matching, results

    registry_meta = json.loads(config.REGISTRY_JSON.read_text())
    return {
        "ok": True,
        "model": config.GEOBENCH_MODEL,
        "datasets": len(matching.registry()),
        "datasets_with_results": sorted(results.datasets_with_results()),
        "backbones": len(results.model_paths()),
        "registry_generated_at": registry_meta.get("generated_at"),
        "results_csv": str(config.TGB_RESULTS_CSV),
        "results_as_of": _results_as_of(),
    }


# Small, theme-neutral "results as of <date>" badge injected into the served
# page (bottom-right, low-key). Tells viewers how fresh the ranking data is —
# it's a snapshot on a hosted deploy.
_BADGE = """
<div style="position:fixed;bottom:10px;right:12px;z-index:99999;
  font:11px/1.4 'IBM Plex Mono',ui-monospace,monospace;
  color:rgba(128,128,128,.85);background:rgba(128,128,128,.10);
  padding:3px 8px;border-radius:6px;pointer-events:none;letter-spacing:.02em;">
  benchmark results as of {date}</div>
"""


@app.get("/")
async def index():
    from fastapi.responses import HTMLResponse

    page = config.FRONTEND_DIR / config.FRONTEND_PAGE
    try:
        html = page.read_text(encoding="utf-8")
    except OSError:
        # fall back to the plain static redirect if the page can't be read here
        return RedirectResponse(f"/{urllib.parse.quote(config.FRONTEND_PAGE)}")
    badge = _BADGE.format(date=_results_as_of())
    html = html.replace("</body>", badge + "</body>", 1) if "</body>" in html else html + badge
    return HTMLResponse(html)


# mounted last so /api/* wins
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
