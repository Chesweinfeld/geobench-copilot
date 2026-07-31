"""Run lifecycle: upload storage, per-run event queues, SSE replay."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import config

_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06")

HEARTBEAT_S = 15
RUN_TTL_S = 60 * 60  # cleanup finished runs older than an hour
# Beyond this a run cannot still be legitimately in flight — the agent caps
# itself at AGENT_TIMEOUT_S — so anything older is stranded and safe to reap
# whatever its status claims.
STALE_RUN_S = config.AGENT_TIMEOUT_S + 30 * 60


@dataclass
class Run:
    id: str
    task: str
    filename: str
    size_bytes: int
    dir: Path
    status: str = "pending"  # pending | running | done | error
    events: list[dict] = field(default_factory=list)  # replay buffer
    # Broadcast signal: set on every emit. Subscribers each keep their own
    # read index into `events`, so any number of SSE clients (reconnects,
    # second tabs) see every event — a shared Queue would hand each event to
    # only one of them.
    new_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Strong reference to the agent asyncio.Task so it can't be GC'd mid-run.
    agent_task: Any = None
    created_at: float = field(default_factory=time.time)
    # caches the agent fills as tools run (used for numeric cross-checks)
    inspection: Any = None
    match_result: Any = None
    query_result: Any = None
    wrapper: Any = None
    final_payload: Any = None
    emitted_steps: set = field(default_factory=set)

    @property
    def archive_path(self) -> Path:
        return self.dir / "archive.zip"

    @property
    def workdir(self) -> Path:
        return self.dir / "extracted"


class RunManager:
    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        # Runs live only in this process's memory, so anything on disk at
        # startup was orphaned by a previous process (crash, restart) and can
        # never be served again — sweep it instead of leaking uploads forever.
        for stale in config.RUNS_DIR.iterdir():
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)

    async def create_run(self, upload, task: str) -> Run:
        run_id = uuid.uuid4().hex[:12]
        run_dir = config.RUNS_DIR / run_id
        run_dir.mkdir(parents=True)
        raw = run_dir / "upload.raw"
        dest = run_dir / "archive.zip"
        size = 0
        with open(raw, "wb") as fh:
            while chunk := await upload.read(1 << 20):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    fh.close()
                    shutil.rmtree(run_dir, ignore_errors=True)
                    raise ValueError(f"upload exceeds {config.MAX_UPLOAD_MB} MB limit")
                fh.write(chunk)

        orig_name = upload.filename or "upload"
        with open(raw, "rb") as fh:
            header = fh.read(4)
        if header in _ZIP_MAGIC:
            # already a zip archive — use as-is
            raw.rename(dest)
        else:
            # single loose file (image, .csv, .geojson, .kml, ...) — wrap it
            # in a one-entry zip so the rest of the pipeline (which only
            # ever reads archive.zip) doesn't need to know the difference.
            ext = Path(orig_name).suffix.lower()
            allowed = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".csv",
                       ".geojson", ".kml", ".json"}
            if ext not in allowed:
                raw.unlink(missing_ok=True)
                shutil.rmtree(run_dir, ignore_errors=True)
                raise ValueError(
                    f"unsupported file type {ext or '(none)'!r} — upload a .zip "
                    "archive, or a single image/.csv/.geojson/.kml file"
                )
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(raw, arcname=Path(orig_name).name)
            raw.unlink()

        run = Run(
            id=run_id,
            task=task,
            filename=orig_name,
            size_bytes=size,
            dir=run_dir,
        )
        run.workdir.mkdir()
        self.runs[run_id] = run
        self._cleanup_old()
        return run

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def emit(self, run: Run, event_type: str, data: dict | None = None) -> None:
        run.events.append({
            "id": len(run.events),
            "type": event_type,
            "data": data or {},
        })
        run.new_event.set()

    async def events(self, run: Run, last_event_id: int | None = None) -> AsyncIterator[str]:
        """SSE generator: replay missed events, then stream live ones.

        Each subscriber reads `run.events` at its own index and waits on the
        shared `new_event` signal, so concurrent subscribers (reconnects,
        multiple tabs) all receive every event.
        """
        idx = (last_event_id + 1) if last_event_id is not None else 0
        while True:
            # clear before draining: an emit during the drain re-sets the
            # flag, so the wait below wakes immediately instead of stalling
            run.new_event.clear()
            while idx < len(run.events):
                event = run.events[idx]
                idx += 1
                yield _sse(event)
                if event["type"] == "done":
                    return
            if run.status in ("done", "error"):
                return
            try:
                await asyncio.wait_for(run.new_event.wait(), timeout=HEARTBEAT_S)
            except asyncio.TimeoutError:
                yield ": ping\n\n"

    def _cleanup_old(self) -> None:
        now = time.time()
        cutoff = now - RUN_TTL_S
        # A run only leaves "running" if the agent reaches its own try/finally.
        # Anything that bypasses that — the worker being OOM-killed, a restart
        # mid-run — strands the run in "running" forever, and the old condition
        # (done/error only) meant its uploaded archive was never reclaimed. On a
        # 512 MB instance a few stranded 50 MB uploads matter. Sweep those too,
        # well past the point where the agent could still be legitimately busy.
        stale_cutoff = now - STALE_RUN_S
        for run_id in list(self.runs):
            run = self.runs[run_id]
            finished = run.created_at < cutoff and run.status in ("done", "error")
            stranded = run.created_at < stale_cutoff
            if finished or stranded:
                shutil.rmtree(run.dir, ignore_errors=True)
                del self.runs[run_id]


def _sse(event: dict) -> str:
    return (
        f"id: {event['id']}\n"
        f"event: {event['type']}\n"
        f"data: {json.dumps(event['data'])}\n\n"
    )
