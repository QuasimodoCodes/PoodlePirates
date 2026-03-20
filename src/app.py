"""
FastAPI application — entry point for the competition platform.

The platform sends POST /solve with task details.
We return {"status": "completed"} when done.
"""
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agent.runner import run_agent
from src.config import settings
from src.tripletex.client import TripletexClient

from src.config import settings

log = structlog.get_logger()


# ── Request / Response models ─────────────────────────────────────────────────

class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class SolveRequest(BaseModel):
    """Payload sent by the competition platform to /solve."""
    prompt: str
    files: list[dict] | None = None
    tripletex_credentials: TripletexCredentials


class SolveResponse(BaseModel):
    status: str = "completed"


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("tripletex_agent_started")
    yield
    log.info("tripletex_agent_stopped")


app = FastAPI(title="Tripletex AI Accounting Agent", version="0.1.0", lifespan=lifespan)

# Task counter for correlating with competition results
_task_counter = 0


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _handle_solve(req: SolveRequest):
    """Shared handler for /solve and /solv (typo-safe alias)."""
    global _task_counter
    _task_counter += 1
    task_num = _task_counter
    run_id = str(uuid.uuid4())[:8]
    start = time.time()
    log.info("solve_start", run_id=run_id, task_num=task_num,
             prompt_preview=req.prompt[:120])
    try:
        client = TripletexClient(
            proxy_url=req.tripletex_credentials.base_url,
            session_token=req.tripletex_credentials.session_token,
        )
        await run_agent(client=client, prompt=req.prompt, files=req.files or [], run_id=run_id)
    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        log.error("solve_failed", run_id=run_id, task_num=task_num, elapsed=elapsed, error=str(exc))
    elapsed = round(time.time() - start, 2)
    log.info("solve_done", run_id=run_id, task_num=task_num, elapsed=elapsed)
    return SolveResponse(status="completed")


@app.post("/", response_model=SolveResponse)
async def root_solve(req: SolveRequest):
    return await _handle_solve(req)


@app.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest):
    return await _handle_solve(req)


@app.post("/solv", response_model=SolveResponse)  # typo-safe alias
async def solv(req: SolveRequest):
    return await _handle_solve(req)

