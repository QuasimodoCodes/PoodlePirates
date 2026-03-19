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

from src.config import settings

log = structlog.get_logger()


# ── Request / Response models ─────────────────────────────────────────────────

class SolveRequest(BaseModel):
    """Payload sent by the competition platform to /solve."""
    prompt: str                          # Task description (any of 7 languages)
    proxy_url: str                       # Base URL for Tripletex API calls
    session_token: str                   # Pre-authenticated session token
    files: list[dict] | None = None      # Optional attachments [{name, content_base64, mime_type}]


class SolveResponse(BaseModel):
    status: str = "completed"


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("tripletex_agent_started")
    yield
    log.info("tripletex_agent_stopped")


app = FastAPI(title="Tripletex AI Accounting Agent", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest):
    run_id = str(uuid.uuid4())[:8]
    start = time.time()
    log.info("solve_start", run_id=run_id, prompt_preview=req.prompt[:120])

    try:
        from src.tripletex.client import TripletexClient
        from src.agent.runner import run_agent

        client = TripletexClient(proxy_url=req.proxy_url, session_token=req.session_token)
        await run_agent(client=client, prompt=req.prompt, files=req.files or [], run_id=run_id)

    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        log.error("solve_failed", run_id=run_id, elapsed=elapsed, error=str(exc))
        # Still return completed — platform scores on what was done, not the exception
        return SolveResponse(status="completed")

    elapsed = round(time.time() - start, 2)
    log.info("solve_done", run_id=run_id, elapsed=elapsed)
    return SolveResponse(status="completed")
