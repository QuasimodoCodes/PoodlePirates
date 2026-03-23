"""
client.py — The ONLY place in the codebase that makes HTTP calls.

Every endpoint is a typed method. Auth, error handling, and rate limiting
all live here. Nothing outside this file calls requests directly.
"""

import time
import requests
from typing import List, Optional

from src.api.models import (
    RoundSummary, RoundDetail, InitialState, Settlement,
    BudgetResponse, SimulateRequest, SimulateResponse,
    ViewportInfo, SubmitRequest, SubmitResponse, AnalysisResponse,
)
import config


class AstarClient:
    """
    Authenticated HTTP client for the Astar Island API.

    Usage:
        client = AstarClient(token="eyJ...")
        budget = client.get_budget(round_id)
        result = client.simulate(round_id, seed_index=0, x=0, y=0, w=15, h=15)
    """

    def __init__(self, token: str):
        if not token:
            raise ValueError(
                "API token is empty. Set ASTAR_API_TOKEN environment variable.\n"
                "Get your token from: app.ainm.no → DevTools → Cookies → access_token"
            )
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {token}"
        self._base = config.API_BASE_URL

    # ─────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        url = f"{self._base}{path}"
        resp = self._session.get(url)
        self._raise_for_status(resp)
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self._base}{path}"
        resp = self._session.post(url, json=body)
        self._raise_for_status(resp)
        return resp.json()

    def _raise_for_status(self, resp: requests.Response) -> None:
        if resp.status_code == 400:
            raise ValueError(f"API 400 Bad Request: {resp.text}")
        elif resp.status_code == 403:
            raise PermissionError("API 403: Not on a team — join a team at app.ainm.no")
        elif resp.status_code == 404:
            raise LookupError(f"API 404 Not Found: {resp.url}")
        elif resp.status_code == 429:
            raise RuntimeError(
                "API 429: Budget exhausted OR rate limit hit. "
                f"Response: {resp.text}"
            )
        elif not resp.ok:
            raise RuntimeError(f"API {resp.status_code}: {resp.text}")

    @staticmethod
    def _parse_settlement(raw: dict) -> Settlement:
        return Settlement(
            x=raw["x"],
            y=raw["y"],
            has_port=raw.get("has_port", False),
            alive=raw.get("alive", True),
            population=raw.get("population"),
            food=raw.get("food"),
            wealth=raw.get("wealth"),
            defense=raw.get("defense"),
            tech_level=raw.get("tech_level"),
            longship=raw.get("longship"),
            owner_id=raw.get("owner_id"),
        )

    # ─────────────────────────────────────────────
    # ROUNDS  (free — no query cost)
    # ─────────────────────────────────────────────

    def get_rounds(self) -> List[RoundSummary]:
        """GET /rounds — list all rounds."""
        data = self._get("/rounds")
        return [
            RoundSummary(
                id=r["id"],
                status=r["status"],
                map_width=r.get("map_width", config.MAP_WIDTH),
                map_height=r.get("map_height", config.MAP_HEIGHT),
                seeds_count=r.get("seeds_count", config.NUM_SEEDS),
            )
            for r in data
        ]

    def get_active_round_id(self) -> str:
        """Returns the round_id of the currently active round. Raises if none."""
        rounds = self.get_rounds()
        active = [r for r in rounds if r.status == "active"]
        if not active:
            raise RuntimeError("No active round found. Check app.ainm.no.")
        return active[0].id

    def get_round_detail(self, round_id: str) -> RoundDetail:
        """
        GET /rounds/{round_id} — full round details including initial_states.
        FREE — does not cost a query. Call this first before any simulate().
        """
        data = self._get(f"/rounds/{round_id}")
        initial_states = []
        for seed_data in data["initial_states"]:
            settlements = [self._parse_settlement(s) for s in seed_data.get("settlements", [])]
            initial_states.append(InitialState(
                grid=seed_data["grid"],
                settlements=settlements,
            ))
        return RoundDetail(
            id=data["id"],
            status=data["status"],
            map_width=data["map_width"],
            map_height=data["map_height"],
            seeds_count=data["seeds_count"],
            initial_states=initial_states,
        )

    # ─────────────────────────────────────────────
    # BUDGET  (free — no query cost)
    # ─────────────────────────────────────────────

    def get_budget(self, round_id: str) -> BudgetResponse:
        """
        GET /budget — check remaining query budget.
        Always call this before starting observations.
        """
        data = self._get("/budget")
        return BudgetResponse(
            round_id=data["round_id"],
            queries_used=data["queries_used"],
            queries_max=data["queries_max"],
            active=data["active"],
        )

    # ─────────────────────────────────────────────
    # SIMULATE  (costs 1 query — be careful)
    # ─────────────────────────────────────────────

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        x: int,
        y: int,
        w: int = config.VIEWPORT_MAX_WIDTH,
        h: int = config.VIEWPORT_MAX_HEIGHT,
    ) -> SimulateResponse:
        """
        POST /simulate — observe a viewport. COSTS 1 QUERY.

        x, y: top-left corner of viewport on the 40×40 map
        w, h: viewport size (max 15×15, min 5×5)

        Response.grid = terrain codes for the viewport only.
        Response.width/height = full map dims (40×40), not viewport.
        Use response.viewport.x/y to place cells on the full map.
        """
        # Clamp w/h to valid range as a safety net
        w = max(5, min(w, config.VIEWPORT_MAX_WIDTH))
        h = max(5, min(h, config.VIEWPORT_MAX_HEIGHT))

        body = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": x,
            "viewport_y": y,
            "viewport_w": w,
            "viewport_h": h,
        }
        data = self._post("/simulate", body)

        settlements = [self._parse_settlement(s) for s in data.get("settlements", [])]
        vp = data["viewport"]

        return SimulateResponse(
            grid=data["grid"],
            settlements=settlements,
            viewport=ViewportInfo(x=vp["x"], y=vp["y"], w=vp["w"], h=vp["h"]),
            width=data["width"],    # full map width (40)
            height=data["height"],  # full map height (40)
            queries_used=data["queries_used"],
            queries_max=data["queries_max"],
        )

    # ─────────────────────────────────────────────
    # SUBMIT  (free — no query cost)
    # ─────────────────────────────────────────────

    def submit(
        self,
        round_id: str,
        seed_index: int,
        prediction: List[List[List[float]]],
    ) -> SubmitResponse:
        """
        POST /submit — submit H×W×6 prediction tensor for one seed.
        Resubmitting overwrites the previous prediction.
        Rate limit: 2 req/sec.

        prediction[y][x] must sum to 1.0 ± 0.01. Apply floor before calling.
        """
        body = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        data = self._post("/submit", body)
        return SubmitResponse(
            success=data.get("success", True),
            message=data.get("message", ""),
        )

    # ─────────────────────────────────────────────
    # ANALYSIS  (free — post-round only)
    # ─────────────────────────────────────────────

    def get_analysis(self, round_id: str, seed_index: int) -> AnalysisResponse:
        """
        GET /analysis/{round_id}/{seed_index}
        Only available AFTER round ends. Returns ground truth + our prediction.
        Use in post_round_analysis.py to fill learning_log.md.
        """
        data = self._get(f"/analysis/{round_id}/{seed_index}")
        return AnalysisResponse(
            prediction=data["prediction"],
            ground_truth=data["ground_truth"],
            score=data["score"],
            initial_grid=data["initial_grid"],
        )

    # ─────────────────────────────────────────────
    # CONVENIENCE
    # ─────────────────────────────────────────────

    def get_my_rounds(self) -> dict:
        """GET /my-rounds — team-specific scores, rank, and budget per round."""
        return self._get("/my-rounds")

    def get_leaderboard(self) -> dict:
        """GET /leaderboard — public standings."""
        return self._get("/leaderboard")
