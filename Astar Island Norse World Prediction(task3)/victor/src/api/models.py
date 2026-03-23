"""
models.py — Typed dataclasses for every API request and response.

All field names match the API exactly. If the API changes a field name,
fix it here only — the rest of the codebase stays untouched.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# SHARED / PRIMITIVE TYPES
# ─────────────────────────────────────────────

@dataclass
class Settlement:
    """
    A settlement visible in a simulate() or initial_states response.
    Initial states only expose: x, y, has_port, alive.
    Simulate responses add: population, food, wealth, defense, tech_level,
    longship, owner_id.
    """
    x: int
    y: int
    has_port: bool
    alive: bool
    # Below are only present in simulate() responses (not initial_states)
    population: Optional[float] = None
    food:       Optional[float] = None
    wealth:     Optional[float] = None
    defense:    Optional[float] = None
    tech_level: Optional[float] = None
    longship:   Optional[bool]  = None
    owner_id:   Optional[int]   = None


@dataclass
class ViewportInfo:
    """The actual viewport bounds returned by simulate() — may be clamped."""
    x: int
    y: int
    w: int
    h: int


# ─────────────────────────────────────────────
# ROUND ENDPOINTS
# ─────────────────────────────────────────────

@dataclass
class InitialState:
    """
    The initial map state for one seed.
    grid[y][x] = terrain code (0/1/2/3/4/5/10/11).
    See PROBLEM.md terrain table for code meanings.
    """
    grid: List[List[int]]
    settlements: List[Settlement]


@dataclass
class RoundSummary:
    """Lightweight round info from GET /rounds list."""
    id: str
    status: str          # "active" | "completed" | "pending"
    map_width: int
    map_height: int
    seeds_count: int


@dataclass
class RoundDetail:
    """Full round info from GET /rounds/{round_id}."""
    id: str
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: List[InitialState]   # one per seed, index = seed_index


# ─────────────────────────────────────────────
# BUDGET ENDPOINT
# ─────────────────────────────────────────────

@dataclass
class BudgetResponse:
    """GET /budget — check remaining query budget."""
    round_id: str
    queries_used: int
    queries_max: int      # always 50
    active: bool

    @property
    def queries_remaining(self) -> int:
        return self.queries_max - self.queries_used

    @property
    def is_exhausted(self) -> bool:
        return self.queries_used >= self.queries_max


# ─────────────────────────────────────────────
# SIMULATE ENDPOINT
# ─────────────────────────────────────────────

@dataclass
class SimulateRequest:
    """POST /simulate — run sim and observe a viewport."""
    round_id: str
    seed_index: int          # 0–4
    viewport_x: int
    viewport_y: int
    viewport_w: int          # max 15, min 5
    viewport_h: int          # max 15, min 5


@dataclass
class SimulateResponse:
    """
    POST /simulate response.

    IMPORTANT: width/height = FULL MAP dimensions (always 40×40).
    grid shape = viewport_h × viewport_w (only the observed window).
    Use viewport.x and viewport.y to place cells onto the full 40×40 tensor.
    """
    grid: List[List[int]]           # terrain codes — shape: [viewport_h][viewport_w]
    settlements: List[Settlement]   # settlements WITHIN the viewport
    viewport: ViewportInfo          # actual bounds (may differ if clamped)
    width: int                      # full map width (40) — NOT viewport width
    height: int                     # full map height (40) — NOT viewport height
    queries_used: int
    queries_max: int

    @property
    def queries_remaining(self) -> int:
        return self.queries_max - self.queries_used


# ─────────────────────────────────────────────
# SUBMIT ENDPOINT
# ─────────────────────────────────────────────

@dataclass
class SubmitRequest:
    """
    POST /submit — submit prediction tensor for one seed.
    prediction[y][x] = list of 6 floats summing to 1.0 ± 0.01.
    Shape: map_height × map_width × 6.
    """
    round_id: str
    seed_index: int
    prediction: List[List[List[float]]]   # [y][x][class_index]


@dataclass
class SubmitResponse:
    """POST /submit response."""
    success: bool
    message: str = ""


# ─────────────────────────────────────────────
# ANALYSIS ENDPOINT (post-round only)
# ─────────────────────────────────────────────

@dataclass
class AnalysisResponse:
    """
    GET /analysis/{round_id}/{seed_index} — available after round ends.
    Returns our prediction, the real ground truth, score, and initial grid.
    Use this in post_round_analysis.py to fill learning_log.md.
    """
    prediction:   List[List[List[float]]]   # our submitted prediction [y][x][6]
    ground_truth: List[List[List[float]]]   # Monte Carlo ground truth  [y][x][6]
    score:        float                     # our seed score (0–100)
    initial_grid: List[List[int]]           # initial terrain codes [y][x]
