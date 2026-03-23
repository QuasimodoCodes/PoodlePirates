# Astar Island Norse World Prediction — Full System Review Request

## What I Need From You

I'm competing in NM i AI 2026 "Astar Island" (team "Poodle Pirates"). I need you to:

1. **Analyze my current approach** — identify weaknesses, blind spots, or suboptimal choices
2. **Search for similar problems and solutions** — this is essentially a probabilistic terrain prediction problem with limited queries (multi-armed bandit / active learning / Bayesian inference under budget constraints)
3. **Come back with a prioritized list of recommendations** — what would give the biggest score improvement for the least risk

## The Problem

Observe a black-box Norse civilisation simulator through a limited viewport and predict the final world state as probability distributions.

- **Map:** 40×40 grid, 5 seeds per round
- **Budget:** 50 queries total (shared across all 5 seeds = ~10 per seed)
- **Viewport:** 15×15 max per query
- **Goal:** Predict H×W×6 probability tensor (6 terrain classes) for each seed
- **Ground truth:** Monte Carlo simulation — GT is itself a probability distribution, not deterministic
- **Each query returns a single stochastic sample** from the GT distribution

### Terrain Types
- **Static (free from initial state):** Ocean (code 10) → class 0, Mountain (code 5) → class 5
- **Dynamic (must predict):** Empty/Plains → class 0, Settlement → class 1, Port → class 2, Ruin → class 3, Forest → class 4

### Scoring
```
KL(p || q) = Σᵢ pᵢ × log(pᵢ / qᵢ)          # p = ground truth, q = our prediction
entropy(cell) = -Σᵢ pᵢ × log(pᵢ)             # cell weight
weighted_kl = Σ_cells [entropy(cell) × KL(cell)] / Σ_cells entropy(cell)
score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

- Static cells have ~zero entropy → excluded from scoring
- High-entropy cells dominate the score
- KL divergence is asymmetric: predicting too low on a GT class is MUCH worse than predicting too high
- **Leaderboard = best single weighted round score** (later rounds have higher weight ~2.0+)

### Score Weight Breakdown (from our diagnostics)
- Plains contribute ~66% of score weight
- Forest contribute ~27.5%
- Settlement contribute ~6.1%
- Port contribute ~0.3%

## Our Current Approach

### Architecture
1. **Initial analysis (free):** Classify cells as static/dynamic, identify settlements
2. **Phase 1 queries (25):** 5 tiles per seed targeting settlement clusters (greedy coverage)
3. **Phase 2 queries (25):** 5 tiles per seed at fixed spread positions (corners + centre)
4. **Calibration:** Blend round observations with historical conditional transition matrix
5. **Prediction:** Context-aware prior + Bayesian update + temperature scaling + floors

### Key Parameters (all tuned via leave-one-round-out Monte Carlo cross-validation)
- **α = 0.05** (Bayesian update weight for observed cells)
- **Temperature = 1.10** (softens predictions in log-space)
- **N_HIST = 50** (virtual historical samples for blending — higher = more conservative)
- **N_HIST_SURPRISED = 5** (for buckets where round diverges from historical)
- **SURPRISE_THRESHOLD = 0.30** (symmetric KL to trigger surprise)
- **Per-terrain probability floors:** Settlement=0.008, Port=0.008, Ruin=0.006, Forest=0.003, Plains=0.004

### Context Buckets (19 total)
Each dynamic cell is classified by: `(terrain_code, settlement_density, ocean_proximity)`
- terrain_code: 0, 1, 2, 3, 4, 11
- settlement_density (r=2 neighborhood): sett_hi (3+), sett_lo (1-2), sett_no (0)
- ocean_proximity (r=2): ocean (1+), inland (0)

### What We've Tested and Rejected
- **Wider context radius (r=3,4,5):** All worse — dilutes local signal
- **Finer context buckets:** Worse — not enough samples per bucket
- **Higher alpha:** Always worse — observations are stochastic single samples
- **Spatial smoothing:** Neutral — propagating noisy observations adds noise
- **Different query strategies (40+10, 45+5, all spread, context-diverse):** Current 5-settlement + 5-spread is optimal
- **Adaptive temperature on extreme rounds:** Zero-sum — helps extreme but hurts normal
- **Power recalibration (gamma):** +0.1 pts at best — within noise

## Our Results

| Round | Score | Rank | Type | Notes |
|-------|-------|------|------|-------|
| R4 | 59.0 | #60/86 | ? | Early, bad model |
| R5 | 43.9 | #103/144 | ? | Early |
| R6 | 12.9 | #163/186 | ? | Early, very bad |
| R7 | 46.8 | #131/199 | ? | Early |
| R8 | 89.9 | #22/214 | Normal | Best raw score |
| R9 | 85.9 | #70/221 | Normal | |
| R10 | 72.9 | #95/238 | ? | |
| R11 | 73.5 | #81/171 | ? | |
| R12 | 33.0 | #114/146 | Extreme (Port) | Catastrophic |
| R13 | 86.7 | #59/186 | Normal | Post-improvements |
| R14 | 59.7 | #133/244 | Extreme (Sett) | |
| R15 | 86.8 | #87/262 | Normal | |
| R16 | 79.0 | #115/272 | Quiet | |
| R17 | ? | ? | Moderate shift | Submitted, waiting |

**Current leaderboard: #87, score 180.5 (= best weighted round)**

### Key Findings
1. **~43% of rounds are "extreme"** — simulator parameters shift wildly (Settlement prob doubles, Port explodes, etc.)
2. **Perfect calibration ceiling on extreme rounds is ~78 pts** (vs ~91 on normal)
3. **Our calibration works well on big buckets** (Plains, Forest) but under-calibrates Settlement cells
4. **Query coverage doesn't matter much** — 84% vs 100% coverage = same score because α=0.05
5. **Learning curve is flat** — 3 training rounds scores same as 11. No overfit/underfit.
6. **Calibration underpredicts in high-confidence bins** — when we predict 70%, GT is ~78%

## The Code

### config.py
```python
import os

API_BASE_URL = "https://api.ainm.no/astar-island"
API_TOKEN = os.getenv("ASTAR_API_TOKEN", "")

TOTAL_QUERIES = 50
NUM_SEEDS = 5
QUERIES_PER_SEED = TOTAL_QUERIES // NUM_SEEDS
MAP_WIDTH = 40
MAP_HEIGHT = 40
MAP_TOTAL_CELLS = MAP_WIDTH * MAP_HEIGHT
VIEWPORT_MAX_WIDTH = 15
VIEWPORT_MAX_HEIGHT = 15
NUM_TERRAIN_CLASSES = 6
SIMULATION_YEARS = 50

TERRAIN_CLASSES = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain"}
TILE_ANCHORS = [0, 15, 25]
TILES_TO_COVER_MAP = 9
SPARE_QUERIES = TOTAL_QUERIES - TILES_TO_COVER_MAP * NUM_SEEDS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OBSERVATIONS_DIR = os.path.join(DATA_DIR, "observations")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")

PROB_FLOOR_DYNAMIC = 0.005
PROB_FLOOR_STATIC = 1e-5
UNIFORM_PRIOR = [1.0 / NUM_TERRAIN_CLASSES] * NUM_TERRAIN_CLASSES
SMOOTHING_EPSILON = 1e-9

STATIC_TERRAIN_CODES = {10, 5}
DYNAMIC_TERRAIN_CODES = {0, 1, 2, 3, 4, 11}
```

### terrain_estimator.py
```python
"""
terrain_estimator.py — Build per-cell H×W×6 probability distributions.

Three stacked layers:
  Layer A (static):     Mountains & Ocean → predict with near-certainty from initial state
  Layer C (transition): Prior from historical conditional transition matrix
  Layer B (Bayesian):   posterior = (1-α)*prior + α*one_hot(observed)

α = 0.05 keeps predictions close to the calibrated prior.
Temperature scaling softens predictions to hedge against round-specific variance.
"""

import os
import json
import math
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import config
from src.model.initial_analyzer import SeedMap, CellInfo, CODE_TO_CLASS, STATIC_CODES

N = config.NUM_TERRAIN_CLASSES          # 6

ALPHA = 0.05                            # Bayesian update weight
TEMPERATURE = 1.10                      # Prediction softening: >1 softens, <1 sharpens
FLOOR_STATIC = config.PROB_FLOOR_STATIC # 1e-5
TERRAIN_FLOOR = {                       # Per-terrain-code probability floor
    1:  0.008,   # Settlement
    2:  0.008,   # Port
    3:  0.006,   # Ruin
    4:  0.003,   # Forest
    11: 0.004,   # Plains
    0:  0.005,   # Empty
}
FLOOR_DYNAMIC = config.PROB_FLOOR_DYNAMIC  # 0.005


# ─── Context classification ───────────────────────────────────────────

def cell_context(cells, y: int, x: int, H: int, W: int) -> Tuple:
    code = cells[y][x].initial_code
    sett_count = 0
    ocean_count = 0
    r = 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                nc = cells[ny][nx].initial_code
                if nc == 1:
                    sett_count += 1
                elif nc == 10:
                    ocean_count += 1
    sett_bin = "sett_hi" if sett_count >= 3 else ("sett_lo" if sett_count >= 1 else "sett_no")
    ocean_bin = "ocean" if ocean_count >= 1 else "inland"
    return (code, sett_bin, ocean_bin)


def cell_context_from_grid(igrid, y: int, x: int) -> Tuple:
    H = len(igrid)
    W = len(igrid[0]) if H > 0 else 0
    code = igrid[y][x]
    sett_count = 0
    ocean_count = 0
    r = 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                nc = igrid[ny][nx]
                if nc == 1:
                    sett_count += 1
                elif nc == 10:
                    ocean_count += 1
    sett_bin = "sett_hi" if sett_count >= 3 else ("sett_lo" if sett_count >= 1 else "sett_no")
    ocean_bin = "ocean" if ocean_count >= 1 else "inland"
    return (code, sett_bin, ocean_bin)


# ─── Temperature scaling ─────────────────────────────────────────────

def _apply_temperature(dist: List[float], temp: float) -> List[float]:
    log_dist = [math.log(max(p, 1e-12)) / temp for p in dist]
    max_log = max(log_dist)
    exp_dist = [math.exp(v - max_log) for v in log_dist]
    total = sum(exp_dist)
    return [v / total for v in exp_dist]


# ─── Matrix loading ──────────────────────────────────────────────────

def load_transition_matrix(calibrated: bool = True) -> Optional[Dict]:
    if calibrated:
        cal_path = os.path.join(config.DATA_DIR, "round_calibrated_matrix.json")
        if os.path.exists(cal_path):
            print("  Using round-calibrated transition matrix.")
            with open(cal_path) as f:
                return json.load(f)
    hist_path = os.path.join(config.DATA_DIR, "transition_matrix.json")
    if os.path.exists(hist_path):
        print("  Using historical transition matrix (no calibrated matrix found).")
        with open(hist_path) as f:
            return json.load(f)
    return None


def load_conditional_matrix() -> Dict[Tuple, List[float]]:
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    if not os.path.exists(history_dir):
        return {}

    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    if not files:
        return {}

    cond_acc: Dict[Tuple, List[List[float]]] = defaultdict(list)
    for fname in files:
        with open(os.path.join(history_dir, fname)) as fh:
            data = json.load(fh)
        gt = data.get("ground_truth")
        igrid = data.get("initial_grid")
        if not gt or not igrid:
            continue
        H = len(igrid)
        W = len(igrid[0]) if H > 0 else 0
        for y in range(H):
            for x in range(W):
                code = igrid[y][x]
                if code in STATIC_CODES:
                    continue
                ctx = cell_context_from_grid(igrid, y, x)
                cond_acc[ctx].append(gt[y][x])

    cond_matrix = {}
    for ctx, samples in cond_acc.items():
        n = len(samples)
        avg = [sum(s[i] for s in samples) / n for i in range(N)]
        cond_matrix[ctx] = avg

    print(f"  Conditional matrix loaded: {len(cond_matrix)} context buckets from {len(files)} files.")
    return cond_matrix


# ─── Observation index ────────────────────────────────────────────────

def build_observation_index(observations: list) -> Dict[Tuple[int, int, int], List[int]]:
    obs_index: Dict[Tuple[int, int, int], List[int]] = {}
    for obs in observations:
        query = obs["query"]
        seed = query["seed_index"]
        x0, y0 = query["x"], query["y"]
        grid = obs["response"]["grid"]
        for dy, row in enumerate(grid):
            for dx, code in enumerate(row):
                key = (seed, y0 + dy, x0 + dx)
                if key not in obs_index:
                    obs_index[key] = []
                obs_index[key].append(code)
    return obs_index


# ─── Core estimation ─────────────────────────────────────────────────

CODE_TO_PRED_CLASS = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
    10: 0,   # Ocean → Empty
    11: 0,   # Plains → Empty
}


def estimate(seed_map: SeedMap,
             transition_matrix: Optional[Dict],
             obs_index: Dict[Tuple[int, int, int], List[int]],
             alpha: float = ALPHA,
             conditional_matrix: Optional[Dict[Tuple, List[float]]] = None,
             ) -> List[List[List[float]]]:
    H, W = config.MAP_HEIGHT, config.MAP_WIDTH
    seed = seed_map.seed_index
    tensor = []

    for y in range(H):
        row = []
        for x in range(W):
            cell = seed_map.cells[y][x]
            code = cell.initial_code

            # Layer A: static cells
            if cell.is_static:
                pred_class = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC] * N
                dist[pred_class] = 1.0 - (N - 1) * FLOOR_STATIC
                row.append(dist)
                continue

            # Layer C: conditional prior (with fallback to flat)
            ctx = cell_context(seed_map.cells, y, x, H, W)
            if conditional_matrix and ctx in conditional_matrix:
                prior = conditional_matrix[ctx][:]
            elif transition_matrix and str(code) in transition_matrix:
                prior = transition_matrix[str(code)][:]
            else:
                prior = [1.0 / N] * N

            # Layer B: Bayesian update from observations
            key = (seed, y, x)
            if key in obs_index:
                codes = obs_index[key]
                obs_hist = [0.0] * N
                for c in codes:
                    pc = CODE_TO_PRED_CLASS.get(c, 0)
                    obs_hist[pc] += 1.0 / len(codes)
                dist = [(1 - alpha) * prior[i] + alpha * obs_hist[i]
                        for i in range(N)]
            else:
                dist = prior[:]

            # Temperature scaling
            dist = _apply_temperature(dist, TEMPERATURE)

            # Floor
            floor = TERRAIN_FLOOR.get(code, FLOOR_DYNAMIC)
            dist = [max(v, floor) for v in dist]

            # Renormalize
            total = sum(dist)
            dist = [v / total for v in dist]

            row.append(dist)
        tensor.append(row)

    return tensor


def estimate_all_seeds(seed_maps: List[SeedMap],
                       observations: list = None,
                       ) -> List[List[List[List[float]]]]:
    transition_matrix = load_transition_matrix(calibrated=True)
    conditional_matrix = load_conditional_matrix()

    obs_index = {}
    if observations:
        obs_index = build_observation_index(observations)
        n_observed = len(obs_index)
        total_per_seed = config.MAP_TOTAL_CELLS
        print(f"\n  Transition matrix loaded.")
        print(f"  Observations indexed: {n_observed} cells observed "
              f"({n_observed / total_per_seed * 100:.1f}% of {total_per_seed} total per seed)")
        print(f"  α = {ALPHA}  |  temp = {TEMPERATURE}  |  "
              f"floor_dynamic = {FLOOR_DYNAMIC}  |  floor_static = {FLOOR_STATIC}")

    tensors = []
    for sm in seed_maps:
        tensor = estimate(sm, transition_matrix, obs_index,
                          conditional_matrix=conditional_matrix)
        n_obs = sum(1 for y in range(config.MAP_HEIGHT)
                    for x in range(config.MAP_WIDTH)
                    if (sm.seed_index, y, x) in obs_index
                    and not sm.cells[y][x].is_static)
        n_prior = sm.n_dynamic - n_obs
        print(f"  Seed {sm.seed_index}: tensor built — "
              f"{n_obs} cells from observations, {n_prior} from transition prior")
        tensors.append(tensor)

    return tensors
```

### round_calibrator.py
```python
"""
round_calibrator.py — Build round-specific transition matrix from this round's observations.

For each dynamic terrain code observed this round, compute the empirical
class-frequency and blend with the historical transition matrix.

Blend formula:
    blended[key] = (n_round * round_freq + N_HIST * historical) / (n_round + N_HIST)

N_HIST controls conservatism: higher values trust the historical prior more.
"""

import os
import json
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import config
from src.model.initial_analyzer import STATIC_CODES, CODE_TO_CLASS
from src.model.terrain_estimator import cell_context_from_grid

N = config.NUM_TERRAIN_CLASSES   # 6

N_HIST = 50
N_HIST_SURPRISED = 5              # N_HIST for surprised context buckets
SURPRISE_THRESHOLD = 0.30         # symmetric KL threshold

CODE_TO_PRED_CLASS = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
    10: 0, 11: 0,
}

CODE_NAMES = {
    0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin",
    4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains",
}
CLASS_NAMES = ["Empty", "Settl", "Port", "Ruin", "Forest", "Mtn"]


def _build_obs_index(initial_states: dict, observations: list):
    obs_index = {}
    for obs in observations:
        q = obs["query"]
        seed = q["seed_index"]
        x0, y0 = q["x"], q["y"]
        grid = obs["response"]["grid"]
        for dy, row in enumerate(grid):
            for dx, code in enumerate(row):
                obs_index[(seed, y0 + dy, x0 + dx)] = code
    return obs_index


def _compute_surprise(obs_list: List[int], hist: List[float]) -> float:
    nr = len(obs_list)
    if nr < 3:
        return 0.0
    rf = [0.0] * N
    for v in obs_list:
        rf[v] += 1.0 / nr
    kl_fwd = sum(rf[i] * math.log(max(rf[i], 1e-12) / max(hist[i], 1e-12))
                 for i in range(N) if rf[i] > 1e-12)
    kl_rev = sum(hist[i] * math.log(max(hist[i], 1e-12) / max(rf[i], 1e-12))
                 for i in range(N) if hist[i] > 1e-12)
    return (kl_fwd + kl_rev) / 2


def _blend(round_counts: Dict[int, List[int]],
           historical: Optional[Dict],
           n_hist: int = N_HIST) -> Dict[str, List[float]]:
    blended = {}

    all_keys = set(round_counts.keys())
    if historical:
        all_keys |= {int(k) for k in historical.keys()}

    for code in all_keys:
        hist = [1.0 / N] * N
        if historical and str(code) in historical:
            hist = historical[str(code)]

        counts = round_counts.get(code, [])
        n_round = len(counts)

        if n_round == 0:
            blended[str(code)] = hist[:]
            continue

        freq = [0.0] * N
        for c in counts:
            pc = CODE_TO_PRED_CLASS.get(c, 0)
            freq[pc] += 1.0 / n_round

        total = n_round + n_hist
        blended[str(code)] = [
            (n_round * freq[i] + n_hist * hist[i]) / total
            for i in range(N)
        ]

    return blended


def _blend_conditional(round_counts_ctx: Dict[Tuple, List[int]],
                       conditional_matrix: Dict[Tuple, List[float]],
                       n_hist: int = N_HIST,
                       n_hist_surprised: int = N_HIST_SURPRISED,
                       surprise_threshold: float = SURPRISE_THRESHOLD,
                       ) -> Tuple[Dict[Tuple, List[float]], int]:
    blended = {}
    n_surprised = 0

    all_keys = set(round_counts_ctx.keys()) | set(conditional_matrix.keys())

    for ctx in all_keys:
        hist = conditional_matrix.get(ctx, [1.0 / N] * N)
        obs_list = round_counts_ctx.get(ctx, [])
        n_round = len(obs_list)

        if n_round == 0:
            blended[ctx] = hist[:]
            continue

        surprise = _compute_surprise(obs_list, hist)
        if surprise > surprise_threshold and n_round >= 5:
            nh = n_hist_surprised
            n_surprised += 1
        else:
            nh = n_hist

        freq = [0.0] * N
        for c in obs_list:
            freq[c] += 1.0 / n_round

        total = n_round + nh
        blended[ctx] = [
            (n_round * freq[i] + nh * hist[i]) / total
            for i in range(N)
        ]

    return blended, n_surprised


def calibrate(initial_states: dict,
              observations: list,
              historical_matrix: Optional[Dict],
              verbose: bool = True,
              conditional_matrix: Optional[Dict[Tuple, List[float]]] = None,
              ) -> Dict[str, List[float]]:
    obs_index = _build_obs_index(initial_states, observations)

    round_counts: Dict[int, List[int]] = defaultdict(list)
    round_counts_ctx: Dict[Tuple, List[int]] = defaultdict(list)

    seeds = initial_states.get("initial_states", initial_states.get("seeds", []))
    for seed_idx, seed_data in enumerate(seeds):
        igrid = seed_data["grid"]
        for (s, y, x), obs_code in obs_index.items():
            if s != seed_idx:
                continue
            init_code = igrid[y][x]
            if init_code in STATIC_CODES:
                continue
            pred_class = CODE_TO_PRED_CLASS.get(obs_code, 0)
            round_counts[init_code].append(obs_code)

            ctx = cell_context_from_grid(igrid, y, x)
            round_counts_ctx[ctx].append(pred_class)

    blended = _blend(round_counts, historical_matrix)

    if conditional_matrix is not None:
        blended_cond, n_surprised = _blend_conditional(
            round_counts_ctx, conditional_matrix)
        conditional_matrix.clear()
        conditional_matrix.update(blended_cond)
        print(f"  Conditional matrix calibrated: {len(blended_cond)} context buckets "
              f"updated ({n_surprised} surprised, using N_HIST={N_HIST_SURPRISED}).")

    if verbose:
        _print_report(round_counts, blended, historical_matrix)

    return blended


def _print_report(round_counts, blended, historical):
    print(f"\n  Round calibration — historical vs this round:")
    print(f"  (N_HIST={N_HIST} virtual samples — higher = more conservative)\n")

    for code in sorted(round_counts.keys()):
        name = CODE_NAMES.get(code, f"code {code}")
        n = len(round_counts[code])
        w = n / (n + N_HIST) * 100

        hist = [1.0 / N] * N
        if historical and str(code) in historical:
            hist = historical[str(code)]
        bl = blended[str(code)]

        print(f"  {name} (code {code})  n={n}  round_weight={w:.1f}%")
        for i in range(N):
            if hist[i] < 0.002 and bl[i] < 0.002:
                continue
            delta = bl[i] - hist[i]
            bar_h = "█" * int(hist[i] * 20)
            bar_b = "█" * int(bl[i] * 20)
            flag = " ⚠" if abs(delta) > 0.05 else ""
            print(f"    {CLASS_NAMES[i]:<8} hist={hist[i]:.3f} {bar_h:<20}  "
                  f"blend={bl[i]:.3f} {bar_b:<20}  Δ={delta:+.3f}{flag}")
        print()

    large_deltas = any(
        abs(blended[str(c)][i] - (historical[str(c)][i] if historical and str(c) in historical else 1/N))
        > 0.05
        for c in round_counts
        for i in range(N)
        if str(c) in blended
    )
    if large_deltas:
        print("  ⚠  Large deltas detected — this round may have unusual parameters.")
        print("     The blended matrix will adapt the prior accordingly.")


def save_calibrated_matrix(blended: Dict[str, List[float]]) -> None:
    path = os.path.join(config.DATA_DIR, "round_calibrated_matrix.json")
    with open(path, "w") as f:
        json.dump(blended, f, indent=2)
    print(f"  Calibrated matrix saved to {path}")
```

### adaptive_planner.py
```python
"""
adaptive_planner.py — Two-phase query planning.

Phase 1 (25 queries): 5 settlement-cluster tiles x 5 seeds.
Phase 2 (25 queries): 5 spatially-spread tiles x 5 seeds (corners + centre).
"""

import math
from typing import List, Dict, Tuple, Set

import config
from src.observation.query_planner import Query

TILE_W = config.VIEWPORT_MAX_WIDTH    # 15
TILE_H = config.VIEWPORT_MAX_HEIGHT   # 15

SPREAD_ANCHORS = [(0, 0), (25, 0), (0, 25), (25, 25), (12, 12)]

PHASE1_QUERIES = 25
PHASE2_QUERIES = 25


def _get_all_tile_anchors():
    return [
        (ax, ay)
        for ay in range(config.MAP_HEIGHT - TILE_H + 1)
        for ax in range(config.MAP_WIDTH - TILE_W + 1)
    ]


def select_settlement_cluster_tiles(seed_map, n_tiles: int = 5) -> List[Tuple[int, int]]:
    H, W = config.MAP_HEIGHT, config.MAP_WIDTH
    settlements = set()
    for y in range(H):
        for x in range(W):
            if seed_map.cells[y][x].initial_code == 1:
                settlements.add((y, x))
    if not settlements:
        return SPREAD_ANCHORS[:n_tiles]

    all_anchors = _get_all_tile_anchors()
    anchor_cells = {a: set(_covered_cells(*a)) for a in all_anchors}
    covered_setts: Set[Tuple[int, int]] = set()
    selected = []
    for _ in range(n_tiles):
        best, best_count = None, -1
        for a in all_anchors:
            count = len((anchor_cells[a] & settlements) - covered_setts)
            if count > best_count:
                best_count, best = count, a
        if best is None or best_count <= 0:
            break
        selected.append(best)
        covered_setts |= (anchor_cells[best] & settlements)

    while len(selected) < n_tiles and len(selected) < len(SPREAD_ANCHORS):
        a = SPREAD_ANCHORS[len(selected)]
        if a not in selected:
            selected.append(a)
    return selected[:n_tiles]


def build_phase1_queries(seed_maps) -> List[Query]:
    per_seed_tiles = {}
    for sm in seed_maps:
        tiles = select_settlement_cluster_tiles(sm, n_tiles=5)
        per_seed_tiles[sm.seed_index] = tiles
        anchors_str = "  ".join(f"({a[0]},{a[1]})" for a in tiles)
        print(f"  Seed {sm.seed_index} settlement tiles: {anchors_str}")

    queries = []
    for tile_idx in range(5):
        for sm in seed_maps:
            tiles = per_seed_tiles[sm.seed_index]
            if tile_idx < len(tiles):
                ax, ay = tiles[tile_idx]
                queries.append(Query(
                    seed_index=sm.seed_index, x=ax, y=ay,
                    w=TILE_W, h=TILE_H, phase="phase1",
                    tile_id=f"s{sm.seed_index}_p1t{tile_idx}",
                ))
    return queries


def _entropy(dist: List[float]) -> float:
    return -sum(p * math.log(p) for p in dist if p > 1e-12)


def _covered_cells(ax: int, ay: int) -> List[Tuple[int, int]]:
    return [
        (ay + dy, ax + dx)
        for dy in range(TILE_H)
        for dx in range(TILE_W)
        if ay + dy < config.MAP_HEIGHT and ax + dx < config.MAP_WIDTH
    ]


def build_phase2_spread_queries(seed_maps) -> List[Query]:
    queries = []
    for tile_idx, (ax, ay) in enumerate(SPREAD_ANCHORS):
        for sm in seed_maps:
            queries.append(Query(
                seed_index=sm.seed_index, x=ax, y=ay,
                w=TILE_W, h=TILE_H, phase="phase2",
                tile_id=f"s{sm.seed_index}_p2t{tile_idx}",
            ))
    return queries
```

## What I Want You To Look For

1. **Similar competitions/papers** — Kaggle competitions, academic papers on:
   - Probabilistic prediction under limited observation budget
   - Active learning with KL divergence scoring
   - Bayesian inference for grid worlds / cellular automata
   - Monte Carlo estimation with budget constraints
   - Any "predict probability distribution from samples" competition

2. **Architecture improvements** — Are there fundamentally better approaches I'm missing?
   - Should I use a different prior structure?
   - Is my Bayesian update optimal for KL scoring?
   - Are there better ways to handle the extreme round problem?
   - Should I be using a different model entirely (e.g., Gaussian processes, neural networks)?

3. **Parameter optimization** — Given the scoring formula, are my parameters theoretically optimal?
   - Is α=0.05 correct given that observations are single samples from distributions?
   - Is temperature=1.10 the right amount of softening?
   - Are my probability floors optimal for KL divergence?

4. **Scoring function exploitation** — The KL divergence scoring has specific properties:
   - Entropy-weighted means high-entropy cells matter most
   - KL is asymmetric: q_i too small is infinitely worse than q_i too large
   - Are there mathematical tricks to minimize expected KL given our uncertainty?

5. **What are we leaving on the table?** — Given our constraints (50 queries, 15×15 viewport), what's the theoretical best score and how close are we?

Come back with a **prioritized list of actionable recommendations** sorted by expected impact.
