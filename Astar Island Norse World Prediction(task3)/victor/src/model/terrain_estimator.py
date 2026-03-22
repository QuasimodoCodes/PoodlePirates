"""
terrain_estimator.py — Build per-cell H×W×6 probability distributions.

Three stacked layers (applied in order):

  LAYER A — Static cells (Mountain=5, Ocean=10)
    Predict with near-certainty from initial_states. Free, no queries.
    Floor: 1e-5

  LAYER C — Conditional transition matrix prior (base for ALL dynamic cells)
    Uses context-aware distributions: P(outcome | terrain_code, sett_bin, ocean_bin)
    Built from historical round data. Falls back to flat matrix per terrain code.
    Floor: per-terrain (see TERRAIN_FLOOR)

  LAYER B — Bayesian update when simulate() observations exist
    posterior = (1 - α) × transition_prior  +  α × one_hot(observed_class)
    α = 0.05 (tuned via Monte Carlo simulation)
    Applied ON TOP of Layer C when we have a direct observation.

Order: C → B (if observed) → A (if static) → floor → renormalize
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import config
from src.model.initial_analyzer import SeedMap, CODE_TO_CLASS, STATIC_CODES


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

ALPHA_1OBS = 0.03     # Bayesian weight for single observation (noisy)
ALPHA_MULTI = 0.10    # Bayesian weight for 2+ observations (more reliable)
                      # Stepped alpha tested: +0.18 pts (18-round LOO)
ALPHA = 0.05          # Legacy default (used if called with explicit alpha)
TEMPERATURE = 1.10    # Prediction softening: >1 softens, <1 sharpens
                      # Tested: t=1.10 -> +0.53 pts (11-round LOO)
                      # Hedges against round-specific parameter variance
FLOOR_STATIC = 1e-5   # floor for static cells (Mountain/Ocean)
N_CLASSES = config.NUM_TERRAIN_CLASSES

# Per-terrain floors: data-driven from historical minimums
# Tested: 0.001 uniform = +0.93 pts over previous per-terrain floors (18-round LOO)
# Lower floors steal less probability from the dominant class
TERRAIN_FLOOR = {
    1: 0.001,   # Settlement
    2: 0.001,   # Port
    3: 0.001,   # Ruin
    4: 0.001,   # Forest
    11: 0.001,  # Plains
    0: 0.001,   # Empty
}
FLOOR_DYNAMIC = 0.001  # fallback for any code not in TERRAIN_FLOOR


def _apply_temperature(dist: List[float], temp: float) -> List[float]:
    """Apply temperature scaling: temp>1 softens (spreads probability), temp<1 sharpens."""
    if temp == 1.0:
        return dist
    import math
    log_dist = [math.log(max(p, 1e-12)) / temp for p in dist]
    max_log = max(log_dist)
    exp_dist = [math.exp(v - max_log) for v in log_dist]
    total = sum(exp_dist)
    return [v / total for v in exp_dist]


# ─────────────────────────────────────────────
# CELL CONTEXT (for conditional matrix lookup)
# ─────────────────────────────────────────────

def _min_dist_to_code(cells_or_grid, y: int, x: int, target_code: int, H: int, W: int, is_grid: bool = False) -> int:
    """Manhattan distance to nearest cell with target_code. Returns 99 if none found."""
    best = 99
    for ny in range(H):
        for nx in range(W):
            if is_grid:
                nc = cells_or_grid[ny][nx]
            else:
                nc = cells_or_grid[ny][nx].initial_code
            if nc == target_code:
                d = abs(ny - y) + abs(nx - x)
                if d < best:
                    best = d
    return best


def _bin_dist_ocean(d: int) -> str:
    """Bin ocean distance: 0-1, 2-4, 5-10, 11+. Tested: +1.64 pts (18-round LOO)."""
    if d <= 1:
        return "od0"
    if d <= 4:
        return "od1"
    if d <= 10:
        return "od2"
    return "od3"


# Cache for precomputed ocean distance maps per grid id
_ocean_dist_cache: Dict = {}


def _get_ocean_dist_grid(igrid, grid_id=None) -> List[List[int]]:
    """Compute ocean distance map for an entire grid using BFS (fast)."""
    if grid_id and grid_id in _ocean_dist_cache:
        return _ocean_dist_cache[grid_id]

    H, W = len(igrid), len(igrid[0])
    dist = [[99] * W for _ in range(H)]
    queue = []

    # Seed BFS from all ocean cells
    for y in range(H):
        for x in range(W):
            if igrid[y][x] == 10:
                dist[y][x] = 0
                queue.append((y, x))

    # BFS for Manhattan distance
    head = 0
    while head < len(queue):
        cy, cx = queue[head]
        head += 1
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and dist[ny][nx] > dist[cy][cx] + 1:
                dist[ny][nx] = dist[cy][cx] + 1
                queue.append((ny, nx))

    if grid_id:
        _ocean_dist_cache[grid_id] = dist
    return dist


def cell_context(cells, y: int, x: int, H: int, W: int, ocean_dist_map=None) -> Tuple:
    """
    Classify a cell into a context bucket based on its neighbors.
    Returns: (terrain_code, sett_bin, ocean_bin) for non-Plains
             (terrain_code, sett_bin, ocean_dist_bin) for Plains (code 11)

    Plains cells get a 4-bin ocean distance feature instead of binary ocean flag.
    Tested: +1.64 pts improvement on Plains predictions (18-round LOO).
    """
    code = cells[y][x].initial_code
    sett_n = 0
    ocean_n = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                nc = cells[ny][nx].initial_code
                if nc == 1:
                    sett_n += 1
                elif nc == 10:
                    ocean_n += 1

    if sett_n >= 3:
        sett_bin = "sett_hi"
    elif sett_n >= 1:
        sett_bin = "sett_lo"
    else:
        sett_bin = "sett_no"

    # Plains cells: use distance-to-ocean (4 bins) instead of binary ocean flag
    if code == 11 and ocean_dist_map is not None:
        ocean_bin = _bin_dist_ocean(ocean_dist_map[y][x])
    else:
        ocean_bin = "ocean" if ocean_n >= 1 else "inland"

    return (code, sett_bin, ocean_bin)


def cell_context_from_grid(igrid, y: int, x: int, ocean_dist_map=None) -> Tuple:
    """Same as cell_context but works on raw grid (list of lists of ints)."""
    code = igrid[y][x]
    H, W = len(igrid), len(igrid[0])
    sett_n = 0
    ocean_n = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                nc = igrid[ny][nx]
                if nc == 1:
                    sett_n += 1
                elif nc == 10:
                    ocean_n += 1

    if sett_n >= 3:
        sett_bin = "sett_hi"
    elif sett_n >= 1:
        sett_bin = "sett_lo"
    else:
        sett_bin = "sett_no"

    # Plains cells: use distance-to-ocean (4 bins) instead of binary ocean flag
    if code == 11 and ocean_dist_map is not None:
        ocean_bin = _bin_dist_ocean(ocean_dist_map[y][x])
    else:
        ocean_bin = "ocean" if ocean_n >= 1 else "inland"

    return (code, sett_bin, ocean_bin)


# ─────────────────────────────────────────────
# TRANSITION MATRIX LOADERS
# ─────────────────────────────────────────────

def load_transition_matrix(calibrated: bool = True) -> Dict[int, List[float]]:
    """
    Load flat transition matrix. Used as fallback when conditional matrix
    doesn't have a matching context bucket.
    """
    # Try calibrated matrix first
    if calibrated:
        cal_path = os.path.join(config.DATA_DIR, "round_calibrated_matrix.json")
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                data = json.load(f)
            matrix = {int(k): v for k, v in data["transition_matrix"].items()}
            for code in [0, 1, 2, 3, 4, 5, 10, 11]:
                if code not in matrix:
                    matrix[code] = [1.0 / N_CLASSES] * N_CLASSES
            print("  Using round-calibrated transition matrix.")
            return matrix

    # Fall back to historical matrix
    path = os.path.join(config.DATA_DIR, "transition_matrix.json")
    if not os.path.exists(path):
        print("  ⚠️  transition_matrix.json not found — using uniform prior.")
        uniform = [1.0 / N_CLASSES] * N_CLASSES
        return {c: uniform[:] for c in [0, 1, 2, 3, 4, 5, 10, 11]}

    with open(path) as f:
        data = json.load(f)

    matrix = {}
    for code_str, dist in data["transition_matrix"].items():
        matrix[int(code_str)] = dist

    for code in [0, 1, 2, 3, 4, 5, 10, 11]:
        if code not in matrix:
            matrix[code] = [1.0 / N_CLASSES] * N_CLASSES

    print("  Using historical transition matrix (no calibrated matrix found).")
    return matrix


def load_conditional_matrix() -> Dict[Tuple, List[float]]:
    """
    Load conditional transition matrix from round_history.
    Keys are (terrain_code, sett_bin, ocean_bin) tuples.
    Built from all available historical round data.
    """
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    if not os.path.exists(history_dir):
        print("  ⚠️  No round_history — conditional matrix empty.")
        return {}

    files = [f for f in os.listdir(history_dir) if f.endswith("_analysis.json")]
    if not files:
        print("  ⚠️  No analysis files in round_history — conditional matrix empty.")
        return {}

    accum = defaultdict(list)
    for fname in files:
        with open(os.path.join(history_dir, fname)) as fh:
            data = json.load(fh)
        gt = data.get("ground_truth")
        igrid = data.get("initial_grid")
        if not gt or not igrid:
            continue
        H, W = len(igrid), len(igrid[0])
        odm = _get_ocean_dist_grid(igrid, grid_id=fname)
        for y in range(H):
            for x in range(W):
                code = igrid[y][x]
                if code in STATIC_CODES:
                    continue
                ctx = cell_context_from_grid(igrid, y, x, ocean_dist_map=odm)
                accum[ctx].append(gt[y][x])

    matrix = {}
    for ctx, samples in accum.items():
        n = len(samples)
        matrix[ctx] = [sum(s[i] for s in samples) / n for i in range(N_CLASSES)]

    print(f"  Conditional matrix loaded: {len(matrix)} context buckets from {len(files)} files.")
    return matrix


# ─────────────────────────────────────────────
# OBSERVATION INDEX
# ─────────────────────────────────────────────

def build_observation_index(observations: List[dict]) -> Dict[Tuple[int,int,int], List[int]]:
    """
    Build a lookup: (seed_index, y, x) → list of observed terrain codes.
    Multiple observations of same cell = multiple samples for Bayesian update.
    """
    index: Dict[Tuple[int,int,int], List[int]] = {}

    for obs in observations:
        q = obs["query"]
        r = obs["response"]
        seed = q["seed_index"]
        vp = r["viewport"]
        grid = r["grid"]

        for row_i, row in enumerate(grid):
            map_y = vp["y"] + row_i
            for col_i, code in enumerate(row):
                map_x = vp["x"] + col_i
                key = (seed, map_y, map_x)
                if key not in index:
                    index[key] = []
                index[key].append(code)

    return index


# ─────────────────────────────────────────────
# CORE ESTIMATOR
# ─────────────────────────────────────────────

def estimate(
    seed_map: SeedMap,
    transition_matrix: Dict[int, List[float]],
    obs_index: Dict[Tuple[int,int,int], List[int]],
    alpha: float = ALPHA,
    conditional_matrix: Optional[Dict[Tuple, List[float]]] = None,
) -> List[List[List[float]]]:
    """
    Build H×W×6 probability tensor for one seed.

    Uses conditional matrix (per-context bucket) when available,
    falling back to flat transition matrix per terrain code.

    Returns: tensor[y][x] = [P(class0), ..., P(class5)] summing to 1.0
    """
    H = seed_map.height
    W = seed_map.width

    # Precompute ocean distance map for spatial context (Plains cells)
    igrid = [[seed_map.cells[y][x].initial_code for x in range(W)] for y in range(H)]
    odm = _get_ocean_dist_grid(igrid, grid_id=f"seed_{seed_map.seed_index}")

    tensor = []

    for y in range(H):
        row = []
        for x in range(W):
            cell = seed_map.cells[y][x]
            code = cell.initial_code

            # ── LAYER A: static cells ──────────────────────────────
            if code in STATIC_CODES:
                pred_class = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC] * N_CLASSES
                dist[pred_class] = 1.0 - (FLOOR_STATIC * (N_CLASSES - 1))

            else:
                # ── LAYER C: conditional or flat transition prior ──
                if conditional_matrix:
                    ctx = cell_context(seed_map.cells, y, x, H, W, ocean_dist_map=odm)
                    prior = conditional_matrix.get(
                        ctx,
                        transition_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
                    )[:]
                else:
                    prior = transition_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]

                # ── LAYER B: Bayesian update if observed ───────────
                key = (seed_map.seed_index, y, x)
                if key in obs_index:
                    observed_codes = obs_index[key]
                    n_obs = len(observed_codes)
                    # Average across multiple observations of same cell
                    avg_one_hot = [0.0] * N_CLASSES
                    for obs_code in observed_codes:
                        obs_class = CODE_TO_CLASS.get(obs_code, 0)
                        avg_one_hot[obs_class] += 1.0 / n_obs

                    dist = [
                        (1 - alpha) * prior[i] + alpha * avg_one_hot[i]
                        for i in range(N_CLASSES)
                    ]
                else:
                    dist = prior[:]

                # Apply temperature scaling (softens predictions to hedge uncertainty)
                if TEMPERATURE != 1.0:
                    dist = _apply_temperature(dist, TEMPERATURE)

                # Apply per-terrain floor
                floor = TERRAIN_FLOOR.get(code, FLOOR_DYNAMIC)
                dist = [max(v, floor) for v in dist]

            # ── Renormalize ────────────────────────────────────────
            total = sum(dist)
            dist = [v / total for v in dist]
            row.append(dist)
        tensor.append(row)

    return tensor


def estimate_all_seeds(
    seed_maps: List[SeedMap],
    observations: Optional[List[dict]] = None,
    alpha: float = ALPHA,
    conditional_matrix: Optional[dict] = None,
) -> List[List[List[List[float]]]]:
    """
    Build tensors for all 5 seeds.
    Returns: list of 5 tensors, each H×W×6.
    """
    transition_matrix = load_transition_matrix()
    if conditional_matrix is None:
        conditional_matrix = load_conditional_matrix()
    obs_index = build_observation_index(observations or [])

    n_obs = len(obs_index)
    n_cells = config.MAP_WIDTH * config.MAP_HEIGHT
    print(f"\n  Transition matrix loaded.")
    print(f"  Observations indexed: {n_obs} cells observed "
          f"({n_obs/n_cells*100:.1f}% of {n_cells} total per seed)")
    print(f"  alpha = {alpha}  |  temp = {TEMPERATURE}  |  floor = {FLOOR_DYNAMIC}")

    tensors = []
    for sm in seed_maps:
        tensor = estimate(sm, transition_matrix, obs_index, alpha, conditional_matrix)
        tensors.append(tensor)
        obs_this_seed = sum(
            1 for (s, y, x) in obs_index if s == sm.seed_index
        )
        print(f"  Seed {sm.seed_index}: tensor built — "
              f"{obs_this_seed} cells from observations, "
              f"{n_cells - obs_this_seed} from transition prior")

    return tensors
