"""
terrain_estimator.py — Step 9: Build per-cell H×W×6 probability distributions.

Three stacked layers (applied in order):

  LAYER A — Static cells (Mountain=5, Ocean=10)
    Predict with near-certainty from initial_states. Free, no queries.
    Floor: 1e-5

  LAYER C — Transition matrix prior (base for ALL dynamic cells)
    Real data from 3 rounds of ground truth (24,000 cells).
    transition_matrix[initial_code] = [P(class0)..P(class5)]
    Floor: 0.01

  LAYER B — Bayesian update when simulate() observations exist
    posterior = (1 - α) × transition_prior  +  α × one_hot(observed_class)
    α = 0.55 (tunable — see learning_log.md)
    Applied ON TOP of Layer C when we have a direct observation.
    Floor: 0.01

Order: C → B (if observed) → A (if static) → floor → renormalize
"""

import json
import os
from typing import List, Dict, Optional, Tuple

import config
from src.model.initial_analyzer import SeedMap, CODE_TO_CLASS, STATIC_CODES


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

ALPHA = 0.05          # Bayesian update weight for single observation
                      # Realistic MC sim (stochastic obs): a=0.05 N=50 -> 71.13 avg (best)
                      # a=0.10 N=50 -> 70.74, a=0.50 N=50 -> 48.01 (too noisy)
FLOOR_STATIC = 1e-5   # floor for static cells (Mountain/Ocean)
N_CLASSES = config.NUM_TERRAIN_CLASSES

# Per-terrain floors: lower for stable cells, higher for volatile
# Tested: +0.27 pts over uniform 0.005 floor (9-round LOO)
TERRAIN_FLOOR = {
    1: 0.008,   # Settlement — very unpredictable (46% Empty, 29% Sett, 22% Forest)
    2: 0.008,   # Port — very unpredictable
    3: 0.006,   # Ruin — moderately unpredictable
    4: 0.003,   # Forest — quite stable (77%)
    11: 0.004,  # Plains — fairly stable (82%)
    0: 0.005,   # Empty — unknown
}
FLOOR_DYNAMIC = 0.005  # fallback for any code not in TERRAIN_FLOOR


# ─────────────────────────────────────────────
# TRANSITION MATRIX LOADER
# ─────────────────────────────────────────────

def load_transition_matrix(calibrated: bool = True) -> Dict[int, List[float]]:
    """
    Load transition matrix. Prefers round_calibrated_matrix.json if available
    (built from this round's observations). Falls back to historical matrix.

    Args:
        calibrated: if True, use round-calibrated matrix when available
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

def _build_neighbor_maps(seed_map):
    """Count settlement and ocean neighbors within distance 2 for each dynamic cell."""
    H, W = seed_map.height, seed_map.width
    sett_counts = {}
    ocean_counts = {}
    for y in range(H):
        for x in range(W):
            code = seed_map.cells[y][x].initial_code
            if code in STATIC_CODES:
                continue
            s_count = 0
            o_count = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        nc = seed_map.cells[ny][nx].initial_code
                        if nc == 1:
                            s_count += 1
                        elif nc == 10:
                            o_count += 1
            if s_count > 0:
                sett_counts[(y, x)] = s_count
            if o_count > 0:
                ocean_counts[(y, x)] = o_count
    return sett_counts, ocean_counts


# Neighbor-aware adjustments based on 10-round data analysis:
#
# Settlement proximity (Forest/Plains):
#   Forest near sett: -16.8% Forest, +9.4% Settlement, +7.1% Empty
#   Plains near sett: -12.4% Empty, +8.8% Settlement
#   Tested: +0.94 pts with sett+empty boost (10-round LOO)
SETT_BOOST_PER = 0.01   # per nearby settlement
SETT_BOOST_MAX = 0.05   # max total
SETT_BOOST_CODES = {4, 11}  # Forest and Plains
#
# Ocean proximity (Forest/Plains/Settlement):
#   Near ocean: +2.7% Port, -3.8% Settlement (forest), -3.9% Settlement (plains)
#   Coastal settlements: +1.7% Port
#   Tested: +1.76 pts with ocean boost alone, +2.65 combined (10-round LOO)
OCEAN_BOOST_PER = 0.005  # per nearby ocean cell
OCEAN_BOOST_MAX = 0.03   # max total
OCEAN_BOOST_CODES = {1, 4, 11}  # Settlement, Forest, Plains


def estimate(
    seed_map: SeedMap,
    transition_matrix: Dict[int, List[float]],
    obs_index: Dict[Tuple[int,int,int], List[int]],
    alpha: float = ALPHA,
) -> List[List[List[float]]]:
    """
    Build H×W×6 probability tensor for one seed.

    Returns: tensor[y][x] = [P(class0), ..., P(class5)] summing to 1.0
    """
    H = seed_map.height
    W = seed_map.width

    # Precompute neighbor counts
    sett_neighbors, ocean_neighbors = _build_neighbor_maps(seed_map)

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
                # ── LAYER C: transition matrix prior ───────────────
                prior = transition_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]

                # ── Settlement neighbor adjustment ─────────────────
                nc = sett_neighbors.get((y, x), 0)
                if nc > 0 and code in SETT_BOOST_CODES:
                    boost = min(nc * SETT_BOOST_PER, SETT_BOOST_MAX)
                    prior[1] += boost        # class 1 = Settlement (expansion)
                    prior[0] += boost * 0.7  # class 0 = Empty (resource depletion)
                    total_p = sum(prior)
                    prior = [p / total_p for p in prior]

                # ── Ocean neighbor adjustment ──────────────────────
                oc = ocean_neighbors.get((y, x), 0)
                if oc > 0 and code in OCEAN_BOOST_CODES:
                    boost = min(oc * OCEAN_BOOST_PER, OCEAN_BOOST_MAX)
                    prior[2] += boost  # class 2 = Port
                    if code in (4, 11):  # suppress settlements near coast
                        prior[1] = max(prior[1] - boost * 0.5, 0.001)
                    total_p = sum(prior)
                    prior = [p / total_p for p in prior]

                # ── LAYER B: Bayesian update if observed ───────────
                key = (seed_map.seed_index, y, x)
                if key in obs_index:
                    observed_codes = obs_index[key]
                    # Average across multiple observations of same cell
                    avg_one_hot = [0.0] * N_CLASSES
                    for obs_code in observed_codes:
                        obs_class = CODE_TO_CLASS.get(obs_code, 0)
                        avg_one_hot[obs_class] += 1.0 / len(observed_codes)

                    dist = [
                        (1 - alpha) * prior[i] + alpha * avg_one_hot[i]
                        for i in range(N_CLASSES)
                    ]
                else:
                    dist = prior[:]

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
) -> List[List[List[List[float]]]]:
    """
    Build tensors for all 5 seeds.
    Returns: list of 5 tensors, each H×W×6.
    """
    transition_matrix = load_transition_matrix()
    obs_index = build_observation_index(observations or [])

    n_obs = len(obs_index)
    n_cells = config.MAP_WIDTH * config.MAP_HEIGHT
    print(f"\n  Transition matrix loaded.")
    print(f"  Observations indexed: {n_obs} cells observed "
          f"({n_obs/n_cells*100:.1f}% of {n_cells} total per seed)")
    print(f"  α = {alpha}  |  floor_dynamic = {FLOOR_DYNAMIC}  |  floor_static = {FLOOR_STATIC}")

    tensors = []
    for sm in seed_maps:
        tensor = estimate(sm, transition_matrix, obs_index, alpha)
        tensors.append(tensor)
        obs_this_seed = sum(
            1 for (s, y, x) in obs_index if s == sm.seed_index
        )
        print(f"  Seed {sm.seed_index}: tensor built — "
              f"{obs_this_seed} cells from observations, "
              f"{n_cells - obs_this_seed} from transition prior")

    return tensors
