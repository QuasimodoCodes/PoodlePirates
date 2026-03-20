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
FLOOR_DYNAMIC = 0.001 # floor for dynamic cells — sweep: 0.001->72.48 vs 0.01->70.87
FLOOR_STATIC = 1e-5   # floor for static cells (Mountain/Ocean)
N_CLASSES = config.NUM_TERRAIN_CLASSES


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

                # Apply dynamic floor
                dist = [max(v, FLOOR_DYNAMIC) for v in dist]

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
