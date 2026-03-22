"""
round_calibrator.py — Build round-specific transition matrices from this round's observations.

Supports both flat (per terrain code) and conditional (per context bucket) calibration.

Blending formula:
    blended[key] = (n_round * round_freq + N_HIST * historical) / (n_round + N_HIST)

Called from main.py after observations, before predictions.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.model.terrain_estimator import cell_context_from_grid, _get_ocean_dist_grid

N_CLASSES    = config.NUM_TERRAIN_CLASSES
N_HIST       = 50    # virtual historical sample weight — lower = trust round obs more
N_HIST_SURPRISED = 5  # N_HIST for context buckets with surprise > threshold
SURPRISE_THRESHOLD = 0.30  # symmetric KL threshold for "surprised" bucket
CLASS_NAMES  = ["Empty", "Settl", "Port", "Ruin", "Forest", "Mtn"]
CODE_NAMES   = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin",
                4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains"}


def _build_obs_index(observations: List[dict]) -> Dict[Tuple[int,int,int], int]:
    """(seed, y, x) → last observed terrain code."""
    index = {}
    for obs in observations:
        q = obs["query"]
        r = obs["response"]
        seed = q["seed_index"]
        vp   = r["viewport"]
        for row_i, row in enumerate(r["grid"]):
            map_y = vp["y"] + row_i
            for col_i, code in enumerate(row):
                map_x = vp["x"] + col_i
                index[(seed, map_y, map_x)] = code
    return index


def _blend(round_counts, historical_matrix, n_hist):
    """Build blended flat matrix with given N_HIST."""
    blended = {}
    for code in [0, 1, 2, 3, 4, 5, 10, 11]:
        hist = historical_matrix.get(code, [1.0 / N_CLASSES] * N_CLASSES)
        if code in STATIC_CODES:
            blended[code] = hist[:]
            continue
        obs_list = round_counts.get(code, [])
        n_round = len(obs_list)
        if n_round == 0:
            blended[code] = hist[:]
            continue
        round_freq = [0.0] * N_CLASSES
        for cls in obs_list:
            round_freq[cls] += 1.0 / n_round
        total = n_round + n_hist
        blended[code] = [
            (n_round * round_freq[i] + n_hist * hist[i]) / total
            for i in range(N_CLASSES)
        ]
    return blended


def _compute_surprise(obs_list, hist):
    """Compute symmetric KL divergence between round observations and historical."""
    import math
    n = len(obs_list)
    if n < 3:
        return 0.0
    rf = [0.0] * N_CLASSES
    for v in obs_list:
        rf[v] += 1.0 / n
    kl_fwd = sum(rf[i] * math.log(max(rf[i], 1e-12) / max(hist[i], 1e-12))
                 for i in range(N_CLASSES) if rf[i] > 1e-12)
    kl_rev = sum(hist[i] * math.log(max(hist[i], 1e-12) / max(rf[i], 1e-12))
                 for i in range(N_CLASSES) if hist[i] > 1e-12)
    return (kl_fwd + kl_rev) / 2


def _compute_global_shift(round_counts_ctx, conditional_matrix):
    """Compute global class-frequency ratio R[i] = round_freq[i] / hist_freq[i].

    Uses all round observations vs historical average across all buckets.
    Returns R as a list of length N_CLASSES, sqrt-damped (R^0.5).
    """
    import math

    # Aggregate all round observations across buckets
    round_total = [0.0] * N_CLASSES
    n_obs = 0
    for ctx, obs_list in round_counts_ctx.items():
        for cls in obs_list:
            round_total[cls] += 1
            n_obs += 1

    if n_obs == 0:
        return [1.0] * N_CLASSES

    round_freq = [c / n_obs for c in round_total]

    # Historical average across all buckets
    hist_total = [0.0] * N_CLASSES
    n_buckets = 0
    for ctx, dist in conditional_matrix.items():
        for i in range(N_CLASSES):
            hist_total[i] += dist[i]
        n_buckets += 1

    if n_buckets == 0:
        return [1.0] * N_CLASSES

    hist_freq = [h / n_buckets for h in hist_total]

    # Sqrt-damped ratio
    R = [1.0] * N_CLASSES
    for i in range(N_CLASSES):
        if hist_freq[i] > 1e-8:
            R[i] = math.sqrt(round_freq[i] / hist_freq[i])
    return R


def _apply_global_shift(dist, R):
    """Multiply distribution by global shift R and renormalize."""
    shifted = [max(dist[i] * R[i], 1e-12) for i in range(N_CLASSES)]
    s = sum(shifted)
    return [v / s for v in shifted]


def _blend_conditional(round_counts_ctx, conditional_matrix, n_hist,
                       n_hist_surprised=N_HIST_SURPRISED,
                       surprise_threshold=SURPRISE_THRESHOLD):
    """Build blended conditional matrix with per-bucket adaptive N_HIST.

    Buckets where round observations diverge heavily from historical
    (symmetric KL > threshold) get a lower N_HIST to trust round data more.
    Global shift multiplier is applied to historical priors before blending.
    """
    # Compute global shift from round observations
    R = _compute_global_shift(round_counts_ctx, conditional_matrix)

    blended = {}
    n_surprised = 0
    all_keys = set(list(conditional_matrix.keys()) + list(round_counts_ctx.keys()))
    for ctx in all_keys:
        hist_raw = conditional_matrix.get(ctx, [1.0 / N_CLASSES] * N_CLASSES)
        # Apply global shift to historical prior
        hist = _apply_global_shift(hist_raw, R)
        obs_list = round_counts_ctx.get(ctx, [])
        n_round = len(obs_list)
        if n_round == 0:
            blended[ctx] = hist[:]
            continue

        # Adaptive N_HIST: detect surprise per bucket (compare to shifted hist)
        surprise = _compute_surprise(obs_list, hist)
        if surprise > surprise_threshold and n_round >= 5:
            effective_n_hist = n_hist_surprised
            n_surprised += 1
        else:
            effective_n_hist = n_hist

        round_freq = [0.0] * N_CLASSES
        for cls in obs_list:
            round_freq[cls] += 1.0 / n_round
        total = n_round + effective_n_hist
        blended[ctx] = [
            (n_round * round_freq[i] + effective_n_hist * hist[i]) / total
            for i in range(N_CLASSES)
        ]
    return blended, n_surprised


HARD_ROUND_THRESHOLD = 5  # n_surprised >= this → use stepped alpha

def calibrate(
    initial_states: dict,
    observations: List[dict],
    historical_matrix: Dict[int, List[float]],
    verbose: bool = True,
    conditional_matrix: Dict[Tuple, List[float]] = None,
) -> Tuple[Dict[int, List[float]], dict]:
    """
    Build round-calibrated transition matrices (flat + conditional).

    Returns:
        (blended flat matrix, round_metrics dict)
        round_metrics includes 'n_surprised' and 'hard_round' flag.
    Also calibrates the conditional matrix in-place if provided.
    """
    obs_index = _build_obs_index(observations)

    # Accumulate per terrain code (flat) and per context (conditional)
    round_counts = defaultdict(list)       # code → [obs_class, ...]
    round_counts_ctx = defaultdict(list)   # ctx_tuple → [obs_class, ...]

    for seed_idx, seed_state in enumerate(initial_states.get("initial_states", [])):
        igrid = seed_state["grid"]
        H, W  = len(igrid), len(igrid[0])
        odm = _get_ocean_dist_grid(igrid, grid_id=f"cal_seed_{seed_idx}")
        for y in range(H):
            for x in range(W):
                init_code = igrid[y][x]
                if init_code in STATIC_CODES:
                    continue
                obs_code = obs_index.get((seed_idx, y, x))
                if obs_code is not None:
                    obs_class = CODE_TO_CLASS.get(obs_code, 0)
                    round_counts[init_code].append(obs_class)
                    if conditional_matrix is not None:
                        ctx = cell_context_from_grid(igrid, y, x, ocean_dist_map=odm)
                        round_counts_ctx[ctx].append(obs_class)

    n_hist = N_HIST
    blended = _blend(round_counts, historical_matrix, n_hist)

    # Calibrate conditional matrix if provided
    n_surprised = 0
    if conditional_matrix is not None:
        blended_cond, n_surprised = _blend_conditional(round_counts_ctx, conditional_matrix, n_hist)
        # Update conditional_matrix in-place so caller gets calibrated version
        conditional_matrix.clear()
        conditional_matrix.update(blended_cond)
        n_ctx_calibrated = sum(1 for ctx in round_counts_ctx if len(round_counts_ctx[ctx]) > 0)
        print(f"  Conditional matrix calibrated: {n_ctx_calibrated} context buckets updated"
              f" ({n_surprised} surprised, using N_HIST={N_HIST_SURPRISED}).")

    if verbose:
        _print_report(historical_matrix, blended, round_counts, n_hist)

    hard_round = n_surprised >= HARD_ROUND_THRESHOLD
    if hard_round:
        print(f"  Round flagged as HARD ({n_surprised} surprised buckets >= {HARD_ROUND_THRESHOLD})"
              f" -> using stepped alpha (0.03/0.10)")
    else:
        print(f"  Round is NORMAL ({n_surprised} surprised buckets < {HARD_ROUND_THRESHOLD})"
              f" -> using fixed alpha (0.05)")

    round_metrics = {
        "n_surprised": n_surprised,
        "hard_round": hard_round,
    }
    return blended, round_metrics


def _print_report(
    historical: Dict[int, List[float]],
    blended:    Dict[int, List[float]],
    round_counts: dict,
    n_hist: int = N_HIST,
) -> None:
    print(f"\n  Round calibration — historical vs this round:")
    print(f"  (N_HIST={n_hist} virtual samples — higher = more conservative)")
    print()

    any_flag = False
    for code in [1, 2, 4, 11]:
        name    = CODE_NAMES.get(code, f"code{code}")
        n_round = len(round_counts.get(code, []))
        hist    = historical.get(code, [])
        blend   = blended.get(code, [])
        weight  = n_round / (n_round + n_hist) if n_round else 0.0

        print(f"  {name} (code {code})  n={n_round}  round_weight={weight:.1%}")

        for i, cname in enumerate(CLASS_NAMES):
            delta = blend[i] - hist[i]
            flag  = " ⚠" if abs(delta) > 0.05 else "  "
            if abs(hist[i]) > 0.005 or abs(blend[i]) > 0.005:
                bar_h = "█" * int(hist[i]  * 20)
                bar_b = "█" * int(blend[i] * 20)
                print(f"    {cname:<8} hist={hist[i]:.3f} {bar_h:<20}  "
                      f"blend={blend[i]:.3f} {bar_b:<20}  Δ={delta:+.3f}{flag}")
                if abs(delta) > 0.05:
                    any_flag = True
        print()

    if any_flag:
        print("  ⚠  Large deltas detected — this round may have unusual parameters.")
        print("     The blended matrix will adapt the prior accordingly.")
    else:
        print("  ✅ Round dynamics match historical matrix closely.")


def save_calibrated_matrix(blended: Dict[int, List[float]]) -> None:
    """Save blended matrix to data/round_calibrated_matrix.json."""
    path = os.path.join(config.DATA_DIR, "round_calibrated_matrix.json")
    out  = {"transition_matrix": {str(k): v for k, v in blended.items()}}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Calibrated matrix saved to {path}")
