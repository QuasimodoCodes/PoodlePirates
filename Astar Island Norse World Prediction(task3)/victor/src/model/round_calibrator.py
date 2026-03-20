"""
round_calibrator.py — Build a round-specific transition matrix from this round's observations.

After running all queries we have observed year-50 terrain for every cell.
Combined with the initial grid we can compute how THIS round's hidden parameters
actually drove terrain change — and use that as a better prior than the historical average.

Blending formula (per initial terrain code):
    blended[code] = (n_round * round_freq[code] + N_HIST * historical[code])
                    / (n_round + N_HIST)

N_HIST is a "virtual" sample count representing confidence in the historical matrix.
  - N_HIST = 300: with 100 round observations, round gets ~25% weight
  - N_HIST = 100: with 100 round observations, round gets ~50% weight

This auto-adapts: codes with many observations (Plains n≈5000) trust history less;
codes with few observations (Port n≈14) stay close to history.

Called from main.py after observations, before predictions.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES

N_CLASSES    = config.NUM_TERRAIN_CLASSES
N_HIST       = 50    # virtual historical sample weight — lower = trust round obs more
                     # Sweep: N_HIST=50 → 24.89, N_HIST=2000 → 24.59 (leave-one-out test)
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


def calibrate(
    initial_states: dict,
    observations: List[dict],
    historical_matrix: Dict[int, List[float]],
    verbose: bool = True,
) -> Dict[int, List[float]]:
    """
    Build a round-calibrated transition matrix.

    Args:
        initial_states: raw dict from GET /rounds/{id} or initial_states.json
        observations:   list of saved observation dicts from data/observations/
        historical_matrix: loaded from data/transition_matrix.json
        verbose: print calibration report

    Returns:
        blended matrix {initial_code: [P(class0)..P(class5)]}
    """
    obs_index = _build_obs_index(observations)

    # Accumulate (initial_code → list of observed classes) for this round
    round_counts = defaultdict(list)   # code → [obs_class, ...]

    for seed_idx, seed_state in enumerate(initial_states.get("initial_states", [])):
        igrid = seed_state["grid"]
        H, W  = len(igrid), len(igrid[0])
        for y in range(H):
            for x in range(W):
                init_code = igrid[y][x]
                if init_code in STATIC_CODES:
                    continue
                obs_code = obs_index.get((seed_idx, y, x))
                if obs_code is not None:
                    round_counts[init_code].append(CODE_TO_CLASS.get(obs_code, 0))

    # Build blended matrix
    blended = {}
    for code in [0, 1, 2, 3, 4, 5, 10, 11]:
        hist = historical_matrix.get(code, [1.0 / N_CLASSES] * N_CLASSES)

        if code in STATIC_CODES:
            blended[code] = hist[:]
            continue

        obs_list = round_counts.get(code, [])
        n_round  = len(obs_list)

        if n_round == 0:
            blended[code] = hist[:]
            continue

        # Empirical round frequency
        round_freq = [0.0] * N_CLASSES
        for cls in obs_list:
            round_freq[cls] += 1.0 / n_round

        # Weighted blend
        total      = n_round + N_HIST
        blended[code] = [
            (n_round * round_freq[i] + N_HIST * hist[i]) / total
            for i in range(N_CLASSES)
        ]

    if verbose:
        _print_report(historical_matrix, blended, round_counts)

    return blended


def _print_report(
    historical: Dict[int, List[float]],
    blended:    Dict[int, List[float]],
    round_counts: dict,
) -> None:
    print(f"\n  Round calibration — historical vs this round:")
    print(f"  (N_HIST={N_HIST} virtual samples — higher = more conservative)")
    print()

    any_flag = False
    for code in [1, 2, 4, 11]:
        name    = CODE_NAMES.get(code, f"code{code}")
        n_round = len(round_counts.get(code, []))
        hist    = historical.get(code, [])
        blend   = blended.get(code, [])
        weight  = n_round / (n_round + N_HIST) if n_round else 0.0

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
