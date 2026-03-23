"""
test_twophase_offline.py — Validate two-phase adaptive querying on historical data.

For each saved analysis file (ground truth known), simulate three strategies:
  A) Baseline:   transition matrix only, zero observations
  B) Old:        9-tile uniform scan, all cells observed, α=0.55
  C) New:        5-tile phase1 + 5 entropy-guided phase2 tiles, α=0.30

Uses ground_truth argmax as the "simulated observation" for each cell within a tile.
This lets us compare strategies without spending real queries.

Run from victor/ folder:
    python -m scripts.test_twophase_offline
"""

import os
import sys
import json
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import build_seed_maps, CODE_TO_CLASS, STATIC_CODES
from src.model.terrain_estimator import load_transition_matrix, FLOOR_DYNAMIC, FLOOR_STATIC
from src.observation.adaptive_planner import PHASE1_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N_CLASSES = config.NUM_TERRAIN_CLASSES

# ── Scoring ──────────────────────────────────────────────────────────

def kl_divergence(p, q):
    result = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-12:
            result += pi * math.log(pi / max(qi, 1e-12))
    return result

def score_tensor(prediction, ground_truth):
    H, W = len(ground_truth), len(ground_truth[0])
    weighted_kl = total_entropy = 0.0
    for y in range(H):
        for x in range(W):
            ent = _entropy(ground_truth[y][x])
            kl  = kl_divergence(ground_truth[y][x], prediction[y][x])
            weighted_kl   += ent * kl
            total_entropy += ent
    if total_entropy < 1e-12:
        return 100.0
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * (weighted_kl / total_entropy))))


# ── Simulated observation ─────────────────────────────────────────────

def simulated_obs_code(ground_truth_dist: List[float]) -> int:
    """Return argmax class of ground truth as the 'observed' code."""
    return max(range(N_CLASSES), key=lambda i: ground_truth_dist[i])


# ── Prediction builder ────────────────────────────────────────────────

def build_prediction(igrid, ground_truth, transition_matrix, observed_cells: Set[Tuple[int,int]], alpha: float):
    """
    Build H×W×6 tensor.
    observed_cells: set of (y, x) cells we 'query' — use GT argmax as observation.
    """
    H, W = len(igrid), len(igrid[0])
    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pred_class = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC] * N_CLASSES
                dist[pred_class] = 1.0 - FLOOR_STATIC * (N_CLASSES - 1)
            else:
                prior = transition_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                if (y, x) in observed_cells:
                    obs_class = simulated_obs_code(ground_truth[y][x])
                    one_hot = [0.0] * N_CLASSES
                    one_hot[obs_class] = 1.0
                    dist = [(1 - alpha) * prior[i] + alpha * one_hot[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, FLOOR_DYNAMIC) for v in dist]
            total = sum(dist)
            row.append([v / total for v in dist])
        tensor.append(row)
    return tensor


# ── Tile cells ────────────────────────────────────────────────────────

def tile_cells(ax, ay) -> Set[Tuple[int,int]]:
    return set(_covered_cells(ax, ay))


# ── Phase 2: entropy-guided tile selection ────────────────────────────

def entropy_guided_tiles(tensor, observed: Set[Tuple[int,int]], n_tiles: int = 5):
    """Greedy selection of n_tiles maximising entropy of unobserved cells."""
    H, W = len(tensor), len(tensor[0])
    all_anchors = [
        (ax, ay)
        for ay in range(H - TILE_H + 1)
        for ax in range(W - TILE_W + 1)
    ]
    anchor_cells = {(ax, ay): tile_cells(ax, ay) for ax, ay in all_anchors}

    entropy_map = {
        (y, x): _entropy(tensor[y][x])
        for y in range(H) for x in range(W)
        if (y, x) not in observed
    }

    selected = []
    remaining = dict(entropy_map)

    for _ in range(n_tiles):
        best, best_score = None, -1.0
        for ax, ay in all_anchors:
            score = sum(remaining.get(c, 0.0) for c in anchor_cells[(ax, ay)])
            if score > best_score:
                best_score, best = score, (ax, ay)
        if best is None or best_score <= 1e-9:
            break
        selected.append(best)
        for c in anchor_cells[best]:
            remaining.pop(c, None)

    return selected


# ── Old strategy: 9-tile uniform grid ────────────────────────────────

OLD_ANCHORS = [(ax, ay) for ay in [0, 15, 25] for ax in [0, 15, 25]]  # 9 tiles


# ── Main ──────────────────────────────────────────────────────────────

def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))

    if not files:
        print("No analysis files found.")
        return

    matrix = load_transition_matrix(calibrated=False)

    print(f"Testing on {len(files)} historical files...\n")
    print(f"  Strategy A: transition matrix only (α=0, no observations)")
    print(f"  Strategy B: 9-tile uniform scan    (α=0.10, testing lower alpha)")
    print(f"  Strategy C: 5+5 adaptive two-phase (α=0.10, new approach)")
    print()
    print(f"  {'File':<50} {'A':>6} {'B':>6} {'C':>6} {'B-A':>6} {'C-A':>6} {'C-B':>6}")
    print("  " + "─" * 88)

    scores_A, scores_B, scores_C = [], [], []

    for fname in files:
        with open(os.path.join(history_dir, fname)) as f:
            data = json.load(f)
        gt    = data.get("ground_truth")
        igrid = data.get("initial_grid")
        if not gt or not igrid:
            continue

        # ── Strategy A: no observations ──────────────────────────────
        pred_A = build_prediction(igrid, gt, matrix, observed_cells=set(), alpha=0.0)
        score_A = score_tensor(pred_A, gt)

        # ── Strategy B: 9-tile uniform, α=0.55 ───────────────────────
        obs_B = set()
        for ax, ay in OLD_ANCHORS:
            obs_B |= tile_cells(ax, ay)
        pred_B = build_prediction(igrid, gt, matrix, obs_B, alpha=0.10)
        score_B = score_tensor(pred_B, gt)

        # ── Strategy C: two-phase adaptive, α=0.30 ───────────────────
        # Phase 1: 5 corner+centre tiles
        obs_C = set()
        for ax, ay in PHASE1_ANCHORS:
            obs_C |= tile_cells(ax, ay)
        pred_C1 = build_prediction(igrid, gt, matrix, obs_C, alpha=0.10)

        # Phase 2: 5 entropy-guided tiles
        phase2_anchors = entropy_guided_tiles(pred_C1, obs_C, n_tiles=5)
        for ax, ay in phase2_anchors:
            obs_C |= tile_cells(ax, ay)
        pred_C = build_prediction(igrid, gt, matrix, obs_C, alpha=0.10)
        score_C = score_tensor(pred_C, gt)

        scores_A.append(score_A)
        scores_B.append(score_B)
        scores_C.append(score_C)

        label = fname[:48]
        print(f"  {label:<50} {score_A:6.2f} {score_B:6.2f} {score_C:6.2f} "
              f"{score_B-score_A:+6.2f} {score_C-score_A:+6.2f} {score_C-score_B:+6.2f}")

    print("  " + "─" * 88)
    avg_A = sum(scores_A) / len(scores_A)
    avg_B = sum(scores_B) / len(scores_B)
    avg_C = sum(scores_C) / len(scores_C)

    print(f"\n  {'AVERAGE':<50} {avg_A:6.2f} {avg_B:6.2f} {avg_C:6.2f} "
          f"{avg_B-avg_A:+6.2f} {avg_C-avg_A:+6.2f} {avg_C-avg_B:+6.2f}")
    print(f"  {'BEST':<50} {max(scores_A):6.2f} {max(scores_B):6.2f} {max(scores_C):6.2f}")
    print(f"  {'WORST':<50} {min(scores_A):6.2f} {min(scores_B):6.2f} {min(scores_C):6.2f}")

    print()
    if avg_C > avg_B:
        print(f"  ✅ Two-phase adaptive (C) beats uniform scan (B) by {avg_C-avg_B:+.2f} pts avg")
    else:
        print(f"  ⚠️  Uniform scan (B) still leads by {avg_B-avg_C:.2f} pts — reconsider strategy")

    if avg_C > avg_A:
        print(f"  ✅ Two-phase adaptive (C) beats no-observe baseline by {avg_C-avg_A:+.2f} pts")
    else:
        print(f"  ❌ Two-phase adaptive (C) WORSE than no-observe baseline — α too high")


if __name__ == "__main__":
    main()
