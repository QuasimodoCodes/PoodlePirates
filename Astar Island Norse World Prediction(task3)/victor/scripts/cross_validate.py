"""
cross_validate.py — Leave-one-round-out cross-validation.

For each of the 3 completed rounds:
  - Build transition matrix from the OTHER 2 rounds
  - Predict on the held-out round
  - Score against its ground truth

This tells us how well our model generalises to a brand new round
(exactly the situation we're in with round 4).

Also tests the effect of alpha (Bayesian smoothing weight) — but since
we have no observations here, it only tests the transition matrix (Layer C + A).

Run from victor/ folder:
    python -m scripts.cross_validate
"""

import os
import sys
import json
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import build_seed_maps, STATIC_CODES, CODE_TO_CLASS
from src.model.terrain_estimator import estimate, build_observation_index
from src.prediction.tensor_builder import validate_tensor

FLOOR_DYNAMIC = 0.01
FLOOR_STATIC  = 1e-5
N_CLASSES     = config.NUM_TERRAIN_CLASSES


# ── Scoring ──────────────────────────────────────────────────────────

def entropy(dist):
    return -sum(p * math.log(p) for p in dist if p > 1e-12)

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
            ent = entropy(ground_truth[y][x])
            kl  = kl_divergence(ground_truth[y][x], prediction[y][x])
            weighted_kl   += ent * kl
            total_entropy += ent
    if total_entropy < 1e-12:
        return 100.0
    wkl = weighted_kl / total_entropy
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))


# ── Transition matrix builder (from subset of files) ─────────────────

def build_transition_matrix_from_files(files: list, history_dir: str) -> dict:
    accumulator = defaultdict(list)
    for fname in files:
        with open(os.path.join(history_dir, fname)) as f:
            data = json.load(f)
        gt    = data.get("ground_truth")
        igrid = data.get("initial_grid")
        if not gt or not igrid:
            continue
        H, W = len(igrid), len(igrid[0])
        for y in range(H):
            for x in range(W):
                code = igrid[y][x]
                accumulator[code].append(gt[y][x])

    matrix = {}
    for code, samples in accumulator.items():
        n = len(samples)
        matrix[code] = [sum(s[i] for s in samples) / n for i in range(N_CLASSES)]

    # Fill missing codes
    uniform = [1.0 / N_CLASSES] * N_CLASSES
    for code in [0, 1, 2, 3, 4, 5, 10, 11]:
        if code not in matrix:
            matrix[code] = uniform[:]
    return matrix


# ── Predict for one file ─────────────────────────────────────────────

def predict_file(fpath: str, transition_matrix: dict) -> float:
    with open(fpath) as f:
        data = json.load(f)
    gt    = data.get("ground_truth")
    igrid = data.get("initial_grid")
    if not gt or not igrid:
        return None

    raw = {
        "map_width": len(igrid[0]),
        "map_height": len(igrid),
        "seeds_count": 1,
        "initial_states": [{"grid": igrid, "settlements": []}],
    }
    seed_map = build_seed_maps(raw)[0]
    prediction = estimate(seed_map, transition_matrix, obs_index={})
    validate_tensor(prediction, seed_index=0)
    return score_tensor(prediction, gt)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    all_files   = sorted([f for f in os.listdir(history_dir) if f.endswith("_analysis.json")])

    # Group by round_id (first 36 chars of filename)
    rounds = defaultdict(list)
    for f in all_files:
        round_id = f[:36]
        rounds[round_id].append(f)

    round_ids = sorted(rounds.keys())
    print(f"Rounds found: {len(round_ids)}")
    for rid in round_ids:
        print(f"  {rid}: {len(rounds[rid])} seeds")

    print(f"\n{'═'*65}")
    print(f"  LEAVE-ONE-ROUND-OUT CROSS-VALIDATION")
    print(f"{'═'*65}")

    all_cv_scores = []

    for test_round in round_ids:
        train_files = [f for rid in round_ids if rid != test_round for f in rounds[rid]]
        test_files  = rounds[test_round]

        # Build transition matrix from training rounds
        matrix = build_transition_matrix_from_files(train_files, history_dir)

        # Score on held-out round
        scores = []
        for fname in test_files:
            fpath = os.path.join(history_dir, fname)
            s = predict_file(fpath, matrix)
            if s is not None:
                scores.append(s)

        avg = sum(scores) / len(scores) if scores else 0
        all_cv_scores.extend(scores)

        print(f"\n  Test round: {test_round}")
        print(f"  Trained on: {len(train_files)} files from {len(round_ids)-1} other rounds")
        for fname, s in zip(test_files, scores):
            seed_label = fname.split("_seed")[1].split("_")[0]
            print(f"    seed {seed_label}: {s:.2f}")
        print(f"  → Round avg: {avg:.2f}")

    print(f"\n{'═'*65}")
    overall = sum(all_cv_scores) / len(all_cv_scores) if all_cv_scores else 0
    print(f"  Overall CV score: {overall:.2f} / 100")
    print(f"  Min: {min(all_cv_scores):.2f}   Max: {max(all_cv_scores):.2f}")
    print()

    # Compare to full-matrix score
    print(f"  Reminder: training on ALL 3 rounds gave 63.66 avg.")
    print(f"  CV score shows how we generalise to a UNSEEN round.")
    if overall >= 55:
        print(f"  ✅ Strong generalisation — transition matrix is stable across rounds.")
    elif overall >= 40:
        print(f"  ⚠️  Moderate generalisation — hidden parameters vary a lot between rounds.")
    else:
        print(f"  ❌ Poor generalisation — each round's parameters dominate the outcome.")
    print()
    print(f"  This CV score is our honest expected score for round 4.")


if __name__ == "__main__":
    main()
