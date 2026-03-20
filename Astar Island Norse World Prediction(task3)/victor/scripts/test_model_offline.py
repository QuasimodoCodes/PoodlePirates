"""
test_model_offline.py — Score our model against saved ground truth WITHOUT spending queries.

Uses: data/round_history/*_analysis.json (rounds 1-3 ground truth)
Uses: data/transition_matrix.json

For each saved analysis file:
  1. Load initial_grid (what the map looked like at start)
  2. Build prediction tensor using terrain_estimator (transition matrix only, no observations)
  3. Compare against ground_truth tensor using the real scoring formula
  4. Print per-seed score and overall estimate

This tells us our baseline score BEFORE touching round 4's budget.

Run from victor/ folder:
    python -m scripts.test_model_offline
"""

import os
import sys
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import build_seed_maps, STATIC_CODES, CODE_TO_CLASS
from src.model.terrain_estimator import estimate, load_transition_matrix, build_observation_index
from src.prediction.tensor_builder import validate_tensor


def entropy(dist):
    """Shannon entropy of a distribution."""
    return -sum(p * math.log(p) for p in dist if p > 1e-12)


def kl_divergence(p, q):
    """KL(p || q) — p=ground truth, q=our prediction."""
    result = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-12:
            qi_safe = max(qi, 1e-12)
            result += pi * math.log(pi / qi_safe)
    return result


def score_tensor(prediction, ground_truth):
    """Compute entropy-weighted KL score (0-100)."""
    H = len(ground_truth)
    W = len(ground_truth[0])

    weighted_kl = 0.0
    total_entropy = 0.0

    for y in range(H):
        for x in range(W):
            p = ground_truth[y][x]
            q = prediction[y][x]
            ent = entropy(p)
            kl = kl_divergence(p, q)
            weighted_kl += ent * kl
            total_entropy += ent

    if total_entropy < 1e-12:
        return 100.0

    wkl = weighted_kl / total_entropy
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))


def build_fake_seed_map(initial_grid, seed_index):
    """Build a minimal SeedMap from an initial_grid for testing."""
    raw = {
        "map_width": len(initial_grid[0]),
        "map_height": len(initial_grid),
        "seeds_count": 1,
        "initial_states": [{
            "grid": initial_grid,
            "settlements": []
        }]
    }
    maps = build_seed_maps(raw)
    maps[0].seed_index = seed_index
    return maps[0]


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted([f for f in os.listdir(history_dir) if f.endswith("_analysis.json")])

    if not files:
        print("No analysis files found. Run fetch_past_rounds.py first.")
        return

    transition_matrix = load_transition_matrix()
    all_scores = []

    print(f"Testing model on {len(files)} saved ground truth files...\n")
    print(f"{'File':<55} {'Score':>7}")
    print("─" * 65)

    for fname in files:
        fpath = os.path.join(history_dir, fname)
        with open(fpath) as f:
            data = json.load(f)

        ground_truth = data.get("ground_truth")
        initial_grid = data.get("initial_grid")

        if not ground_truth or not initial_grid:
            print(f"  {fname}: missing data — skip")
            continue

        # Build seed map from initial grid
        seed_map = build_fake_seed_map(initial_grid, seed_index=0)

        # Build prediction (transition matrix only — no observations)
        prediction = estimate(
            seed_map=seed_map,
            transition_matrix=transition_matrix,
            obs_index={},   # no observations
        )

        # Validate
        try:
            validate_tensor(prediction, seed_index=0)
        except ValueError as e:
            print(f"  {fname}: validation error — {e}")
            continue

        # Score
        score = score_tensor(prediction, ground_truth)
        all_scores.append(score)

        print(f"  {fname:<53} {score:7.2f}")

    print("─" * 65)
    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        best = max(all_scores)
        worst = min(all_scores)
        print(f"\n  Results across {len(all_scores)} files:")
        print(f"    Average score: {avg:.2f} / 100")
        print(f"    Best:          {best:.2f}")
        print(f"    Worst:         {worst:.2f}")
        print()
        if avg > 5:
            print(f"  ✅ Above uniform baseline (~1-5). Transition matrix is working.")
        else:
            print(f"  ⚠️  Near or below uniform baseline. Model needs tuning.")
        print()
        print(f"  This is our expected score in round 4 with ZERO simulate() queries.")
        print(f"  Every query we spend in round 4 should push this higher via Bayesian update.")


if __name__ == "__main__":
    main()
