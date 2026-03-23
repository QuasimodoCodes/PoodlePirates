"""
inspect_round4.py — Show what we actually observed in round 4
           and how confident our predictions were.

Run from victor/ folder:
    python -m scripts.inspect_round4
"""

import os
import sys
import json
import math
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DATA_DIR = config.DATA_DIR
OBS_DIR  = os.path.join(DATA_DIR, "observations")
PRED_DIR = os.path.join(DATA_DIR, "predictions")

CODE_NAMES = {
    0: "Empty",
    1: "Settlement",
    2: "Port",
    3: "Ruin",
    4: "Forest",
    5: "Mountain",
    10: "Ocean",
    11: "Plains",
}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

TERRAIN_CHAR = {
    0: "·",   # Empty
    1: "S",   # Settlement
    2: "P",   # Port
    3: "R",   # Ruin
    4: "F",   # Forest
    5: "M",   # Mountain
    10: "~",  # Ocean
    11: ".",  # Plains
}


def load_observations_for_seed(seed_idx):
    """Load all grid observations for a given seed into a 40x40 map."""
    grid = [[None] * config.MAP_WIDTH for _ in range(config.MAP_HEIGHT)]
    files = [f for f in os.listdir(OBS_DIR) if f.startswith(f"seed{seed_idx}_")]
    for fname in sorted(files):
        with open(os.path.join(OBS_DIR, fname)) as f:
            data = json.load(f)
        resp = data["response"]
        vp   = resp["viewport"]
        obs_grid = resp["grid"]
        for dy, row in enumerate(obs_grid):
            for dx, code in enumerate(row):
                y, x = vp["y"] + dy, vp["x"] + dx
                if 0 <= y < config.MAP_HEIGHT and 0 <= x < config.MAP_WIDTH:
                    grid[y][x] = code
    return grid


def entropy(dist):
    return -sum(p * math.log(p) for p in dist if p > 1e-12)


def load_tensor(seed_idx):
    path = os.path.join(PRED_DIR, f"seed{seed_idx}_tensor.json")
    with open(path) as f:
        data = json.load(f)
    return data["tensor"]


def print_ascii_map(grid, title):
    print(f"\n  {title}")
    print("  " + "─" * config.MAP_WIDTH)
    for row in grid:
        chars = "".join(TERRAIN_CHAR.get(c, "?") if c is not None else " " for c in row)
        print(f"  |{chars}|")
    print("  " + "─" * config.MAP_WIDTH)
    print("  Legend: ~=Ocean  .=Plains  S=Settlement  P=Port  F=Forest  M=Mountain  ·=Empty  R=Ruin")


def analyse_seed(seed_idx):
    print(f"\n{'═'*65}")
    print(f"  SEED {seed_idx}")
    print(f"{'═'*65}")

    # ── Observed terrain ──────────────────────────────────────────
    obs_grid = load_observations_for_seed(seed_idx)
    flat     = [c for row in obs_grid for c in row if c is not None]
    counts   = Counter(flat)
    total    = len(flat)

    print(f"\n  Observed terrain at year 50 ({total} cells):")
    for code in sorted(counts, key=lambda c: -counts[c]):
        pct  = 100 * counts[code] / total
        name = CODE_NAMES.get(code, f"code{code}")
        bar  = "█" * int(pct / 2)
        print(f"    {name:<12} {counts[code]:5d} cells  {pct:5.1f}%  {bar}")

    print_ascii_map(obs_grid, f"Seed {seed_idx} — what we SAW at year 50 (from simulate())")

    # ── Prediction confidence ─────────────────────────────────────
    tensor = load_tensor(seed_idx)
    H, W = len(tensor), len(tensor[0])

    entropies    = []
    top_class    = Counter()
    high_conf    = 0  # cells where top class > 0.7
    very_conf    = 0  # cells where top class > 0.9

    conf_grid = [[" "] * W for _ in range(H)]

    for y in range(H):
        for x in range(W):
            dist = tensor[y][x]
            e    = entropy(dist)
            entropies.append(e)
            best_class = max(range(len(dist)), key=lambda i: dist[i])
            best_prob  = dist[best_class]
            top_class[CLASS_NAMES[best_class]] += 1
            if best_prob > 0.9:
                very_conf += 1
                conf_grid[y][x] = "■"
            elif best_prob > 0.7:
                high_conf += 1
                conf_grid[y][x] = "▪"
            else:
                conf_grid[y][x] = "·"

    avg_ent  = sum(entropies) / len(entropies)
    max_ent  = math.log(config.NUM_TERRAIN_CLASSES)

    print(f"\n  Prediction confidence:")
    print(f"    Avg entropy:      {avg_ent:.3f}  (max possible: {max_ent:.3f})")
    print(f"    Very confident (>90%): {very_conf:4d} cells")
    print(f"    Confident      (>70%): {high_conf:4d} cells")
    print(f"    Uncertain      (≤70%): {H*W - very_conf - high_conf:4d} cells")

    print(f"\n  Predicted dominant class per cell:")
    for cls, cnt in sorted(top_class.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / (H * W)
        bar = "█" * int(pct / 2)
        print(f"    {cls:<12} {cnt:5d} cells  {pct:5.1f}%  {bar}")

    print(f"\n  Confidence map (■ = >90%  ▪ = >70%  · = uncertain):")
    print("  " + "─" * W)
    for row in conf_grid:
        print("  |" + "".join(row) + "|")
    print("  " + "─" * W)


def compare_observed_vs_transition():
    """Compare what we actually saw vs what transition matrix predicts."""
    from src.model.terrain_estimator import load_transition_matrix
    matrix = load_transition_matrix()

    # Load initial states
    initial_path = os.path.join(DATA_DIR, "initial_states.json")
    with open(initial_path) as f:
        raw = json.load(f)

    print(f"\n{'═'*65}")
    print(f"  TRANSITION MATRIX CHECK — Did round 4 match past rounds?")
    print(f"{'═'*65}")
    print(f"  For each initial terrain code, comparing:")
    print(f"  what the matrix PREDICTS → what we actually SAW in observations")
    print()

    # Aggregate: for each initial code, what did we observe at year 50?
    from src.model.initial_analyzer import build_seed_maps
    seed_maps = build_seed_maps(raw)

    code_to_observed = defaultdict(Counter)  # initial_code → Counter of observed codes

    for sm in seed_maps:
        seed_idx = sm.seed_index
        obs = load_observations_for_seed(seed_idx)
        igrid = raw["initial_states"][seed_idx]["grid"]
        H, W = len(igrid), len(igrid[0])
        for y in range(H):
            for x in range(W):
                init_code = igrid[y][x]
                obs_code  = obs[y][x]
                if obs_code is not None:
                    code_to_observed[init_code][obs_code] += 1

    CODE_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

    for init_code in sorted(code_to_observed.keys()):
        counts = code_to_observed[init_code]
        total  = sum(counts.values())
        name   = CODE_NAMES.get(init_code, f"code{init_code}")

        # Build observed distribution over 6 classes
        obs_dist = [0.0] * config.NUM_TERRAIN_CLASSES
        for obs_code, cnt in counts.items():
            cls = CODE_TO_CLASS.get(obs_code, 0)
            obs_dist[cls] += cnt / total

        # Matrix prediction
        pred_dist = matrix.get(init_code, [1/6]*6)

        print(f"  Initial: {name} (code {init_code})  n={total}")
        for i, cname in enumerate(CLASS_NAMES):
            pred_bar = "█" * int(pred_dist[i] * 20)
            obs_bar  = "█" * int(obs_dist[i]  * 20)
            diff     = obs_dist[i] - pred_dist[i]
            diff_str = f"{diff:+.3f}"
            if abs(diff) > 0.1:
                diff_str = f"⚠ {diff_str}"
            print(f"    {cname:<12}  pred={pred_dist[i]:.3f} {pred_bar:<20}  obs={obs_dist[i]:.3f} {obs_bar:<20}  Δ={diff_str}")
        print()


def main():
    print(f"\n{'═'*65}")
    print(f"  ROUND 4 — LOCAL INSPECTION")
    print(f"  What we observed vs what we predicted")
    print(f"{'═'*65}")

    # Show seed 0 in detail, others briefly
    analyse_seed(0)

    for seed_idx in range(1, config.NUM_SEEDS):
        obs_grid = load_observations_for_seed(seed_idx)
        tensor   = load_tensor(seed_idx)
        flat     = [c for row in obs_grid for c in row if c is not None]
        counts   = Counter(flat)
        total    = len(flat)
        ents     = [entropy(tensor[y][x]) for y in range(len(tensor)) for x in range(len(tensor[0]))]
        avg_ent  = sum(ents) / len(ents)
        print(f"\n  Seed {seed_idx}: {total} cells observed  |  avg entropy={avg_ent:.3f}")
        dominant = sorted(counts.items(), key=lambda x: -x[1])[:4]
        for code, cnt in dominant:
            print(f"    {CODE_NAMES.get(code,'?'):<12} {100*cnt/total:.1f}%")

    compare_observed_vs_transition()


if __name__ == "__main__":
    main()
