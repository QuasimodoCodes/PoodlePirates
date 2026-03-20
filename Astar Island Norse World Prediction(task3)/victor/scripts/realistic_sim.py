"""
realistic_sim.py - Monte Carlo simulation of the full query+predict pipeline.

Unlike alpha_sweep.py which uses GT argmax as observation, this script
SAMPLES from the ground truth distribution (like the real simulate() API).
Runs N_MC trials per seed to average out observation noise.

Tests alpha + N_HIST combos under realistic conditions.

Run from victor/ folder:
    python -m scripts.realistic_sim
"""

import os, sys, json, math, random
from collections import defaultdict
from typing import List, Dict, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import PHASE1_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N_CLASSES = config.NUM_TERRAIN_CLASSES
FLOOR_DYNAMIC = 0.01
FLOOR_STATIC = 1e-5

random.seed(42)

# -- Scoring ------------------------------------------------------------------

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


# -- Tile helpers -------------------------------------------------------------

def tile_cells(ax, ay) -> Set[Tuple[int,int]]:
    return set(_covered_cells(ax, ay))

def entropy_guided_tiles(tensor, observed: Set[Tuple[int,int]], n_tiles: int = 5):
    H, W = len(tensor), len(tensor[0])
    all_anchors = [(ax, ay) for ay in range(H - TILE_H + 1) for ax in range(W - TILE_W + 1)]
    anchor_cells = {a: tile_cells(*a) for a in all_anchors}
    remaining = {(y, x): _entropy(tensor[y][x]) for y in range(H) for x in range(W) if (y, x) not in observed}
    selected = []
    for _ in range(n_tiles):
        best, best_score = None, -1.0
        for a in all_anchors:
            score = sum(remaining.get(c, 0.0) for c in anchor_cells[a])
            if score > best_score:
                best_score, best = score, a
        if best is None or best_score <= 1e-9:
            break
        selected.append(best)
        for c in anchor_cells[best]:
            remaining.pop(c, None)
    return selected


# -- Stochastic observation sampling ------------------------------------------

def sample_observation(gt_dist):
    """Sample one class from ground truth distribution (like real simulate())."""
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(gt_dist):
        cumsum += p
        if r < cumsum:
            return i
    return len(gt_dist) - 1


# -- Prediction builder (with sampled observations) --------------------------

def build_prediction_sampled(igrid, ground_truth, matrix, observed_cells, alpha, obs_samples):
    """
    obs_samples: dict of (y,x) -> sampled class (from sample_observation)
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
                prior = matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                if (y, x) in observed_cells and alpha > 0:
                    obs_class = obs_samples[(y, x)]
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


# -- Simulated round calibration (with sampled obs) --------------------------

def simulate_calibration(igrid, observed_cells, obs_samples, historical_matrix, n_hist):
    round_counts = defaultdict(list)
    for y, x in observed_cells:
        code = igrid[y][x]
        if code in STATIC_CODES:
            continue
        round_counts[code].append(obs_samples[(y, x)])

    blended = {}
    for code in [0, 1, 2, 3, 4, 5, 10, 11]:
        hist = historical_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
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
        blended[code] = [(n_round * round_freq[i] + n_hist * hist[i]) / total for i in range(N_CLASSES)]
    return blended


# -- LOO matrix builder -------------------------------------------------------

def build_matrix_excluding_round(all_files, exclude_round_id, history_dir):
    counts = defaultdict(lambda: [0.0] * N_CLASSES)
    for fname in all_files:
        round_id = fname.split("_seed")[0]
        if round_id == exclude_round_id:
            continue
        with open(os.path.join(history_dir, fname)) as f:
            data = json.load(f)
        gt = data.get("ground_truth")
        igrid = data.get("initial_grid")
        if not gt or not igrid:
            continue
        H, W = len(igrid), len(igrid[0])
        for y in range(H):
            for x in range(W):
                code = igrid[y][x]
                if code in STATIC_CODES:
                    continue
                gt_class = max(range(N_CLASSES), key=lambda i: gt[y][x][i])
                counts[code][gt_class] += 1.0
    matrix = {}
    for code, freq in counts.items():
        total = sum(freq)
        if total > 0:
            matrix[code] = [f / total for f in freq]
        else:
            matrix[code] = [1.0/N_CLASSES] * N_CLASSES
    for code in [0, 1, 2, 3, 4, 5, 10, 11]:
        if code not in matrix:
            matrix[code] = [1.0/N_CLASSES] * N_CLASSES
    return matrix


# -- Main ---------------------------------------------------------------------

N_MC = 10  # Monte Carlo trials per seed

def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files:
        rounds[f.split("_seed")[0]].append(f)

    print(f"Realistic Monte Carlo simulation ({N_MC} trials per seed)")
    print(f"Observations SAMPLED from GT distribution (not argmax)")
    print(f"{len(files)} files, {len(rounds)} rounds, leave-one-round-out")
    print()

    # Test combos
    combos = [
        (0.00, 50,   "no-obs baseline"),
        (0.05, 50,   "a=0.05 N=50"),
        (0.10, 50,   "a=0.10 N=50"),
        (0.10, 300,  "a=0.10 N=300"),
        (0.10, 2000, "a=0.10 N=2000"),
        (0.20, 50,   "a=0.20 N=50"),
        (0.30, 50,   "a=0.30 N=50"),
        (0.50, 50,   "a=0.50 N=50"),
    ]

    header = f"  {'Strategy':<22}"
    for round_id in sorted(rounds.keys()):
        short = round_id[:8]
        header += f" {short:>10}"
    header += f" {'AVG':>10}"
    print(header)
    print("  " + "-" * (22 + 11 * (len(rounds) + 1)))

    for ci, (alpha, n_hist, label) in enumerate(combos):
        print(f"  [{ci+1}/{len(combos)}] Running {label}...", end="", flush=True)
        round_avgs = {}

        for round_id, round_files in sorted(rounds.items()):
            loo_matrix = build_matrix_excluding_round(files, round_id, history_dir)
            seed_scores = []

            for fname in round_files:
                with open(os.path.join(history_dir, fname)) as f:
                    data = json.load(f)
                gt = data.get("ground_truth")
                igrid = data.get("initial_grid")
                if not gt or not igrid:
                    continue
                H, W = len(igrid), len(igrid[0])

                if alpha == 0.0:
                    # No observations - just use matrix
                    pred = build_prediction_sampled(igrid, gt, loo_matrix, set(), 0.0, {})
                    seed_scores.append(score_tensor(pred, gt))
                else:
                    # Monte Carlo: run N_MC trials with different random observations
                    trial_scores = []
                    for trial in range(N_MC):
                        # Sample observations for ALL observable cells
                        all_obs_samples = {}
                        for y in range(H):
                            for x in range(W):
                                all_obs_samples[(y, x)] = sample_observation(gt[y][x])

                        # Phase 1: fixed tiles
                        obs_p1 = set()
                        for ax, ay in PHASE1_ANCHORS:
                            obs_p1 |= tile_cells(ax, ay)

                        # Calibrate with phase 1
                        cal_matrix = simulate_calibration(igrid, obs_p1, all_obs_samples, loo_matrix, n_hist)

                        # Intermediate prediction for entropy targeting
                        pred_p1 = build_prediction_sampled(igrid, gt, cal_matrix, obs_p1, alpha, all_obs_samples)

                        # Phase 2: entropy-guided
                        phase2 = entropy_guided_tiles(pred_p1, obs_p1, n_tiles=5)
                        obs_all = set(obs_p1)
                        for ax, ay in phase2:
                            obs_all |= tile_cells(ax, ay)

                        # Final calibration + prediction
                        cal_matrix2 = simulate_calibration(igrid, obs_all, all_obs_samples, loo_matrix, n_hist)
                        pred_final = build_prediction_sampled(igrid, gt, cal_matrix2, obs_all, alpha, all_obs_samples)
                        trial_scores.append(score_tensor(pred_final, gt))

                    seed_scores.append(sum(trial_scores) / len(trial_scores))

            round_avgs[round_id] = sum(seed_scores) / len(seed_scores)

        overall_avg = sum(round_avgs.values()) / len(round_avgs)

        print(f" done (avg={overall_avg:.2f})")
        line = f"  {label:<22}"
        for round_id in sorted(rounds.keys()):
            line += f" {round_avgs[round_id]:10.2f}"
        line += f" {overall_avg:10.2f}"
        print(line)

    print()
    print("  Note: scores are Monte Carlo averages over stochastic observations.")
    print("  Real API will give ONE random sample per cell, so actual results will vary.")


if __name__ == "__main__":
    main()
