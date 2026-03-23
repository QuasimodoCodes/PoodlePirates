"""
alpha_sweep.py — Comprehensive test: sweep a from 0.0 to 0.50 across all historical rounds.

Also tests the effect of round calibration by simulating it per-round
(leave-one-round-out: build matrix from other rounds, calibrate with this round's obs).

Strategies tested for each a:
  - No-obs: pure transition matrix, no observations at all
  - 10-tile (two-phase): 5 fixed + 5 entropy-guided tiles, WITH round calibration
  - 10-tile (no-cal):    same tiles but WITHOUT round calibration (historical matrix only)

Run from victor/ folder:
    python -m scripts.alpha_sweep
"""

import os, sys, json, math
from collections import defaultdict
from typing import List, Dict, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import PHASE1_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N_CLASSES = config.NUM_TERRAIN_CLASSES
FLOOR_DYNAMIC = 0.01
FLOOR_STATIC = 1e-5

# -- Scoring ----------------------------------------------------------

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


# -- Tile helpers -----------------------------------------------------

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


# -- Prediction builder -----------------------------------------------

def build_prediction(igrid, ground_truth, matrix, observed_cells: Set[Tuple[int,int]], alpha: float):
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
                    obs_class = max(range(N_CLASSES), key=lambda i: ground_truth[y][x][i])
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


# -- Simulated round calibration -------------------------------------

N_HIST = 2000

def simulate_calibration(igrid, ground_truth, observed_cells, historical_matrix):
    """Simulate round calibration using observed cells from ground truth."""
    round_counts = defaultdict(list)
    for y, x in observed_cells:
        code = igrid[y][x]
        if code in STATIC_CODES:
            continue
        obs_class = max(range(N_CLASSES), key=lambda i: ground_truth[y][x][i])
        round_counts[code].append(obs_class)

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
        total = n_round + N_HIST
        blended[code] = [(n_round * round_freq[i] + N_HIST * hist[i]) / total for i in range(N_CLASSES)]
    return blended


# -- Leave-one-round-out matrix builder -------------------------------

def build_matrix_excluding_round(all_files, exclude_round_id, history_dir):
    """Build transition matrix from all rounds EXCEPT the target one."""
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


# -- Main -------------------------------------------------------------

def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))

    if not files:
        print("No analysis files found.")
        return

    # Group by round
    rounds = defaultdict(list)
    for f in files:
        round_id = f.split("_seed")[0]
        rounds[round_id].append(f)

    alphas = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]

    print(f"Alpha sweep across {len(files)} files ({len(rounds)} rounds)")
    print(f"Leave-one-round-out: matrix built from OTHER rounds, calibrated with target's obs")
    print(f"N_HIST = {N_HIST}")
    print()

    # Results: alpha → list of scores
    results_noobs = {a: [] for a in alphas}
    results_twophase = {a: [] for a in alphas}
    results_twophase_nocal = {a: [] for a in alphas}

    # Per-round results for analysis
    round_results = {}

    for round_id, round_files in sorted(rounds.items()):
        # Build matrix excluding this round (leave-one-out)
        loo_matrix = build_matrix_excluding_round(files, round_id, history_dir)

        round_scores = {a: {"noobs": [], "twophase": [], "nocal": []} for a in alphas}

        for fname in round_files:
            with open(os.path.join(history_dir, fname)) as f:
                data = json.load(f)
            gt = data.get("ground_truth")
            igrid = data.get("initial_grid")
            if not gt or not igrid:
                continue

            for alpha in alphas:
                # Strategy 1: No observations at all
                pred_no = build_prediction(igrid, gt, loo_matrix, set(), 0.0)
                score_no = score_tensor(pred_no, gt)
                results_noobs[alpha].append(score_no)
                round_scores[alpha]["noobs"].append(score_no)

                # Strategy 2: Two-phase with calibration
                obs_p1 = set()
                for ax, ay in PHASE1_ANCHORS:
                    obs_p1 |= tile_cells(ax, ay)

                # Calibrate using phase 1 observations
                cal_matrix = simulate_calibration(igrid, gt, obs_p1, loo_matrix)

                pred_p1 = build_prediction(igrid, gt, cal_matrix, obs_p1, alpha)
                phase2_anchors = entropy_guided_tiles(pred_p1, obs_p1, n_tiles=5)
                obs_all = set(obs_p1)
                for ax, ay in phase2_anchors:
                    obs_all |= tile_cells(ax, ay)

                # Re-calibrate with all observations
                cal_matrix2 = simulate_calibration(igrid, gt, obs_all, loo_matrix)
                pred_final = build_prediction(igrid, gt, cal_matrix2, obs_all, alpha)
                score_tp = score_tensor(pred_final, gt)
                results_twophase[alpha].append(score_tp)
                round_scores[alpha]["twophase"].append(score_tp)

                # Strategy 3: Two-phase WITHOUT calibration (raw loo_matrix)
                pred_nocal = build_prediction(igrid, gt, loo_matrix, obs_all, alpha)
                score_nc = score_tensor(pred_nocal, gt)
                results_twophase_nocal[alpha].append(score_nc)
                round_scores[alpha]["nocal"].append(score_nc)

        round_results[round_id] = round_scores

    # -- Print summary table ------------------------------------------
    print(f"\n{'='*80}")
    print(f"  OVERALL AVERAGES (across {len(files)} seeds)")
    print(f"{'='*80}")
    print(f"  {'alpha':>6}  {'No-Obs':>8}  {'2Phase+Cal':>10}  {'2Phase-NoCal':>12}  {'Cal-NoObs':>10}  {'Cal-NoCal':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")

    best_alpha = None
    best_score = -1

    for alpha in alphas:
        avg_no = sum(results_noobs[alpha]) / len(results_noobs[alpha])
        avg_tp = sum(results_twophase[alpha]) / len(results_twophase[alpha])
        avg_nc = sum(results_twophase_nocal[alpha]) / len(results_twophase_nocal[alpha])

        delta_cal = avg_tp - avg_no
        delta_nocal = avg_tp - avg_nc

        marker = ""
        if avg_tp > best_score:
            best_score = avg_tp
            best_alpha = alpha

        print(f"  {alpha:6.2f}  {avg_no:8.2f}  {avg_tp:10.2f}  {avg_nc:12.2f}  {delta_cal:+10.2f}  {delta_nocal:+10.2f}")

    print(f"\n  * Best a = {best_alpha} (two-phase+cal avg = {best_score:.2f})")

    # -- Per-round breakdown at best a --------------------------------
    print(f"\n{'='*80}")
    print(f"  PER-ROUND BREAKDOWN at a={best_alpha}")
    print(f"{'='*80}")
    print(f"  {'Round':<40} {'No-Obs':>8}  {'2Ph+Cal':>8}  {'2Ph-NoCal':>10}  {'D(Cal-No)':>10}")
    print(f"  {'-'*40} {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")

    for round_id in sorted(round_results.keys()):
        scores = round_results[round_id][best_alpha]
        avg_no = sum(scores["noobs"]) / len(scores["noobs"])
        avg_tp = sum(scores["twophase"]) / len(scores["twophase"])
        avg_nc = sum(scores["nocal"]) / len(scores["nocal"])
        delta = avg_tp - avg_no
        print(f"  {round_id[:38]:<40} {avg_no:8.2f}  {avg_tp:8.2f}  {avg_nc:10.2f}  {delta:+10.2f}")

    # -- Also show a=0 vs best a per round ---------------------------
    if best_alpha != 0.0:
        print(f"\n{'='*80}")
        print(f"  PER-ROUND: a=0 (no-obs) vs a={best_alpha} (best)")
        print(f"{'='*80}")
        print(f"  {'Round':<40} {'a=0':>8}  {'a={:.2f}'.format(best_alpha):>8}  {'D':>8}")
        print(f"  {'-'*40} {'-'*8}  {'-'*8}  {'-'*8}")

        for round_id in sorted(round_results.keys()):
            s0 = round_results[round_id][0.0]
            sb = round_results[round_id][best_alpha]
            avg_0 = sum(s0["noobs"]) / len(s0["noobs"])
            avg_b = sum(sb["twophase"]) / len(sb["twophase"])
            print(f"  {round_id[:38]:<40} {avg_0:8.2f}  {avg_b:8.2f}  {avg_b-avg_0:+8.2f}")


def nhist_sweep():
    """Test different N_HIST values at alpha=0.50."""
    global N_HIST
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files:
        rounds[f.split("_seed")[0]].append(f)

    nhist_values = [50, 100, 300, 500, 1000, 2000, 5000]
    alpha = 0.10

    print(f"\n{'='*80}")
    print(f"  N_HIST SWEEP (alpha={alpha}, two-phase + calibration)")
    print(f"{'='*80}")
    print(f"  {'N_HIST':>8}  {'Avg Score':>10}")
    print(f"  {'-'*8}  {'-'*10}")

    for nh in nhist_values:
        N_HIST = nh
        scores = []
        for round_id, round_files in rounds.items():
            loo_matrix = build_matrix_excluding_round(files, round_id, history_dir)
            for fname in round_files:
                with open(os.path.join(history_dir, fname)) as f:
                    data = json.load(f)
                gt = data.get("ground_truth")
                igrid = data.get("initial_grid")
                if not gt or not igrid:
                    continue
                obs_p1 = set()
                for ax, ay in PHASE1_ANCHORS:
                    obs_p1 |= tile_cells(ax, ay)
                cal_matrix = simulate_calibration(igrid, gt, obs_p1, loo_matrix)
                pred_p1 = build_prediction(igrid, gt, cal_matrix, obs_p1, alpha)
                phase2 = entropy_guided_tiles(pred_p1, obs_p1, n_tiles=5)
                obs_all = set(obs_p1)
                for ax, ay in phase2:
                    obs_all |= tile_cells(ax, ay)
                cal_matrix2 = simulate_calibration(igrid, gt, obs_all, loo_matrix)
                pred_final = build_prediction(igrid, gt, cal_matrix2, obs_all, alpha)
                scores.append(score_tensor(pred_final, gt))
        avg = sum(scores) / len(scores)
        print(f"  {nh:>8}  {avg:10.2f}")

    N_HIST = 2000  # reset


if __name__ == "__main__":
    main()
    nhist_sweep()
