"""
feature_sweep.py - Test multiple improvement ideas on historical data.

Tests:
  1. Floor value sweep (FLOOR_DYNAMIC)
  2. Phase budget split (Phase1/Phase2)
  3. Per-terrain-code alpha
  4. Neighbor-aware predictions
  5. Overlapping tile strategy

Uses realistic Monte Carlo sampling (not argmax).

Run from victor/ folder:
    python -m scripts.feature_sweep
"""

import os, sys, json, math, random
from collections import defaultdict
from typing import List, Dict, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import PHASE1_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N_CLASSES = config.NUM_TERRAIN_CLASSES
FLOOR_STATIC = 1e-5
N_MC = 5  # trials per seed (lower for speed, still meaningful)

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

# -- Helpers ------------------------------------------------------------------

def tile_cells(ax, ay) -> Set[Tuple[int,int]]:
    return set(_covered_cells(ax, ay))

def sample_observation(gt_dist):
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(gt_dist):
        cumsum += p
        if r < cumsum:
            return i
    return len(gt_dist) - 1

def entropy_guided_tiles(tensor, observed, n_tiles=5):
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

def build_matrix_excluding_round(all_files, exclude_round_id, history_dir):
    counts = defaultdict(lambda: [0.0] * N_CLASSES)
    for fname in all_files:
        if fname.split("_seed")[0] == exclude_round_id:
            continue
        with open(os.path.join(history_dir, fname)) as f:
            data = json.load(f)
        gt, igrid = data.get("ground_truth"), data.get("initial_grid")
        if not gt or not igrid:
            continue
        for y in range(len(igrid)):
            for x in range(len(igrid[0])):
                code = igrid[y][x]
                if code not in STATIC_CODES:
                    counts[code][max(range(N_CLASSES), key=lambda i: gt[y][x][i])] += 1.0
    matrix = {}
    for code, freq in counts.items():
        total = sum(freq)
        matrix[code] = [f/total for f in freq] if total > 0 else [1.0/N_CLASSES]*N_CLASSES
    for code in [0,1,2,3,4,5,10,11]:
        if code not in matrix:
            matrix[code] = [1.0/N_CLASSES]*N_CLASSES
    return matrix

def simulate_calibration(igrid, observed_cells, obs_samples, hist_matrix, n_hist):
    round_counts = defaultdict(list)
    for y, x in observed_cells:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            round_counts[code].append(obs_samples[(y, x)])
    blended = {}
    for code in [0,1,2,3,4,5,10,11]:
        hist = hist_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES:
            blended[code] = hist[:]
            continue
        obs_list = round_counts.get(code, [])
        n_round = len(obs_list)
        if n_round == 0:
            blended[code] = hist[:]
            continue
        round_freq = [0.0]*N_CLASSES
        for cls in obs_list:
            round_freq[cls] += 1.0/n_round
        total = n_round + n_hist
        blended[code] = [(n_round*round_freq[i] + n_hist*hist[i])/total for i in range(N_CLASSES)]
    return blended


# =============================================================================
# PREDICTION BUILDERS (different strategies)
# =============================================================================

def build_pred_standard(igrid, gt, matrix, obs_cells, obs_samples, alpha, floor_dyn):
    """Standard: single alpha, single floor."""
    H, W = len(igrid), len(igrid[0])
    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC]*N_CLASSES
                dist[pc] = 1.0 - FLOOR_STATIC*(N_CLASSES-1)
            else:
                prior = matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                if (y,x) in obs_cells and alpha > 0:
                    oh = [0.0]*N_CLASSES
                    oh[obs_samples[(y,x)]] = 1.0
                    dist = [(1-alpha)*prior[i] + alpha*oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, floor_dyn) for v in dist]
            total = sum(dist)
            row.append([v/total for v in dist])
        tensor.append(row)
    return tensor


def build_pred_per_code_alpha(igrid, gt, matrix, obs_cells, obs_samples, alpha_map, floor_dyn):
    """Different alpha per initial terrain code."""
    H, W = len(igrid), len(igrid[0])
    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC]*N_CLASSES
                dist[pc] = 1.0 - FLOOR_STATIC*(N_CLASSES-1)
            else:
                prior = matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                a = alpha_map.get(code, 0.05)
                if (y,x) in obs_cells and a > 0:
                    oh = [0.0]*N_CLASSES
                    oh[obs_samples[(y,x)]] = 1.0
                    dist = [(1-a)*prior[i] + a*oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, floor_dyn) for v in dist]
            total = sum(dist)
            row.append([v/total for v in dist])
        tensor.append(row)
    return tensor


def build_pred_neighbor(igrid, gt, matrix, obs_cells, obs_samples, alpha, floor_dyn):
    """Neighbor-aware: adjust prior based on surrounding settlements."""
    H, W = len(igrid), len(igrid[0])

    # Count settlement neighbors for each cell
    def count_settlement_neighbors(y, x):
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y+dy, x+dx
                if 0 <= ny < H and 0 <= nx < W and igrid[ny][nx] == 1:
                    count += 1
        return count

    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC]*N_CLASSES
                dist[pc] = 1.0 - FLOOR_STATIC*(N_CLASSES-1)
            else:
                prior = matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                n_sett = count_settlement_neighbors(y, x)

                # Boost settlement probability if many settlement neighbors
                if n_sett >= 2 and code in (11, 4):  # Plains or Forest near settlements
                    boost = min(0.10, n_sett * 0.03)
                    prior[1] += boost  # Settlement class
                    prior[0] -= boost * 0.5  # Take from Empty
                    prior[4] -= boost * 0.5  # Take from Forest
                    prior = [max(v, 0.001) for v in prior]
                    t = sum(prior)
                    prior = [v/t for v in prior]

                # Settlements near other settlements survive more
                if code == 1 and n_sett >= 1:
                    boost = min(0.08, n_sett * 0.03)
                    prior[1] += boost  # More likely to stay Settlement
                    prior[0] -= boost  # Less likely to become Empty
                    prior = [max(v, 0.001) for v in prior]
                    t = sum(prior)
                    prior = [v/t for v in prior]

                if (y,x) in obs_cells and alpha > 0:
                    oh = [0.0]*N_CLASSES
                    oh[obs_samples[(y,x)]] = 1.0
                    dist = [(1-alpha)*prior[i] + alpha*oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, floor_dyn) for v in dist]
            total = sum(dist)
            row.append([v/total for v in dist])
        tensor.append(row)
    return tensor


# =============================================================================
# FULL PIPELINE RUNNER
# =============================================================================

def run_pipeline(igrid, gt, loo_matrix, n_hist, alpha, floor_dyn,
                 n_phase1_tiles=5, n_phase2_tiles=5,
                 pred_fn=None, pred_kwargs=None):
    """Run full two-phase pipeline, return score."""
    H, W = len(igrid), len(igrid[0])
    if pred_fn is None:
        pred_fn = build_pred_standard
    if pred_kwargs is None:
        pred_kwargs = {}

    # Sample all observations once
    obs_samples = {}
    for y in range(H):
        for x in range(W):
            obs_samples[(y,x)] = sample_observation(gt[y][x])

    # Phase 1
    obs_p1 = set()
    for ax, ay in PHASE1_ANCHORS[:n_phase1_tiles]:
        obs_p1 |= tile_cells(ax, ay)

    # Calibrate
    cal = simulate_calibration(igrid, obs_p1, obs_samples, loo_matrix, n_hist)

    # Intermediate prediction for entropy targeting
    pred_p1 = build_pred_standard(igrid, gt, cal, obs_p1, obs_samples, alpha, floor_dyn)

    # Phase 2
    p2_anchors = entropy_guided_tiles(pred_p1, obs_p1, n_tiles=n_phase2_tiles)
    obs_all = set(obs_p1)
    for ax, ay in p2_anchors:
        obs_all |= tile_cells(ax, ay)

    # Final calibration
    cal2 = simulate_calibration(igrid, obs_all, obs_samples, loo_matrix, n_hist)

    # Final prediction
    pred = pred_fn(igrid, gt, cal2, obs_all, obs_samples, alpha, floor_dyn, **pred_kwargs)
    return score_tensor(pred, gt)


def run_pipeline_per_code(igrid, gt, loo_matrix, n_hist, alpha_map, floor_dyn,
                          n_phase1_tiles=5, n_phase2_tiles=5):
    """Pipeline with per-code alpha."""
    H, W = len(igrid), len(igrid[0])
    obs_samples = {}
    for y in range(H):
        for x in range(W):
            obs_samples[(y,x)] = sample_observation(gt[y][x])

    obs_p1 = set()
    for ax, ay in PHASE1_ANCHORS[:n_phase1_tiles]:
        obs_p1 |= tile_cells(ax, ay)
    cal = simulate_calibration(igrid, obs_p1, obs_samples, loo_matrix, 50)
    pred_p1 = build_pred_standard(igrid, gt, cal, obs_p1, obs_samples, 0.05, floor_dyn)
    p2_anchors = entropy_guided_tiles(pred_p1, obs_p1, n_tiles=n_phase2_tiles)
    obs_all = set(obs_p1)
    for ax, ay in p2_anchors:
        obs_all |= tile_cells(ax, ay)
    cal2 = simulate_calibration(igrid, obs_all, obs_samples, loo_matrix, 50)
    pred = build_pred_per_code_alpha(igrid, gt, cal2, obs_all, obs_samples, alpha_map, floor_dyn)
    return score_tensor(pred, gt)


# =============================================================================
# MAIN
# =============================================================================

def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files:
        rounds[f.split("_seed")[0]].append(f)

    print(f"Feature sweep: {len(files)} files, {len(rounds)} rounds, {N_MC} MC trials")
    print()

    # Pre-load all data
    all_data = {}
    loo_matrices = {}
    for round_id, round_files in rounds.items():
        loo_matrices[round_id] = build_matrix_excluding_round(files, round_id, history_dir)
        for fname in round_files:
            with open(os.path.join(history_dir, fname)) as f:
                all_data[fname] = json.load(f)

    def eval_strategy(label, run_fn):
        """Evaluate a strategy across all rounds with MC sampling."""
        print(f"  [{label}]...", end="", flush=True)
        all_scores = []
        for round_id, round_files in sorted(rounds.items()):
            for fname in round_files:
                d = all_data[fname]
                gt, igrid = d.get("ground_truth"), d.get("initial_grid")
                if not gt or not igrid:
                    continue
                trial_scores = []
                for _ in range(N_MC):
                    trial_scores.append(run_fn(igrid, gt, loo_matrices[round_id]))
                all_scores.append(sum(trial_scores)/len(trial_scores))
        avg = sum(all_scores)/len(all_scores)
        print(f" avg={avg:.2f}")
        return avg

    results = {}

    # =========================================================================
    # TEST 1: Floor sweep
    # =========================================================================
    print("=" * 60)
    print("  TEST 1: Floor Value Sweep (alpha=0.05, N_HIST=50)")
    print("=" * 60)
    for floor in [0.001, 0.005, 0.01, 0.02, 0.05]:
        label = f"floor={floor}"
        score = eval_strategy(label,
            lambda ig, gt, m, f=floor: run_pipeline(ig, gt, m, 50, 0.05, f))
        results[label] = score

    # =========================================================================
    # TEST 2: Phase budget split (total 10 tiles per seed)
    # =========================================================================
    print()
    print("=" * 60)
    print("  TEST 2: Phase Budget Split (alpha=0.05, N_HIST=50, floor=0.01)")
    print("=" * 60)

    # Need extra phase1 anchor sets for different sizes
    EXTRA_ANCHORS = [
        (0,0), (25,0), (0,25), (25,25), (12,12),   # standard 5
        (15,0), (0,15), (15,15), (25,15), (15,25),  # extra 5
    ]

    for p1, p2 in [(3, 7), (5, 5), (7, 3), (4, 6), (6, 4)]:
        label = f"split={p1}/{p2}"
        score = eval_strategy(label,
            lambda ig, gt, m, p1=p1, p2=p2: run_pipeline(
                ig, gt, m, 50, 0.05, 0.01,
                n_phase1_tiles=p1, n_phase2_tiles=p2))
        results[label] = score

    # =========================================================================
    # TEST 3: Per-terrain-code alpha
    # =========================================================================
    print()
    print("=" * 60)
    print("  TEST 3: Per-Terrain-Code Alpha (N_HIST=50, floor=0.01)")
    print("=" * 60)

    # Baseline: uniform alpha
    score = eval_strategy("uniform a=0.05",
        lambda ig, gt, m: run_pipeline(ig, gt, m, 50, 0.05, 0.01))
    results["uniform a=0.05"] = score

    # Settlement high, Forest low
    alpha_map_1 = {1: 0.15, 2: 0.10, 4: 0.02, 11: 0.05, 0: 0.05, 3: 0.05}
    score = eval_strategy("sett=0.15 forest=0.02",
        lambda ig, gt, m: run_pipeline_per_code(ig, gt, m, 50, alpha_map_1, 0.01))
    results["sett=0.15 forest=0.02"] = score

    # Settlement very high, Forest zero
    alpha_map_2 = {1: 0.20, 2: 0.15, 4: 0.00, 11: 0.03, 0: 0.05, 3: 0.05}
    score = eval_strategy("sett=0.20 forest=0.00",
        lambda ig, gt, m: run_pipeline_per_code(ig, gt, m, 50, alpha_map_2, 0.01))
    results["sett=0.20 forest=0.00"] = score

    # All codes slightly higher
    alpha_map_3 = {1: 0.10, 2: 0.10, 4: 0.05, 11: 0.05, 0: 0.05, 3: 0.05}
    score = eval_strategy("all=0.05-0.10",
        lambda ig, gt, m: run_pipeline_per_code(ig, gt, m, 50, alpha_map_3, 0.01))
    results["all=0.05-0.10"] = score

    # =========================================================================
    # TEST 4: Neighbor-aware predictions
    # =========================================================================
    print()
    print("=" * 60)
    print("  TEST 4: Neighbor-Aware Predictions (alpha=0.05, N_HIST=50)")
    print("=" * 60)

    score = eval_strategy("no-neighbor (baseline)",
        lambda ig, gt, m: run_pipeline(ig, gt, m, 50, 0.05, 0.01))
    results["no-neighbor"] = score

    score = eval_strategy("neighbor-aware",
        lambda ig, gt, m: run_pipeline(ig, gt, m, 50, 0.05, 0.01,
            pred_fn=build_pred_neighbor))
    results["neighbor-aware"] = score

    # =========================================================================
    # TEST 5: Overlapping tiles (query same area twice for noise reduction)
    # =========================================================================
    print()
    print("=" * 60)
    print("  TEST 5: Overlapping Tiles Strategy")
    print("=" * 60)

    # Strategy: use all 10 tiles on same 5 positions (2 observations per cell)
    # This requires modifying the pipeline to average multiple observations

    def run_overlap_pipeline(igrid, gt, loo_matrix):
        """Query same 5 tiles twice, average observations."""
        H, W = len(igrid), len(igrid[0])

        # Sample TWO observations per cell (simulating querying same tile twice)
        obs_samples_1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
        obs_samples_2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

        obs_cells = set()
        for ax, ay in PHASE1_ANCHORS:
            obs_cells |= tile_cells(ax, ay)

        # Calibrate with double observations (more data for calibration)
        all_obs_for_cal = defaultdict(list)
        for y, x in obs_cells:
            code = igrid[y][x]
            if code not in STATIC_CODES:
                all_obs_for_cal[code].append(obs_samples_1[(y,x)])
                all_obs_for_cal[code].append(obs_samples_2[(y,x)])

        # Manual calibration with doubled data
        blended = {}
        n_hist = 50
        for code in [0,1,2,3,4,5,10,11]:
            hist = loo_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
            if code in STATIC_CODES:
                blended[code] = hist[:]
                continue
            obs_list = all_obs_for_cal.get(code, [])
            n_round = len(obs_list)
            if n_round == 0:
                blended[code] = hist[:]
                continue
            rf = [0.0]*N_CLASSES
            for c in obs_list:
                rf[c] += 1.0/n_round
            total = n_round + n_hist
            blended[code] = [(n_round*rf[i] + n_hist*hist[i])/total for i in range(N_CLASSES)]

        # Build prediction averaging two observations
        tensor = []
        alpha = 0.05
        floor_dyn = 0.01
        for y in range(H):
            row = []
            for x in range(W):
                code = igrid[y][x]
                if code in STATIC_CODES:
                    pc = CODE_TO_CLASS[code]
                    dist = [FLOOR_STATIC]*N_CLASSES
                    dist[pc] = 1.0 - FLOOR_STATIC*(N_CLASSES-1)
                else:
                    prior = blended.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                    if (y,x) in obs_cells:
                        # Average two observations into a soft one-hot
                        avg_oh = [0.0]*N_CLASSES
                        avg_oh[obs_samples_1[(y,x)]] += 0.5
                        avg_oh[obs_samples_2[(y,x)]] += 0.5
                        dist = [(1-alpha)*prior[i] + alpha*avg_oh[i] for i in range(N_CLASSES)]
                    else:
                        dist = prior[:]
                    dist = [max(v, floor_dyn) for v in dist]
                total = sum(dist)
                row.append([v/total for v in dist])
            tensor.append(row)
        return score_tensor(tensor, gt)

    score = eval_strategy("spread (5+5 tiles, baseline)",
        lambda ig, gt, m: run_pipeline(ig, gt, m, 50, 0.05, 0.01))
    results["spread-baseline"] = score

    score = eval_strategy("overlap (5 tiles x2 obs each)",
        lambda ig, gt, m: run_overlap_pipeline(ig, gt, m))
    results["overlap-2x"] = score

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 60)
    print("  SUMMARY — All Results Ranked")
    print("=" * 60)
    ranked = sorted(results.items(), key=lambda x: -x[1])
    for i, (label, score) in enumerate(ranked):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1:2d}. {label:<35} {score:6.2f}{marker}")

    print()
    best_label, best_score = ranked[0]
    baseline = results.get("uniform a=0.05", results.get("no-neighbor", 0))
    print(f"  Best: {best_label} ({best_score:.2f})")
    print(f"  vs baseline a=0.05: {best_score - baseline:+.2f}")


if __name__ == "__main__":
    main()
