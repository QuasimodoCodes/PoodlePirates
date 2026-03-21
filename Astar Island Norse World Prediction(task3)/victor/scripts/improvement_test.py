"""
improvement_test.py - Test improvement ideas on top of Strategy 3 (settlement cluster + spread).

Ideas tested:
  A. Baseline — settlement cluster P1 + spread P2 (current pipeline)
  B. Regime detection — cluster historical rounds, pick best-matching regime matrix
  C. Neighbor-aware prior — settlements near other settlements get adjusted prior
  D. Per-terrain floor — lower floor for stable cells, higher for volatile
  E. Per-terrain alpha — higher alpha for settlements, lower for forest
  F. Spatial smoothing — unobserved cells borrow from nearby observed cells
  G. Combined best — stack improvements that help

Run from victor/ folder:
    python -m scripts.improvement_test
"""

import os, sys, json, math, random
from collections import defaultdict
from typing import List, Dict, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N_CLASSES = config.NUM_TERRAIN_CLASSES
FLOOR_STATIC = 1e-5
FLOOR_DYN = 0.005
N_MC = 5
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

def tile_cells(ax, ay):
    return set(_covered_cells(ax, ay))

def sample_observation(gt_dist):
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(gt_dist):
        cumsum += p
        if r < cumsum:
            return i
    return len(gt_dist) - 1

def get_all_tile_anchors():
    return [(ax, ay)
            for ay in range(config.MAP_HEIGHT - TILE_H + 1)
            for ax in range(config.MAP_WIDTH - TILE_W + 1)]

ALL_ANCHORS = get_all_tile_anchors()
ANCHOR_CELLS = {a: set(_covered_cells(*a)) for a in ALL_ANCHORS}


def build_matrix_excluding_round(all_files, exclude_round_id, history_dir, all_data):
    accum = defaultdict(list)
    for fname in all_files:
        if fname.split("_seed")[0] == exclude_round_id:
            continue
        data = all_data[fname]
        gt, igrid = data.get("ground_truth"), data.get("initial_grid")
        if not gt or not igrid:
            continue
        for y in range(len(igrid)):
            for x in range(len(igrid[0])):
                code = igrid[y][x]
                if code not in STATIC_CODES:
                    accum[code].append(gt[y][x])
    matrix = {}
    for code, samples in accum.items():
        n = len(samples)
        matrix[code] = [sum(s[i] for s in samples)/n for i in range(N_CLASSES)]
    for code in [0,1,2,3,4,5,10,11]:
        if code not in matrix:
            matrix[code] = [1.0/N_CLASSES]*N_CLASSES
    return matrix


def build_regime_matrices(all_files, exclude_round_id, all_data):
    """Build per-round matrices for regime detection."""
    round_matrices = {}  # round_id -> matrix
    round_ids = set()
    for fname in all_files:
        rid = fname.split("_seed")[0]
        if rid == exclude_round_id:
            continue
        round_ids.add(rid)

    for rid in round_ids:
        accum = defaultdict(list)
        for fname in all_files:
            if fname.split("_seed")[0] != rid:
                continue
            data = all_data[fname]
            gt, igrid = data.get("ground_truth"), data.get("initial_grid")
            if not gt or not igrid:
                continue
            for y in range(len(igrid)):
                for x in range(len(igrid[0])):
                    code = igrid[y][x]
                    if code not in STATIC_CODES:
                        accum[code].append(gt[y][x])
        matrix = {}
        for code, samples in accum.items():
            n = len(samples)
            matrix[code] = [sum(s[i] for s in samples)/n for i in range(N_CLASSES)]
        for code in [0,1,2,3,4,5,10,11]:
            if code not in matrix:
                matrix[code] = [1.0/N_CLASSES]*N_CLASSES
        round_matrices[rid] = matrix

    return round_matrices


def simulate_calibration(igrid, obs_codes_by_terrain, hist_matrix, n_hist=50):
    blended = {}
    for code in [0,1,2,3,4,5,10,11]:
        hist = hist_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES:
            blended[code] = hist[:]; continue
        ol = obs_codes_by_terrain.get(code, [])
        nr = len(ol)
        if nr == 0:
            blended[code] = hist[:]; continue
        rf = [0.0]*N_CLASSES
        for c in ol: rf[c] += 1.0/nr
        t = nr + n_hist
        blended[code] = [(nr*rf[i]+n_hist*hist[i])/t for i in range(N_CLASSES)]
    return blended


def select_settlement_cluster_tiles(igrid, n_tiles=5):
    H, W = len(igrid), len(igrid[0])
    settlements = set()
    for y in range(H):
        for x in range(W):
            if igrid[y][x] == 1:
                settlements.add((y, x))
    if not settlements:
        return SPREAD_ANCHORS[:n_tiles]

    covered_setts = set()
    selected = []
    for _ in range(n_tiles):
        best, best_count = None, -1
        for a in ALL_ANCHORS:
            count = len((ANCHOR_CELLS[a] & settlements) - covered_setts)
            if count > best_count:
                best_count, best = count, a
        if best is None or best_count <= 0:
            break
        selected.append(best)
        covered_setts |= (ANCHOR_CELLS[best] & settlements)
    while len(selected) < n_tiles and len(selected) < len(SPREAD_ANCHORS):
        a = SPREAD_ANCHORS[len(selected)]
        if a not in selected:
            selected.append(a)
    return selected[:n_tiles]


def run_pipeline(igrid, gt, matrix, obs_p1, obs_p2, obs1, obs2, alpha=0.05, floor_dyn=FLOOR_DYN,
                 per_terrain_alpha=None, per_terrain_floor=None, spatial_smooth=False):
    """Generic pipeline: calibrate + build tensor."""
    H, W = len(igrid), len(igrid[0])

    # Calibration from both phases
    all_obs_codes = defaultdict(list)
    for y, x in obs_p1:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            all_obs_codes[code].append(obs1[(y, x)])
    for y, x in obs_p2:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            all_obs_codes[code].append(obs2[(y, x)])

    blended = simulate_calibration(igrid, all_obs_codes, matrix, n_hist=50)

    obs_both = obs_p1 & obs_p2
    obs_all = obs_p1 | obs_p2

    # Spatial smoothing: build neighbor observation map
    neighbor_obs = {}
    if spatial_smooth:
        for y in range(H):
            for x in range(W):
                if (y, x) in obs_all:
                    continue
                if igrid[y][x] in STATIC_CODES:
                    continue
                # Average observations from neighbors within distance 2
                neighbor_classes = []
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and (ny, nx) in obs_all:
                            if igrid[ny][nx] == igrid[y][x]:  # same terrain type
                                if (ny, nx) in obs_both:
                                    neighbor_classes.append(obs1[(ny, nx)])
                                    neighbor_classes.append(obs2[(ny, nx)])
                                elif (ny, nx) in obs_p1:
                                    neighbor_classes.append(obs1[(ny, nx)])
                                elif (ny, nx) in obs_p2:
                                    neighbor_classes.append(obs2[(ny, nx)])
                if neighbor_classes:
                    neighbor_obs[(y, x)] = neighbor_classes

    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC] * N_CLASSES
                dist[pc] = 1.0 - FLOOR_STATIC * (N_CLASSES - 1)
            else:
                prior = blended.get(code, [1.0 / N_CLASSES] * N_CLASSES)[:]
                a = per_terrain_alpha.get(code, alpha) if per_terrain_alpha else alpha
                fl = per_terrain_floor.get(code, floor_dyn) if per_terrain_floor else floor_dyn

                if (y, x) in obs_both:
                    oh = [0.0] * N_CLASSES
                    oh[obs1[(y, x)]] += 0.5
                    oh[obs2[(y, x)]] += 0.5
                    dist = [(1 - a) * prior[i] + a * oh[i] for i in range(N_CLASSES)]
                elif (y, x) in obs_p1:
                    oh = [0.0] * N_CLASSES
                    oh[obs1[(y, x)]] = 1.0
                    dist = [(1 - a) * prior[i] + a * oh[i] for i in range(N_CLASSES)]
                elif (y, x) in obs_p2:
                    oh = [0.0] * N_CLASSES
                    oh[obs2[(y, x)]] = 1.0
                    dist = [(1 - a) * prior[i] + a * oh[i] for i in range(N_CLASSES)]
                elif spatial_smooth and (y, x) in neighbor_obs:
                    # Weak Bayesian update from neighbors (very low alpha)
                    nc = neighbor_obs[(y, x)]
                    oh = [0.0] * N_CLASSES
                    for c in nc:
                        oh[c] += 1.0 / len(nc)
                    sa = 0.02  # very gentle
                    dist = [(1 - sa) * prior[i] + sa * oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]

                dist = [max(v, fl) for v in dist]
            total = sum(dist)
            row.append([v / total for v in dist])
        tensor.append(row)
    return tensor


# =============================================================================
# STRATEGIES
# =============================================================================

def get_obs_and_tiles(igrid, gt):
    """Common setup: sample observations, select tiles."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

    p1_tiles = select_settlement_cluster_tiles(igrid, n_tiles=5)
    obs_p1 = set()
    for ax, ay in p1_tiles:
        obs_p1 |= tile_cells(ax, ay)
    obs_p2 = set()
    for ax, ay in SPREAD_ANCHORS:
        obs_p2 |= tile_cells(ax, ay)

    return obs1, obs2, obs_p1, obs_p2


def run_baseline(igrid, gt, loo_matrix, _regime_matrices=None):
    """Strategy 3 baseline: settlement cluster P1 + spread P2."""
    obs1, obs2, obs_p1, obs_p2 = get_obs_and_tiles(igrid, gt)
    tensor = run_pipeline(igrid, gt, loo_matrix, obs_p1, obs_p2, obs1, obs2)
    return score_tensor(tensor, gt)


def run_regime_detection(igrid, gt, loo_matrix, regime_matrices=None):
    """Pick the best-matching historical round's matrix based on observations."""
    obs1, obs2, obs_p1, obs_p2 = get_obs_and_tiles(igrid, gt)

    if not regime_matrices:
        tensor = run_pipeline(igrid, gt, loo_matrix, obs_p1, obs_p2, obs1, obs2)
        return score_tensor(tensor, gt)

    # Build observation frequency from this round
    obs_freq = defaultdict(lambda: [0.0] * N_CLASSES)
    obs_count = defaultdict(int)
    for y, x in obs_p1:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            obs_freq[code][obs1[(y, x)]] += 1
            obs_count[code] += 1
    for y, x in obs_p2:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            obs_freq[code][obs2[(y, x)]] += 1
            obs_count[code] += 1
    for code in obs_freq:
        n = obs_count[code]
        if n > 0:
            obs_freq[code] = [v / n for v in obs_freq[code]]

    # Find best-matching regime by KL distance on key terrain types
    best_rid, best_dist = None, float('inf')
    for rid, rmatrix in regime_matrices.items():
        dist = 0.0
        for code in [1, 4, 11]:  # Settlement, Forest, Plains
            if obs_count.get(code, 0) < 5:
                continue
            of = obs_freq[code]
            rm = rmatrix.get(code, [1.0 / N_CLASSES] * N_CLASSES)
            for i in range(N_CLASSES):
                if of[i] > 1e-12:
                    dist += of[i] * math.log(of[i] / max(rm[i], 1e-12))
        if dist < best_dist:
            best_dist, best_rid = dist, rid

    # Blend: 50% best regime + 50% overall LOO matrix
    if best_rid:
        regime_mat = regime_matrices[best_rid]
        blended_base = {}
        for code in [0, 1, 2, 3, 4, 5, 10, 11]:
            lm = loo_matrix.get(code, [1.0 / N_CLASSES] * N_CLASSES)
            rm = regime_mat.get(code, lm)
            blended_base[code] = [0.5 * lm[i] + 0.5 * rm[i] for i in range(N_CLASSES)]
    else:
        blended_base = loo_matrix

    tensor = run_pipeline(igrid, gt, blended_base, obs_p1, obs_p2, obs1, obs2)
    return score_tensor(tensor, gt)


def _neighbor_aware_helper(igrid, gt, loo_matrix, empty_ratio=0.0):
    """Adjust prior based on neighbor terrain composition."""
    obs1, obs2, obs_p1, obs_p2 = get_obs_and_tiles(igrid, gt)
    H, W = len(igrid), len(igrid[0])

    # Count settlement neighbors for each cell
    sett_neighbor_count = {}
    for y in range(H):
        for x in range(W):
            if igrid[y][x] not in STATIC_CODES:
                count = 0
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and igrid[ny][nx] == 1:
                            count += 1
                sett_neighbor_count[(y, x)] = count

    # Calibrate normally
    all_obs_codes = defaultdict(list)
    for y, x in obs_p1:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            all_obs_codes[code].append(obs1[(y, x)])
    for y, x in obs_p2:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            all_obs_codes[code].append(obs2[(y, x)])
    blended = simulate_calibration(igrid, all_obs_codes, loo_matrix, n_hist=50)

    obs_both = obs_p1 & obs_p2
    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC] * N_CLASSES
                dist[pc] = 1.0 - FLOOR_STATIC * (N_CLASSES - 1)
            else:
                prior = blended.get(code, [1.0 / N_CLASSES] * N_CLASSES)[:]

                # Neighbor adjustment
                nc = sett_neighbor_count.get((y, x), 0)
                if nc > 0 and code in (4, 11):  # Forest or Plains near settlements
                    boost = min(nc * 0.01, 0.05)  # up to 5% boost
                    prior[1] += boost              # boost P(Settlement)
                    prior[0] += boost * empty_ratio # boost P(Empty) — resource depletion
                    total_p = sum(prior)
                    prior = [p / total_p for p in prior]

                if (y, x) in obs_both:
                    oh = [0.0] * N_CLASSES
                    oh[obs1[(y, x)]] += 0.5
                    oh[obs2[(y, x)]] += 0.5
                    dist = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                elif (y, x) in obs_p1:
                    oh = [0.0] * N_CLASSES
                    oh[obs1[(y, x)]] = 1.0
                    dist = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                elif (y, x) in obs_p2:
                    oh = [0.0] * N_CLASSES
                    oh[obs2[(y, x)]] = 1.0
                    dist = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, FLOOR_DYN) for v in dist]
            total = sum(dist)
            row.append([v / total for v in dist])
        tensor.append(row)
    return score_tensor(tensor, gt)


def run_neighbor_sett_only(igrid, gt, loo_matrix, _regime_matrices=None):
    """Neighbor boost: only settlement expansion (no empty boost)."""
    return _neighbor_aware_helper(igrid, gt, loo_matrix, empty_ratio=0.0)


def run_neighbor_sett_empty_05(igrid, gt, loo_matrix, _regime_matrices=None):
    """Neighbor boost: settlement + empty at 0.5 ratio."""
    return _neighbor_aware_helper(igrid, gt, loo_matrix, empty_ratio=0.5)


def run_neighbor_sett_empty_07(igrid, gt, loo_matrix, _regime_matrices=None):
    """Neighbor boost: settlement + empty at 0.7 ratio."""
    return _neighbor_aware_helper(igrid, gt, loo_matrix, empty_ratio=0.7)


def run_neighbor_sett_empty_10(igrid, gt, loo_matrix, _regime_matrices=None):
    """Neighbor boost: settlement + empty at 1.0 ratio."""
    return _neighbor_aware_helper(igrid, gt, loo_matrix, empty_ratio=1.0)


def run_per_terrain_floor(igrid, gt, loo_matrix, _regime_matrices=None):
    obs1, obs2, obs_p1, obs_p2 = get_obs_and_tiles(igrid, gt)
    per_floor = {1: 0.008, 2: 0.008, 3: 0.006, 4: 0.003, 11: 0.004, 0: 0.005}
    tensor = run_pipeline(igrid, gt, loo_matrix, obs_p1, obs_p2, obs1, obs2,
                          per_terrain_floor=per_floor)
    return score_tensor(tensor, gt)


def run_per_terrain_alpha(igrid, gt, loo_matrix, _regime_matrices=None):
    obs1, obs2, obs_p1, obs_p2 = get_obs_and_tiles(igrid, gt)
    per_alpha = {1: 0.10, 2: 0.10, 3: 0.08, 4: 0.02, 11: 0.03, 0: 0.05}
    tensor = run_pipeline(igrid, gt, loo_matrix, obs_p1, obs_p2, obs1, obs2,
                          per_terrain_alpha=per_alpha)
    return score_tensor(tensor, gt)


def run_spatial_smooth(igrid, gt, loo_matrix, _regime_matrices=None):
    obs1, obs2, obs_p1, obs_p2 = get_obs_and_tiles(igrid, gt)
    tensor = run_pipeline(igrid, gt, loo_matrix, obs_p1, obs_p2, obs1, obs2,
                          spatial_smooth=True)
    return score_tensor(tensor, gt)


def run_combined(igrid, gt, loo_matrix, _regime_matrices=None):
    """Stack the best ideas: per-terrain floor + per-terrain alpha + spatial smoothing."""
    obs1, obs2, obs_p1, obs_p2 = get_obs_and_tiles(igrid, gt)
    per_alpha = {1: 0.10, 2: 0.10, 3: 0.08, 4: 0.02, 11: 0.03, 0: 0.05}
    per_floor = {1: 0.008, 2: 0.008, 3: 0.006, 4: 0.003, 11: 0.004, 0: 0.005}
    tensor = run_pipeline(igrid, gt, loo_matrix, obs_p1, obs_p2, obs1, obs2,
                          per_terrain_alpha=per_alpha, per_terrain_floor=per_floor,
                          spatial_smooth=True)
    return score_tensor(tensor, gt)


# =============================================================================
# MAIN
# =============================================================================

def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files:
        rounds[f.split("_seed")[0]].append(f)

    print(f"Improvement test: {len(files)} files, {len(rounds)} rounds, {N_MC} MC trials")
    print()

    all_data = {}
    for fname in files:
        with open(os.path.join(history_dir, fname)) as f:
            all_data[fname] = json.load(f)

    strategies = [
        ("A. Baseline (no neighbor)", run_baseline),
        ("B. Neighbor: sett only", run_neighbor_sett_only),
        ("C. Neighbor: sett+empty 0.5", run_neighbor_sett_empty_05),
        ("D. Neighbor: sett+empty 0.7", run_neighbor_sett_empty_07),
        ("E. Neighbor: sett+empty 1.0", run_neighbor_sett_empty_10),
        ("F. Regime detection", run_regime_detection),
        ("G. Per-terrain floor", run_per_terrain_floor),
    ]

    results = {}
    for label, run_fn in strategies:
        print(f"  [{label}]...", end="", flush=True)
        all_scores = []
        round_scores = defaultdict(list)
        for round_id, round_files in sorted(rounds.items()):
            loo = build_matrix_excluding_round(files, round_id, history_dir, all_data)
            regime_mats = build_regime_matrices(files, round_id, all_data) if "Regime" in label else None
            for fname in round_files:
                d = all_data[fname]
                gt, igrid = d.get("ground_truth"), d.get("initial_grid")
                if not gt or not igrid:
                    continue
                ts = [run_fn(igrid, gt, loo, regime_mats) for _ in range(N_MC)]
                avg = sum(ts) / len(ts)
                all_scores.append(avg)
                round_scores[round_id].append(avg)
        overall = sum(all_scores) / len(all_scores)
        results[label] = (overall, round_scores)
        print(f" avg={overall:.2f}")

    # Summary
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    baseline = results["A. Baseline (no neighbor)"][0]
    print(f"\n  {'Strategy':<32} {'Avg':>6}  {'vs Base':>7}")
    print(f"  {'-'*32} {'-'*6}  {'-'*7}")
    for label in [l for l, _ in strategies]:
        avg = results[label][0]
        delta = avg - baseline
        print(f"  {label:<32} {avg:6.2f}  {delta:+7.2f}")

    # Per-round detail
    print()
    print("  Per-round breakdown:")
    print(f"  {'Round':<12}", end="")
    for label in [l for l, _ in strategies]:
        short = label.split(".")[1][:10].strip()
        print(f" {short:>10}", end="")
    print()
    print("  " + "-" * (12 + 11 * len(strategies)))

    for round_id in sorted(list(rounds.keys())):
        short_id = round_id[:8]
        print(f"  {short_id:<12}", end="")
        for label in [l for l, _ in strategies]:
            scores = results[label][1][round_id]
            avg = sum(scores) / len(scores)
            print(f" {avg:10.2f}", end="")
        print()


if __name__ == "__main__":
    main()
