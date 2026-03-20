"""
smart_query_test.py - Test intelligent query targeting strategies.

Key insight: not all cells benefit equally from observation.
  - Settlement: 45% Empty, 30% Settlement, 21% Forest — HIGH uncertainty, obs very valuable
  - Port: 47% Empty, 20% Port, 22% Forest — HIGH uncertainty
  - Forest: 75% Forest — LOW uncertainty, obs barely helps
  - Plains: 80% Empty — LOW uncertainty, obs barely helps
  - Ocean/Mountain: 100% known — ZERO value from obs

Strategy: target tiles containing the most settlements/ports.

Run from victor/ folder:
    python -m scripts.smart_query_test
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

def build_matrix_excluding_round(all_files, exclude_round_id, history_dir):
    counts = defaultdict(lambda: [0.0] * N_CLASSES)
    n_files = 0
    accum = defaultdict(list)
    for fname in all_files:
        if fname.split("_seed")[0] == exclude_round_id:
            continue
        with open(os.path.join(history_dir, fname)) as f:
            data = json.load(f)
        gt, igrid = data.get("ground_truth"), data.get("initial_grid")
        if not gt or not igrid:
            continue
        n_files += 1
        for y in range(len(igrid)):
            for x in range(len(igrid[0])):
                code = igrid[y][x]
                if code not in STATIC_CODES:
                    accum[code].append(gt[y][x])
    # Average probability distributions (not argmax)
    matrix = {}
    for code, samples in accum.items():
        n = len(samples)
        matrix[code] = [sum(s[i] for s in samples)/n for i in range(N_CLASSES)]
    for code in [0,1,2,3,4,5,10,11]:
        if code not in matrix:
            matrix[code] = [1.0/N_CLASSES]*N_CLASSES
    return matrix

def simulate_calibration(igrid, obs_cells, obs_samples, hist_matrix, n_hist=50):
    round_counts = defaultdict(list)
    for y, x in obs_cells:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            round_counts[code].append(obs_samples[(y, x)])
    blended = {}
    for code in [0,1,2,3,4,5,10,11]:
        hist = hist_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES:
            blended[code] = hist[:]; continue
        obs_list = round_counts.get(code, [])
        nr = len(obs_list)
        if nr == 0:
            blended[code] = hist[:]; continue
        rf = [0.0]*N_CLASSES
        for c in obs_list: rf[c] += 1.0/nr
        t = nr + n_hist
        blended[code] = [(nr*rf[i]+n_hist*hist[i])/t for i in range(N_CLASSES)]
    return blended

def build_prediction(igrid, gt, matrix, obs_cells, obs_samples_list, alpha, floor_dyn):
    """Build prediction. obs_samples_list = list of obs dicts (for averaging multiple obs)."""
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
                    # Average across multiple observations
                    avg_oh = [0.0]*N_CLASSES
                    n_obs = len(obs_samples_list)
                    for obs_s in obs_samples_list:
                        avg_oh[obs_s[(y,x)]] += 1.0/n_obs
                    dist = [(1-alpha)*prior[i] + alpha*avg_oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, floor_dyn) for v in dist]
            total = sum(dist)
            row.append([v/total for v in dist])
        tensor.append(row)
    return tensor


# =============================================================================
# QUERY STRATEGIES
# =============================================================================

def get_all_tile_anchors():
    """All valid 15x15 tile positions on 40x40 map."""
    return [(ax, ay)
            for ay in range(config.MAP_HEIGHT - TILE_H + 1)
            for ax in range(config.MAP_WIDTH - TILE_W + 1)]


def score_tile_by_volatility(igrid, ax, ay):
    """Score a tile by how many volatile cells it contains.
    Settlements and Ports are most volatile (highest entropy in GT)."""
    VOLATILE_CODES = {1: 3.0, 2: 3.0, 3: 2.0}  # Settlement, Port, Ruin
    MEDIUM_CODES = {11: 0.5, 4: 0.3}  # Plains, Forest (some uncertainty)
    # Static codes (5=Mountain, 10=Ocean) get 0
    score = 0.0
    for y, x in _covered_cells(ax, ay):
        code = igrid[y][x]
        score += VOLATILE_CODES.get(code, MEDIUM_CODES.get(code, 0.0))
    return score


def select_volatile_tiles(igrid, n_tiles=5):
    """Greedily select tiles covering the most volatile cells."""
    all_anchors = get_all_tile_anchors()
    anchor_cells = {a: set(_covered_cells(*a)) for a in all_anchors}

    covered = set()
    selected = []

    for _ in range(n_tiles):
        best, best_score = None, -1
        for a in all_anchors:
            # Only count uncovered cells
            uncovered = anchor_cells[a] - covered
            score = 0.0
            for y, x in uncovered:
                code = igrid[y][x]
                if code in (1, 2, 3):
                    score += 3.0
                elif code == 11:
                    score += 0.5
                elif code == 4:
                    score += 0.3
            if score > best_score:
                best_score, best = score, a
        if best is None:
            break
        selected.append(best)
        covered |= anchor_cells[best]
    return selected


def select_settlement_cluster_tiles(igrid, n_tiles=5):
    """Find tiles centered on settlement clusters."""
    H, W = len(igrid), len(igrid[0])

    # Find all settlement positions
    settlements = [(y, x) for y in range(H) for x in range(W) if igrid[y][x] == 1]

    if not settlements:
        return PHASE1_ANCHORS[:n_tiles]

    # For each tile anchor, count settlements covered
    all_anchors = get_all_tile_anchors()
    anchor_cells = {a: set(_covered_cells(*a)) for a in all_anchors}

    sett_set = set(settlements)
    covered_setts = set()
    selected = []

    for _ in range(n_tiles):
        best, best_count = None, -1
        for a in all_anchors:
            count = len((anchor_cells[a] & sett_set) - covered_setts)
            if count > best_count:
                best_count, best = count, a
        if best is None or best_count <= 0:
            break
        selected.append(best)
        covered_setts |= (anchor_cells[best] & sett_set)

    # Fill remaining with spatial spread if needed
    while len(selected) < n_tiles and len(selected) < len(PHASE1_ANCHORS):
        a = PHASE1_ANCHORS[len(selected)]
        if a not in selected:
            selected.append(a)

    return selected[:n_tiles]


# =============================================================================
# PIPELINE RUNNERS
# =============================================================================

def run_no_queries(igrid, gt, loo_matrix):
    """Pure matrix, zero observations."""
    tensor = build_prediction(igrid, gt, loo_matrix, set(), [], 0.0, FLOOR_DYN)
    return score_tensor(tensor, gt)


def run_fixed_spread_overlap(igrid, gt, loo_matrix):
    """Current strategy: 5 fixed tiles x2 (overlap)."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs_cells = set()
    for ax, ay in PHASE1_ANCHORS:
        obs_cells |= tile_cells(ax, ay)
    cal = simulate_calibration(igrid, obs_cells, obs1, loo_matrix, 50)
    # Re-calibrate with both obs
    all_obs_codes = defaultdict(list)
    for y,x in obs_cells:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            all_obs_codes[code].append(obs1[(y,x)])
            all_obs_codes[code].append(obs2[(y,x)])
    blended = {}
    for code in [0,1,2,3,4,5,10,11]:
        hist = loo_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES: blended[code] = hist[:]; continue
        ol = all_obs_codes.get(code, [])
        nr = len(ol)
        if nr == 0: blended[code] = hist[:]; continue
        rf = [0.0]*N_CLASSES
        for c in ol: rf[c] += 1.0/nr
        t = nr + 50
        blended[code] = [(nr*rf[i]+50*hist[i])/t for i in range(N_CLASSES)]
    pred = build_prediction(igrid, gt, blended, obs_cells, [obs1, obs2], 0.05, FLOOR_DYN)
    return score_tensor(pred, gt)


def run_volatile_targeting(igrid, gt, loo_matrix):
    """Target tiles with most volatile cells (settlements/ports), x2 overlap."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

    tiles = select_volatile_tiles(igrid, n_tiles=5)
    obs_cells = set()
    for ax, ay in tiles:
        obs_cells |= tile_cells(ax, ay)

    all_obs_codes = defaultdict(list)
    for y,x in obs_cells:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            all_obs_codes[code].append(obs1[(y,x)])
            all_obs_codes[code].append(obs2[(y,x)])
    blended = {}
    for code in [0,1,2,3,4,5,10,11]:
        hist = loo_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES: blended[code] = hist[:]; continue
        ol = all_obs_codes.get(code, [])
        nr = len(ol)
        if nr == 0: blended[code] = hist[:]; continue
        rf = [0.0]*N_CLASSES
        for c in ol: rf[c] += 1.0/nr
        t = nr + 50
        blended[code] = [(nr*rf[i]+50*hist[i])/t for i in range(N_CLASSES)]
    pred = build_prediction(igrid, gt, blended, obs_cells, [obs1, obs2], 0.05, FLOOR_DYN)
    return score_tensor(pred, gt)


def run_settlement_targeting(igrid, gt, loo_matrix):
    """Target tiles centered on settlement clusters, x2 overlap."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

    tiles = select_settlement_cluster_tiles(igrid, n_tiles=5)
    obs_cells = set()
    for ax, ay in tiles:
        obs_cells |= tile_cells(ax, ay)

    all_obs_codes = defaultdict(list)
    for y,x in obs_cells:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            all_obs_codes[code].append(obs1[(y,x)])
            all_obs_codes[code].append(obs2[(y,x)])
    blended = {}
    for code in [0,1,2,3,4,5,10,11]:
        hist = loo_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES: blended[code] = hist[:]; continue
        ol = all_obs_codes.get(code, [])
        nr = len(ol)
        if nr == 0: blended[code] = hist[:]; continue
        rf = [0.0]*N_CLASSES
        for c in ol: rf[c] += 1.0/nr
        t = nr + 50
        blended[code] = [(nr*rf[i]+50*hist[i])/t for i in range(N_CLASSES)]
    pred = build_prediction(igrid, gt, blended, obs_cells, [obs1, obs2], 0.05, FLOOR_DYN)
    return score_tensor(pred, gt)


def run_volatile_spread(igrid, gt, loo_matrix):
    """Volatile targeting Phase 1 (5 tiles) + spread Phase 2 (5 different tiles).
    Gets both calibration diversity AND settlement coverage."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

    # Phase 1: volatile tiles
    vol_tiles = select_volatile_tiles(igrid, n_tiles=5)
    obs_p1 = set()
    for ax, ay in vol_tiles:
        obs_p1 |= tile_cells(ax, ay)

    # Phase 2: spread tiles (for calibration diversity)
    obs_p2 = set()
    for ax, ay in PHASE1_ANCHORS:
        obs_p2 |= tile_cells(ax, ay)

    obs_all = obs_p1 | obs_p2

    # Use obs1 for volatile tiles, obs2 for spread tiles
    all_obs_codes = defaultdict(list)
    for y,x in obs_p1:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            all_obs_codes[code].append(obs1[(y,x)])
    for y,x in obs_p2:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            all_obs_codes[code].append(obs2[(y,x)])

    blended = {}
    for code in [0,1,2,3,4,5,10,11]:
        hist = loo_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES: blended[code] = hist[:]; continue
        ol = all_obs_codes.get(code, [])
        nr = len(ol)
        if nr == 0: blended[code] = hist[:]; continue
        rf = [0.0]*N_CLASSES
        for c in ol: rf[c] += 1.0/nr
        t = nr + 50
        blended[code] = [(nr*rf[i]+50*hist[i])/t for i in range(N_CLASSES)]

    # Cells in both phases get 2 obs, cells in only one get 1
    obs_both = obs_p1 & obs_p2
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
                prior = blended.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                if (y,x) in obs_both:
                    oh = [0.0]*N_CLASSES
                    oh[obs1[(y,x)]] += 0.5
                    oh[obs2[(y,x)]] += 0.5
                    dist = [0.95*prior[i]+0.05*oh[i] for i in range(N_CLASSES)]
                elif (y,x) in obs_p1:
                    oh = [0.0]*N_CLASSES
                    oh[obs1[(y,x)]] = 1.0
                    dist = [0.95*prior[i]+0.05*oh[i] for i in range(N_CLASSES)]
                elif (y,x) in obs_p2:
                    oh = [0.0]*N_CLASSES
                    oh[obs2[(y,x)]] = 1.0
                    dist = [0.95*prior[i]+0.05*oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, FLOOR_DYN) for v in dist]
            total = sum(dist)
            row.append([v/total for v in dist])
        tensor.append(row)
    return score_tensor(tensor, gt)


def run_volatile_only_alpha(igrid, gt, loo_matrix):
    """Only apply alpha to volatile cells (sett/port), ignore obs for forest/plains."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs_cells = set()
    for ax, ay in PHASE1_ANCHORS:
        obs_cells |= tile_cells(ax, ay)

    all_obs_codes = defaultdict(list)
    for y,x in obs_cells:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            all_obs_codes[code].append(obs1[(y,x)])
            all_obs_codes[code].append(obs2[(y,x)])
    blended = {}
    for code in [0,1,2,3,4,5,10,11]:
        hist = loo_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES: blended[code] = hist[:]; continue
        ol = all_obs_codes.get(code, [])
        nr = len(ol)
        if nr == 0: blended[code] = hist[:]; continue
        rf = [0.0]*N_CLASSES
        for c in ol: rf[c] += 1.0/nr
        t = nr + 50
        blended[code] = [(nr*rf[i]+50*hist[i])/t for i in range(N_CLASSES)]

    VOLATILE = {1, 2, 3}  # Settlement, Port, Ruin
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
                prior = blended.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                if (y,x) in obs_cells and code in VOLATILE:
                    # Higher alpha for volatile cells
                    oh = [0.0]*N_CLASSES
                    oh[obs1[(y,x)]] += 0.5
                    oh[obs2[(y,x)]] += 0.5
                    dist = [0.85*prior[i]+0.15*oh[i] for i in range(N_CLASSES)]
                elif (y,x) in obs_cells:
                    # Very low alpha for stable cells (forest, plains)
                    oh = [0.0]*N_CLASSES
                    oh[obs1[(y,x)]] += 0.5
                    oh[obs2[(y,x)]] += 0.5
                    dist = [0.98*prior[i]+0.02*oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, FLOOR_DYN) for v in dist]
            total = sum(dist)
            row.append([v/total for v in dist])
        tensor.append(row)
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

    print(f"Smart query test: {len(files)} files, {len(rounds)} rounds, {N_MC} MC trials")
    print(f"Floor = {FLOOR_DYN}")
    print()

    all_data = {}
    for fname in files:
        with open(os.path.join(history_dir, fname)) as f:
            all_data[fname] = json.load(f)

    strategies = [
        ("0. Pure matrix (no queries)", run_no_queries),
        ("1. Fixed spread + overlap", run_fixed_spread_overlap),
        ("2. Volatile targeting + overlap", run_volatile_targeting),
        ("3. Settlement cluster + overlap", run_settlement_targeting),
        ("4. Volatile P1 + Spread P2", run_volatile_spread),
        ("5. Selective alpha (high=sett)", run_volatile_only_alpha),
    ]

    results = {}
    for label, run_fn in strategies:
        print(f"  [{label}]...", end="", flush=True)
        all_scores = []
        round_scores = defaultdict(list)
        for round_id, round_files in sorted(rounds.items()):
            loo = build_matrix_excluding_round(files, round_id, history_dir)
            for fname in round_files:
                d = all_data[fname]
                gt, igrid = d.get("ground_truth"), d.get("initial_grid")
                if not gt or not igrid: continue
                ts = [run_fn(igrid, gt, loo) for _ in range(N_MC)]
                avg = sum(ts)/len(ts)
                all_scores.append(avg)
                round_scores[round_id].append(avg)
        overall = sum(all_scores)/len(all_scores)
        results[label] = (overall, round_scores)
        print(f" avg={overall:.2f}")

    # Summary
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    baseline = results["0. Pure matrix (no queries)"][0]
    print(f"\n  {'Strategy':<40} {'Avg':>6}  {'vs NoQ':>7}")
    print(f"  {'-'*40} {'-'*6}  {'-'*7}")
    for label in [l for l, _ in strategies]:
        avg = results[label][0]
        delta = avg - baseline
        print(f"  {label:<40} {avg:6.2f}  {delta:+7.2f}")

    # Per-round detail for top strategies
    print()
    print("  Per-round breakdown:")
    print(f"  {'Round':<12}", end="")
    for label in [l for l, _ in strategies]:
        short = label.split(".")[0] + "." + label.split(".")[1][:8]
        print(f" {short:>10}", end="")
    print()
    print("  " + "-" * (12 + 11 * len(strategies)))

    for round_id in sorted(list(rounds.keys())):
        short_id = round_id[:8]
        print(f"  {short_id:<12}", end="")
        for label in [l for l, _ in strategies]:
            scores = results[label][1][round_id]
            avg = sum(scores)/len(scores)
            print(f" {avg:10.2f}", end="")
        print()


if __name__ == "__main__":
    main()
