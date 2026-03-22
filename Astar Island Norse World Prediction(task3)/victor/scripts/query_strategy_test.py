"""
query_strategy_test.py - Test different query budget splits.

Compare:
  A. Current 25+25: 5 settlement tiles + 5 spread tiles per seed
  B. 40+10 broad: 8 coverage tiles + 2 entropy-targeted tiles per seed
  C. 45+5: 9 tiles (full map) + 1 entropy-targeted per seed
  D. 50 broad: 10 coverage tiles, no phase 2

Run from victor/ folder:
    python -m scripts.query_strategy_test
"""

import os, sys, json, math, random
from collections import defaultdict
from typing import List, Dict, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N_CLASSES = 6
FLOOR_STATIC = 1e-5
FLOOR_DYN = 0.005
N_MC = 5
ALPHA = 0.05
random.seed(42)

# All valid 15x15 tile anchors
ALL_ANCHORS = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
ANCHOR_CELLS = {a: set(_covered_cells(*a)) for a in ALL_ANCHORS}

# 9-tile full coverage grid
FULL_GRID_ANCHORS = [
    (0, 0), (15, 0), (25, 0),
    (0, 15), (15, 15), (25, 15),
    (0, 25), (15, 25), (25, 25),
]


def kl_divergence(p, q):
    return sum(pi * math.log(pi / max(qi, 1e-12)) for pi, qi in zip(p, q) if pi > 1e-12)


def score_tensor(pred, gt):
    H, W = len(gt), len(gt[0])
    wkl = te = 0.0
    for y in range(H):
        for x in range(W):
            e = _entropy(gt[y][x])
            wkl += e * kl_divergence(gt[y][x], pred[y][x])
            te += e
    return max(0, min(100, 100 * math.exp(-3 * (wkl / te)))) if te > 1e-12 else 100


def sample_observation(gt_dist):
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(gt_dist):
        cumsum += p
        if r < cumsum:
            return i
    return len(gt_dist) - 1


def count_neighbors(igrid, y, x, target_code, radius=2):
    H, W = len(igrid), len(igrid[0])
    c = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and igrid[ny][nx] == target_code:
                c += 1
    return c


def cell_context(igrid, y, x):
    code = igrid[y][x]
    sett_n = count_neighbors(igrid, y, x, 1)
    ocean_n = count_neighbors(igrid, y, x, 10)
    sett_bin = "sett_hi" if sett_n >= 3 else ("sett_lo" if sett_n >= 1 else "sett_no")
    ocean_bin = "ocean" if ocean_n >= 1 else "inland"
    return (code, sett_bin, ocean_bin)


def build_conditional_loo(all_files, exclude_round_id, all_data):
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
                    ctx = cell_context(igrid, y, x)
                    accum[ctx].append(gt[y][x])
    matrix = {}
    for ctx, samples in accum.items():
        n = len(samples)
        matrix[ctx] = [sum(s[i] for s in samples) / n for i in range(N_CLASSES)]
    return matrix


def select_settlement_tiles(igrid, n=5):
    setts = {(y, x) for y in range(len(igrid)) for x in range(len(igrid[0])) if igrid[y][x] == 1}
    if not setts:
        return SPREAD_ANCHORS[:n]
    covered = set()
    sel = []
    for _ in range(n):
        best, bc = None, -1
        for a in ALL_ANCHORS:
            c = len((ANCHOR_CELLS[a] & setts) - covered)
            if c > bc:
                bc, best = c, a
        if not best or bc <= 0:
            break
        sel.append(best)
        covered |= (ANCHOR_CELLS[best] & setts)
    return sel


def select_coverage_tiles(igrid, n=8):
    """Select tiles to maximize coverage of dynamic cells, prioritizing settlements."""
    H, W = len(igrid), len(igrid[0])
    # Weight cells: settlements > other dynamic > static (0)
    weights = {}
    for y in range(H):
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                weights[(y, x)] = 0.0
            elif code == 1:  # Settlement
                weights[(y, x)] = 3.0
            elif code == 2:  # Port
                weights[(y, x)] = 2.0
            else:
                weights[(y, x)] = 1.0

    covered = set()
    sel = []
    for _ in range(n):
        best, bscore = None, -1
        for a in ALL_ANCHORS:
            uncovered = ANCHOR_CELLS[a] - covered
            score = sum(weights.get(c, 0) for c in uncovered)
            if score > bscore:
                bscore, best = score, a
        if not best or bscore <= 0:
            break
        sel.append(best)
        covered |= ANCHOR_CELLS[best]
    return sel


def select_entropy_tiles(igrid, gt, prior_tensor, observed_cells, n=2):
    """Select tiles covering highest-entropy unobserved cells."""
    H, W = len(igrid), len(igrid[0])
    entropy_map = {}
    for y in range(H):
        for x in range(W):
            if (y, x) not in observed_cells:
                entropy_map[(y, x)] = _entropy(prior_tensor[y][x])

    sel = []
    remaining = dict(entropy_map)
    for _ in range(n):
        best, bscore = None, -1
        for a in ALL_ANCHORS:
            score = sum(remaining.get(c, 0) for c in ANCHOR_CELLS[a])
            if score > bscore:
                bscore, best = score, a
        if not best or bscore <= 1e-9:
            break
        sel.append(best)
        for c in ANCHOR_CELLS[best]:
            remaining.pop(c, None)
    return sel


def calibrate_conditional(igrid, obs_cells, obs_values, cond_matrix, n_hist=50):
    """Calibrate conditional matrix using observations."""
    round_counts = defaultdict(list)
    for (y, x) in obs_cells:
        code = igrid[y][x]
        if code in STATIC_CODES:
            continue
        ctx = cell_context(igrid, y, x)
        round_counts[ctx].append(obs_values[(y, x)])

    blended = {}
    for ctx in set(list(cond_matrix.keys()) + list(round_counts.keys())):
        hist = cond_matrix.get(ctx, [1.0 / N_CLASSES] * N_CLASSES)
        ol = round_counts.get(ctx, [])
        nr = len(ol)
        if nr == 0:
            blended[ctx] = hist[:]
            continue
        rf = [0.0] * N_CLASSES
        for c in ol:
            rf[c] += 1.0 / nr
        t = nr + n_hist
        blended[ctx] = [(nr * rf[i] + n_hist * hist[i]) / t for i in range(N_CLASSES)]
    return blended


def build_tensor(igrid, gt, obs_cells, obs_values, cond_matrix):
    """Build prediction tensor using conditional matrix + observations."""
    H, W = len(igrid), len(igrid[0])

    # Calibrate with observations
    blended = calibrate_conditional(igrid, obs_cells, obs_values, cond_matrix)

    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                d = [FLOOR_STATIC] * N_CLASSES
                d[pc] = 1.0 - FLOOR_STATIC * 5
            else:
                ctx = cell_context(igrid, y, x)
                prior = blended.get(ctx, cond_matrix.get(ctx, [1.0 / N_CLASSES] * N_CLASSES))[:]

                if (y, x) in obs_cells:
                    oh = [0.0] * N_CLASSES
                    # Could have multiple obs — use obs_values which is the latest
                    oh[obs_values[(y, x)]] = 1.0
                    d = [(1 - ALPHA) * prior[i] + ALPHA * oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]
                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        tensor.append(row)
    return tensor


def build_tensor_multi_obs(igrid, gt, obs_list, cond_matrix):
    """Build tensor with multiple observation phases, each cell may have 1-2 obs."""
    H, W = len(igrid), len(igrid[0])

    # Merge all observations
    all_obs_cells = set()
    all_obs_values = {}  # (y,x) -> list of observed classes
    for obs_cells, obs_values in obs_list:
        for (y, x) in obs_cells:
            all_obs_cells.add((y, x))
            if (y, x) not in all_obs_values:
                all_obs_values[(y, x)] = []
            all_obs_values[(y, x)].append(obs_values[(y, x)])

    # Calibrate with all observations (use latest/all for calibration)
    cal_cells = set()
    cal_values = {}
    for (y, x), vals in all_obs_values.items():
        cal_cells.add((y, x))
        cal_values[(y, x)] = vals[-1]  # use last observation for calibration counts
    blended = calibrate_conditional(igrid, cal_cells, cal_values, cond_matrix)

    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                d = [FLOOR_STATIC] * N_CLASSES
                d[pc] = 1.0 - FLOOR_STATIC * 5
            else:
                ctx = cell_context(igrid, y, x)
                prior = blended.get(ctx, cond_matrix.get(ctx, [1.0 / N_CLASSES] * N_CLASSES))[:]

                if (y, x) in all_obs_cells:
                    obs_vals = all_obs_values[(y, x)]
                    oh = [0.0] * N_CLASSES
                    for v in obs_vals:
                        oh[v] += 1.0 / len(obs_vals)
                    d = [(1 - ALPHA) * prior[i] + ALPHA * oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]
                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        tensor.append(row)
    return tensor


def run_strategy_a(igrid, gt, cond_matrix):
    """Current: 5 settlement + 5 spread = 10 tiles."""
    H, W = len(igrid), len(igrid[0])
    obs_all = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

    # Phase 1: 5 settlement tiles
    p1_tiles = select_settlement_tiles(igrid, n=5)
    p1_cells = set()
    for a in p1_tiles:
        p1_cells |= ANCHOR_CELLS[a]
    p1_obs = {(y, x): obs_all[(y, x)] for (y, x) in p1_cells}

    # Phase 2: 5 spread tiles (independent observations)
    obs_all2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p2_cells = set()
    for a in SPREAD_ANCHORS:
        p2_cells |= ANCHOR_CELLS[a]
    p2_obs = {(y, x): obs_all2[(y, x)] for (y, x) in p2_cells}

    tensor = build_tensor_multi_obs(igrid, gt, [(p1_cells, p1_obs), (p2_cells, p2_obs)], cond_matrix)
    return score_tensor(tensor, gt)


def run_strategy_b(igrid, gt, cond_matrix):
    """40+10: 8 coverage tiles + 2 entropy-targeted tiles."""
    H, W = len(igrid), len(igrid[0])
    obs_all = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

    # Phase 1: 8 coverage tiles
    p1_tiles = select_coverage_tiles(igrid, n=8)
    p1_cells = set()
    for a in p1_tiles:
        p1_cells |= ANCHOR_CELLS[a]
    p1_obs = {(y, x): obs_all[(y, x)] for (y, x) in p1_cells}

    # Build preliminary tensor for entropy targeting
    blended = calibrate_conditional(igrid, p1_cells, p1_obs, cond_matrix)
    prelim = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                d = [FLOOR_STATIC] * N_CLASSES
                d[pc] = 1.0 - FLOOR_STATIC * 5
            else:
                ctx = cell_context(igrid, y, x)
                d = blended.get(ctx, cond_matrix.get(ctx, [1.0 / N_CLASSES] * N_CLASSES))[:]
                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        prelim.append(row)

    # Phase 2: 2 entropy-targeted tiles (new observations)
    obs_all2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p2_tiles = select_entropy_tiles(igrid, gt, prelim, p1_cells, n=2)
    p2_cells = set()
    for a in p2_tiles:
        p2_cells |= ANCHOR_CELLS[a]
    p2_obs = {(y, x): obs_all2[(y, x)] for (y, x) in p2_cells}

    tensor = build_tensor_multi_obs(igrid, gt, [(p1_cells, p1_obs), (p2_cells, p2_obs)], cond_matrix)
    return score_tensor(tensor, gt)


def run_strategy_c(igrid, gt, cond_matrix):
    """45+5: 9 tiles (full grid) + 1 entropy-targeted."""
    H, W = len(igrid), len(igrid[0])
    obs_all = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

    # Phase 1: 9 full-grid tiles
    p1_cells = set()
    for a in FULL_GRID_ANCHORS:
        p1_cells |= ANCHOR_CELLS[a]
    p1_obs = {(y, x): obs_all[(y, x)] for (y, x) in p1_cells}

    # Build preliminary tensor
    blended = calibrate_conditional(igrid, p1_cells, p1_obs, cond_matrix)
    prelim = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                d = [FLOOR_STATIC] * N_CLASSES
                d[pc] = 1.0 - FLOOR_STATIC * 5
            else:
                ctx = cell_context(igrid, y, x)
                d = blended.get(ctx, cond_matrix.get(ctx, [1.0 / N_CLASSES] * N_CLASSES))[:]
                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        prelim.append(row)

    # Phase 2: 1 entropy-targeted tile (re-observe highest uncertainty)
    obs_all2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p2_tiles = select_entropy_tiles(igrid, gt, prelim, p1_cells, n=1)
    p2_cells = set()
    for a in p2_tiles:
        p2_cells |= ANCHOR_CELLS[a]
    p2_obs = {(y, x): obs_all2[(y, x)] for (y, x) in p2_cells}

    tensor = build_tensor_multi_obs(igrid, gt, [(p1_cells, p1_obs), (p2_cells, p2_obs)], cond_matrix)
    return score_tensor(tensor, gt)


def run_strategy_d(igrid, gt, cond_matrix):
    """50 broad: 10 coverage tiles, no phase 2."""
    H, W = len(igrid), len(igrid[0])
    obs_all = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

    # All 10 tiles: coverage-optimized
    tiles = select_coverage_tiles(igrid, n=10)
    cells = set()
    for a in tiles:
        cells |= ANCHOR_CELLS[a]
    obs = {(y, x): obs_all[(y, x)] for (y, x) in cells}

    tensor = build_tensor_multi_obs(igrid, gt, [(cells, obs)], cond_matrix)
    return score_tensor(tensor, gt)


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files:
        rounds[f.split("_seed")[0]].append(f)

    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh:
            all_data[f] = json.load(fh)

    print(f"Query strategy test: {len(files)} files, {len(rounds)} rounds, {N_MC} MC")
    print()

    strategies = {
        "A. 25+25 sett+spread": run_strategy_a,
        "B. 40+10 coverage+entropy": run_strategy_b,
        "C. 45+5 fullgrid+entropy": run_strategy_c,
        "D. 50 coverage only": run_strategy_d,
    }

    results = {s: defaultdict(list) for s in strategies}

    for round_id, round_files in sorted(rounds.items()):
        cond_loo = build_conditional_loo(files, round_id, all_data)

        for fname in round_files:
            d = all_data[fname]
            gt, igrid = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not igrid:
                continue

            for _ in range(N_MC):
                for label, fn in strategies.items():
                    score = fn(igrid, gt, cond_loo)
                    results[label][round_id].append(score)

        short = round_id[:8]
        print(f"  {short}", end="", flush=True)
        for label in strategies:
            avg = sum(results[label][round_id]) / len(results[label][round_id])
            print(f"  {avg:6.1f}", end="")
        print()

    # Coverage stats for one sample
    sample_fname = files[0]
    sample_igrid = all_data[sample_fname].get("initial_grid")
    if sample_igrid:
        all_cells = set((y, x) for y in range(40) for x in range(40))
        dynamic = {(y, x) for y, x in all_cells if sample_igrid[y][x] not in STATIC_CODES}

        for label in ["5 sett+5 spread", "8 coverage", "9 fullgrid", "10 coverage"]:
            if label == "5 sett+5 spread":
                t1 = select_settlement_tiles(sample_igrid, 5)
                t2 = list(SPREAD_ANCHORS)
                tiles = t1 + t2
            elif label == "8 coverage":
                tiles = select_coverage_tiles(sample_igrid, 8)
            elif label == "9 fullgrid":
                tiles = list(FULL_GRID_ANCHORS)
            else:
                tiles = select_coverage_tiles(sample_igrid, 10)

            covered = set()
            for a in tiles:
                covered |= ANCHOR_CELLS[a]
            cov_dyn = covered & dynamic
            print(f"\n  {label}: {len(covered)}/{len(all_cells)} total cells "
                  f"({len(cov_dyn)}/{len(dynamic)} dynamic = {len(cov_dyn)/len(dynamic)*100:.0f}%)")

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  {'Strategy':<30} {'Avg':>6}")
    print(f"  {'-' * 30} {'-' * 6}")
    for label in strategies:
        all_scores = []
        for rid in rounds:
            all_scores.extend(results[label][rid])
        avg = sum(all_scores) / len(all_scores)
        print(f"  {label:<30} {avg:6.2f}")


if __name__ == "__main__":
    main()
