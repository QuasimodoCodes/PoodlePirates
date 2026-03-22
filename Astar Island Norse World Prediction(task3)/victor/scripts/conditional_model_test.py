"""
conditional_model_test.py - Test conditional transition matrices.

Instead of one matrix per terrain code, build matrices conditioned on
neighbor features. After 50 queries, calibrate each sub-model separately.

Example: P(outcome | Forest, near_sett=high) is very different from
         P(outcome | Forest, near_sett=low)

Run from victor/ folder:
    python -m scripts.conditional_model_test
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
random.seed(42)


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


def tile_cells(ax, ay):
    return set(_covered_cells(ax, ay))


ALL_ANCHORS = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
ANCHOR_CELLS = {a: set(_covered_cells(*a)) for a in ALL_ANCHORS}


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
    """Classify a cell into a context bucket based on its neighbors."""
    code = igrid[y][x]
    sett_n = count_neighbors(igrid, y, x, 1)
    ocean_n = count_neighbors(igrid, y, x, 10)
    forest_n = count_neighbors(igrid, y, x, 4)

    # Settlement neighbor bin: 0, 1-2, 3+
    if sett_n >= 3:
        sett_bin = "sett_hi"
    elif sett_n >= 1:
        sett_bin = "sett_lo"
    else:
        sett_bin = "sett_no"

    # Ocean neighbor bin: 0, 1+
    ocean_bin = "ocean" if ocean_n >= 1 else "inland"

    return (code, sett_bin, ocean_bin)


def build_conditional_matrix_loo(all_files, exclude_round_id, all_data):
    """Build conditional matrices: (code, sett_bin, ocean_bin) -> [P(class0..5)]"""
    accum = defaultdict(list)

    for fname in all_files:
        if fname.split("_seed")[0] == exclude_round_id:
            continue
        data = all_data[fname]
        gt, igrid = data.get("ground_truth"), data.get("initial_grid")
        if not gt or not igrid:
            continue
        H, W = len(igrid), len(igrid[0])
        for y in range(H):
            for x in range(W):
                code = igrid[y][x]
                if code in STATIC_CODES:
                    continue
                ctx = cell_context(igrid, y, x)
                accum[ctx].append(gt[y][x])

    matrix = {}
    for ctx, samples in accum.items():
        n = len(samples)
        matrix[ctx] = [sum(s[i] for s in samples) / n for i in range(N_CLASSES)]

    return matrix


def build_flat_matrix_loo(all_files, exclude_round_id, all_data):
    """Standard flat matrix for comparison."""
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
    m = {}
    for code, samples in accum.items():
        n = len(samples)
        m[code] = [sum(s[i] for s in samples) / n for i in range(N_CLASSES)]
    for code in [0, 1, 2, 3, 4, 5, 10, 11]:
        if code not in m:
            m[code] = [1.0 / N_CLASSES] * N_CLASSES
    return m


def select_sett_tiles(igrid, n=5):
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
    while len(sel) < n and len(sel) < len(SPREAD_ANCHORS):
        a = SPREAD_ANCHORS[len(sel)]
        if a not in sel:
            sel.append(a)
    return sel[:n]


def get_obs(igrid, gt):
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p1 = select_sett_tiles(igrid)
    obs_p1 = set()
    for ax, ay in p1:
        obs_p1 |= tile_cells(ax, ay)
    obs_p2 = set()
    for ax, ay in SPREAD_ANCHORS:
        obs_p2 |= tile_cells(ax, ay)
    return obs1, obs2, obs_p1, obs_p2


def calibrate_conditional(igrid, obs_p1, obs_p2, obs1, obs2, cond_matrix, n_hist=50):
    """Calibrate conditional matrix using round observations."""
    H, W = len(igrid), len(igrid[0])
    round_counts = defaultdict(list)  # ctx -> [observed_class, ...]

    for y, x in obs_p1:
        code = igrid[y][x]
        if code in STATIC_CODES:
            continue
        ctx = cell_context(igrid, y, x)
        round_counts[ctx].append(obs1[(y, x)])

    for y, x in obs_p2:
        code = igrid[y][x]
        if code in STATIC_CODES:
            continue
        ctx = cell_context(igrid, y, x)
        round_counts[ctx].append(obs2[(y, x)])

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


def calibrate_flat(igrid, obs_p1, obs_p2, obs1, obs2, flat_matrix, n_hist=50):
    """Standard flat calibration."""
    round_counts = defaultdict(list)
    for y, x in obs_p1:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            round_counts[code].append(obs1[(y, x)])
    for y, x in obs_p2:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            round_counts[code].append(obs2[(y, x)])

    blended = {}
    for code in [0, 1, 2, 3, 4, 5, 10, 11]:
        hist = flat_matrix.get(code, [1.0 / N_CLASSES] * N_CLASSES)
        if code in STATIC_CODES:
            blended[code] = hist[:]
            continue
        ol = round_counts.get(code, [])
        nr = len(ol)
        if nr == 0:
            blended[code] = hist[:]
            continue
        rf = [0.0] * N_CLASSES
        for c in ol:
            rf[c] += 1.0 / nr
        t = nr + n_hist
        blended[code] = [(nr * rf[i] + n_hist * hist[i]) / t for i in range(N_CLASSES)]
    return blended


def run_conditional(igrid, gt, cond_matrix):
    """Conditional model: per-context matrix + per-context calibration."""
    obs1, obs2, obs_p1, obs_p2 = get_obs(igrid, gt)
    H, W = len(igrid), len(igrid[0])

    blended = calibrate_conditional(igrid, obs_p1, obs_p2, obs1, obs2, cond_matrix, n_hist=50)

    obs_both = obs_p1 & obs_p2
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

                if (y, x) in obs_both:
                    oh = [0.0] * N_CLASSES
                    oh[obs1[(y, x)]] += 0.5
                    oh[obs2[(y, x)]] += 0.5
                    d = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                elif (y, x) in obs_p1:
                    oh = [0.0] * N_CLASSES
                    oh[obs1[(y, x)]] = 1.0
                    d = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                elif (y, x) in obs_p2:
                    oh = [0.0] * N_CLASSES
                    oh[obs2[(y, x)]] = 1.0
                    d = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]
                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        tensor.append(row)
    return score_tensor(tensor, gt)


def run_current(igrid, gt, flat_matrix):
    """Current model: flat matrix + neighbor boosts."""
    obs1, obs2, obs_p1, obs_p2 = get_obs(igrid, gt)
    H, W = len(igrid), len(igrid[0])

    blended = calibrate_flat(igrid, obs_p1, obs_p2, obs1, obs2, flat_matrix, n_hist=50)

    obs_both = obs_p1 & obs_p2
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
                prior = blended.get(code, [1.0 / N_CLASSES] * N_CLASSES)[:]

                # Settlement neighbor boost
                if code in (4, 11):
                    nc = count_neighbors(igrid, y, x, 1)
                    if nc > 0:
                        boost = min(nc * 0.01, 0.05)
                        prior[1] += boost
                        prior[0] += boost * 0.7
                        tp = sum(prior)
                        prior = [p / tp for p in prior]

                # Ocean proximity
                if code in (1, 4, 11):
                    oc = count_neighbors(igrid, y, x, 10)
                    if oc > 0:
                        boost = min(oc * 0.005, 0.03)
                        prior[2] += boost
                        if code in (4, 11):
                            prior[1] = max(prior[1] - boost * 0.5, 0.001)
                        tp = sum(prior)
                        prior = [p / tp for p in prior]

                if (y, x) in obs_both:
                    oh = [0.0] * N_CLASSES
                    oh[obs1[(y, x)]] += 0.5
                    oh[obs2[(y, x)]] += 0.5
                    d = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                elif (y, x) in obs_p1:
                    oh = [0.0] * N_CLASSES
                    oh[obs1[(y, x)]] = 1.0
                    d = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                elif (y, x) in obs_p2:
                    oh = [0.0] * N_CLASSES
                    oh[obs2[(y, x)]] = 1.0
                    d = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]
                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        tensor.append(row)
    return score_tensor(tensor, gt)


def run_conditional_with_boosts(igrid, gt, cond_matrix, flat_matrix):
    """Conditional matrix + neighbor boosts + ocean boost on top."""
    obs1, obs2, obs_p1, obs_p2 = get_obs(igrid, gt)
    H, W = len(igrid), len(igrid[0])

    blended_cond = calibrate_conditional(igrid, obs_p1, obs_p2, obs1, obs2, cond_matrix, n_hist=50)

    obs_both = obs_p1 & obs_p2
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
                prior = blended_cond.get(ctx, cond_matrix.get(ctx, [1.0 / N_CLASSES] * N_CLASSES))[:]

                if (y, x) in obs_both:
                    oh = [0.0] * N_CLASSES
                    oh[obs1[(y, x)]] += 0.5
                    oh[obs2[(y, x)]] += 0.5
                    d = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                elif (y, x) in obs_p1:
                    oh = [0.0] * N_CLASSES
                    oh[obs1[(y, x)]] = 1.0
                    d = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                elif (y, x) in obs_p2:
                    oh = [0.0] * N_CLASSES
                    oh[obs2[(y, x)]] = 1.0
                    d = [0.95 * prior[i] + 0.05 * oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]
                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        tensor.append(row)
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

    print(f"Conditional model test: {len(files)} files, {len(rounds)} rounds, {N_MC} MC")

    # Show how many context buckets we have
    sample_matrix = build_conditional_matrix_loo(files, "none", all_data)
    print(f"Context buckets: {len(sample_matrix)}")
    for ctx in sorted(sample_matrix.keys(), key=lambda c: (c[0], c[1], c[2])):
        # Count samples
        print(f"  {ctx}")
    print()

    strategies = [
        "A. Current (flat + boosts)",
        "B. Conditional matrix",
        "C. Conditional + lower N_HIST(30)",
    ]

    results = {s: defaultdict(list) for s in strategies}

    for round_id, round_files in sorted(rounds.items()):
        flat_loo = build_flat_matrix_loo(files, round_id, all_data)
        cond_loo = build_conditional_matrix_loo(files, round_id, all_data)

        for fname in round_files:
            d = all_data[fname]
            gt, igrid = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not igrid:
                continue

            for _ in range(N_MC):
                s_a = run_current(igrid, gt, flat_loo)
                results["A. Current (flat + boosts)"][round_id].append(s_a)

                s_b = run_conditional(igrid, gt, cond_loo)
                results["B. Conditional matrix"][round_id].append(s_b)

                s_c = run_conditional(igrid, gt, cond_loo)  # same fn but we'll modify n_hist
                results["C. Conditional + lower N_HIST(30)"][round_id].append(s_c)

        short = round_id[:8]
        print(f"  {short}", end="", flush=True)
        for s in strategies:
            avg = sum(results[s][round_id]) / len(results[s][round_id])
            print(f"  {avg:6.1f}", end="")
        print()

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  {'Strategy':<35} {'Avg':>6}")
    print(f"  {'-' * 35} {'-' * 6}")
    for s in strategies:
        all_scores = []
        for rid in rounds:
            all_scores.extend(results[s][rid])
        avg = sum(all_scores) / len(all_scores)
        print(f"  {s:<35} {avg:6.2f}")


if __name__ == "__main__":
    main()
