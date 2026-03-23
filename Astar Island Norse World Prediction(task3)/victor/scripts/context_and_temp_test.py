"""
context_and_temp_test.py - Test finer context buckets + temperature scaling.

Finer contexts: add forest neighbor count, ruin neighbor count, edge proximity.
Temperature: soften/sharpen the predicted distributions to hedge against uncertainty.
Per-seed calibration: calibrate each seed independently instead of pooling all 5.

Run from victor/ folder:
    python -m scripts.context_and_temp_test
"""

import os, sys, json, math, random
from collections import defaultdict
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N_CLASSES = 6
FLOOR_STATIC = 1e-5
FLOOR_DYN = 0.005
ALPHA = 0.05
N_MC = 5
random.seed(42)

ALL_ANCHORS = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
ANCHOR_CELLS = {a: set(_covered_cells(*a)) for a in ALL_ANCHORS}


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


# ── Context functions ──────────────────────────

def ctx_basic(igrid, y, x):
    """Current 3-feature context: (code, sett_bin, ocean_bin)."""
    code = igrid[y][x]
    sett_n = count_neighbors(igrid, y, x, 1)
    ocean_n = count_neighbors(igrid, y, x, 10)
    sett_bin = "sett_hi" if sett_n >= 3 else ("sett_lo" if sett_n >= 1 else "sett_no")
    ocean_bin = "ocean" if ocean_n >= 1 else "inland"
    return (code, sett_bin, ocean_bin)


def ctx_with_forest(igrid, y, x):
    """Add forest neighbor density."""
    code = igrid[y][x]
    sett_n = count_neighbors(igrid, y, x, 1)
    ocean_n = count_neighbors(igrid, y, x, 10)
    forest_n = count_neighbors(igrid, y, x, 4)
    sett_bin = "sett_hi" if sett_n >= 3 else ("sett_lo" if sett_n >= 1 else "sett_no")
    ocean_bin = "ocean" if ocean_n >= 1 else "inland"
    forest_bin = "for_hi" if forest_n >= 5 else ("for_lo" if forest_n >= 1 else "for_no")
    return (code, sett_bin, ocean_bin, forest_bin)


def ctx_with_edge(igrid, y, x):
    """Add edge proximity (corner/edge/interior)."""
    code = igrid[y][x]
    H, W = len(igrid), len(igrid[0])
    sett_n = count_neighbors(igrid, y, x, 1)
    ocean_n = count_neighbors(igrid, y, x, 10)
    sett_bin = "sett_hi" if sett_n >= 3 else ("sett_lo" if sett_n >= 1 else "sett_no")
    ocean_bin = "ocean" if ocean_n >= 1 else "inland"
    edge_dist = min(y, x, H - 1 - y, W - 1 - x)
    edge_bin = "corner" if edge_dist <= 2 else ("edge" if edge_dist <= 5 else "inner")
    return (code, sett_bin, ocean_bin, edge_bin)


def ctx_with_plains(igrid, y, x):
    """Add plains neighbor density (since plains = 66% of score weight)."""
    code = igrid[y][x]
    sett_n = count_neighbors(igrid, y, x, 1)
    ocean_n = count_neighbors(igrid, y, x, 10)
    plains_n = count_neighbors(igrid, y, x, 11)
    sett_bin = "sett_hi" if sett_n >= 3 else ("sett_lo" if sett_n >= 1 else "sett_no")
    ocean_bin = "ocean" if ocean_n >= 1 else "inland"
    plains_bin = "pln_hi" if plains_n >= 8 else ("pln_lo" if plains_n >= 1 else "pln_no")
    return (code, sett_bin, ocean_bin, plains_bin)


# ── Build matrix for a given context function ──

def build_matrix(all_files, exclude_round_id, all_data, ctx_fn):
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
                    ctx = ctx_fn(igrid, y, x)
                    accum[ctx].append(gt[y][x])
    matrix = {}
    for ctx, samples in accum.items():
        n = len(samples)
        if n >= 5:  # minimum sample threshold
            matrix[ctx] = [sum(s[i] for s in samples) / n for i in range(N_CLASSES)]
    return matrix


# ── Standard helpers ──

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


def get_observations(igrid, gt):
    H, W = len(igrid), len(igrid[0])
    all_obs = {}
    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p1 = select_settlement_tiles(igrid, 5)
    for a in p1:
        for (y, x) in ANCHOR_CELLS[a]:
            all_obs[(y, x)] = [obs1[(y, x)]]
    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    for a in SPREAD_ANCHORS:
        for (y, x) in ANCHOR_CELLS[a]:
            if (y, x) in all_obs:
                all_obs[(y, x)].append(obs2[(y, x)])
            else:
                all_obs[(y, x)] = [obs2[(y, x)]]
    return all_obs


def calibrate(igrid, all_obs, matrix, ctx_fn, n_hist=50):
    round_counts = defaultdict(list)
    for (y, x), vals in all_obs.items():
        code = igrid[y][x]
        if code in STATIC_CODES:
            continue
        ctx = ctx_fn(igrid, y, x)
        for v in vals:
            round_counts[ctx].append(v)
    blended = {}
    for ctx in set(list(matrix.keys()) + list(round_counts.keys())):
        hist = matrix.get(ctx, [1.0 / N_CLASSES] * N_CLASSES)
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


def apply_temperature(dist, temp):
    """Apply temperature scaling: temp<1 sharpens, temp>1 softens."""
    if temp == 1.0:
        return dist
    log_dist = [math.log(max(p, 1e-12)) / temp for p in dist]
    max_log = max(log_dist)
    exp_dist = [math.exp(v - max_log) for v in log_dist]
    total = sum(exp_dist)
    return [v / total for v in exp_dist]


def build_tensor(igrid, all_obs, matrix, ctx_fn, temperature=1.0):
    H, W = len(igrid), len(igrid[0])
    blended = calibrate(igrid, all_obs, matrix, ctx_fn)

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
                ctx = ctx_fn(igrid, y, x)
                prior = blended.get(ctx, matrix.get(ctx, [1.0 / N_CLASSES] * N_CLASSES))[:]

                if (y, x) in all_obs:
                    obs_vals = all_obs[(y, x)]
                    oh = [0.0] * N_CLASSES
                    for v in obs_vals:
                        oh[v] += 1.0 / len(obs_vals)
                    d = [(1 - ALPHA) * prior[i] + ALPHA * oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]

                # Apply temperature before flooring
                if temperature != 1.0:
                    d = apply_temperature(d, temperature)

                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        tensor.append(row)
    return tensor


def run_test(igrid, gt, matrix, ctx_fn, temperature=1.0):
    all_obs = get_observations(igrid, gt)
    tensor = build_tensor(igrid, all_obs, matrix, ctx_fn, temperature)
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

    print(f"Context & temperature test: {len(files)} files, {len(rounds)} rounds, {N_MC} MC")

    # Count buckets for each context function
    for name, fn in [("basic", ctx_basic), ("+forest", ctx_with_forest),
                     ("+edge", ctx_with_edge), ("+plains", ctx_with_plains)]:
        m = build_matrix(files, "none", all_data, fn)
        print(f"  {name}: {len(m)} buckets")
    print()

    strategies = [
        ("A. basic ctx", ctx_basic, 1.0),
        ("B. +forest ctx", ctx_with_forest, 1.0),
        ("C. +edge ctx", ctx_with_edge, 1.0),
        ("D. +plains ctx", ctx_with_plains, 1.0),
        ("E. basic temp=0.90", ctx_basic, 0.90),
        ("F. basic temp=0.95", ctx_basic, 0.95),
        ("G. basic temp=1.05", ctx_basic, 1.05),
        ("H. basic temp=1.10", ctx_basic, 1.10),
        ("I. basic temp=1.20", ctx_basic, 1.20),
        ("J. +forest temp=0.95", ctx_with_forest, 0.95),
        ("K. +plains temp=1.05", ctx_with_plains, 1.05),
    ]

    results = {label: defaultdict(list) for label, _, _ in strategies}

    for round_id, round_files in sorted(rounds.items()):
        matrices = {}
        for label, ctx_fn, temp in strategies:
            key = id(ctx_fn)
            if key not in matrices:
                matrices[key] = build_matrix(files, round_id, all_data, ctx_fn)

        for fname in round_files:
            d = all_data[fname]
            gt, igrid = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not igrid:
                continue

            for _ in range(N_MC):
                for label, ctx_fn, temp in strategies:
                    matrix = matrices[id(ctx_fn)]
                    score = run_test(igrid, gt, matrix, ctx_fn, temp)
                    results[label][round_id].append(score)

        short = round_id[:8]
        print(f"  {short}", end="", flush=True)
        for label, _, _ in strategies:
            avg = sum(results[label][round_id]) / len(results[label][round_id])
            print(f"  {avg:5.1f}", end="")
        print()

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    baseline = None
    print(f"\n  {'Strategy':<28} {'Avg':>6}  {'vs A':>6}")
    print(f"  {'-' * 28} {'-' * 6}  {'-' * 6}")
    for label, _, _ in strategies:
        all_scores = []
        for rid in rounds:
            all_scores.extend(results[label][rid])
        avg = sum(all_scores) / len(all_scores)
        if baseline is None:
            baseline = avg
        delta = avg - baseline
        print(f"  {label:<28} {avg:6.2f}  {delta:+6.2f}")


if __name__ == "__main__":
    main()
