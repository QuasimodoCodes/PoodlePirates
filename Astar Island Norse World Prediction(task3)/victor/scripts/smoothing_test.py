"""
smoothing_test.py - Test spatial smoothing: propagate observation info to neighbors.

Idea: after normal prediction, for each unobserved dynamic cell, look at nearby
observed cells with the SAME initial terrain code. If their observations differ
from the prior, nudge the unobserved cell toward the local observation pattern.

This lets observations "leak" information to nearby unobserved cells without
increasing alpha (which adds noise to observed cells).

Strategies:
  A. Current: no smoothing
  B. Smooth r=3, weight=0.03
  C. Smooth r=5, weight=0.03
  D. Smooth r=3, weight=0.05
  E. Smooth r=5, weight=0.05
  F. Smooth r=3, weight=0.10
  G. Context-smooth: only smooth from same-context cells

Run from victor/ folder:
    python -m scripts.smoothing_test
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


def calibrate_conditional(igrid, all_obs, cond_matrix, n_hist=50):
    round_counts = defaultdict(list)
    for (y, x), vals in all_obs.items():
        code = igrid[y][x]
        if code in STATIC_CODES:
            continue
        ctx = cell_context(igrid, y, x)
        for v in vals:
            round_counts[ctx].append(v)
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


def get_observations(igrid, gt):
    """Standard 5+5 observations."""
    H, W = len(igrid), len(igrid[0])
    all_obs = {}

    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p1 = select_settlement_tiles(igrid, 5)
    p1_cells = set()
    for a in p1:
        p1_cells |= ANCHOR_CELLS[a]
    for (y, x) in p1_cells:
        all_obs[(y, x)] = [obs1[(y, x)]]

    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p2_cells = set()
    for a in SPREAD_ANCHORS:
        p2_cells |= ANCHOR_CELLS[a]
    for (y, x) in p2_cells:
        if (y, x) in all_obs:
            all_obs[(y, x)].append(obs2[(y, x)])
        else:
            all_obs[(y, x)] = [obs2[(y, x)]]

    return all_obs


def build_base_tensor(igrid, all_obs, cond_matrix):
    """Build tensor WITHOUT smoothing (baseline)."""
    H, W = len(igrid), len(igrid[0])
    blended = calibrate_conditional(igrid, all_obs, cond_matrix)

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

                if (y, x) in all_obs:
                    obs_vals = all_obs[(y, x)]
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


def apply_spatial_smoothing(igrid, tensor, all_obs, radius=3, weight=0.03, context_match=False):
    """
    For each UNOBSERVED dynamic cell, find nearby OBSERVED cells with the same
    initial terrain code. Build a local observation distribution and blend it
    into the cell's prediction.

    If context_match=True, only smooth from cells with the same full context tuple.

    weight: how much to trust the local observation pattern (like a local alpha).
    """
    H, W = len(igrid), len(igrid[0])

    # Build observed outcome map: (y,x) -> observed class (use mode of observations)
    obs_class = {}
    for (y, x), vals in all_obs.items():
        # Use the most common observation
        counts = [0] * N_CLASSES
        for v in vals:
            counts[v] += 1
        obs_class[(y, x)] = counts.index(max(counts))

    # Build index: for each initial code, list of observed (y,x) positions
    obs_by_code = defaultdict(list)  # code -> [(y, x, obs_cls)]
    obs_by_ctx = defaultdict(list)   # ctx -> [(y, x, obs_cls)]
    for (y, x), cls in obs_class.items():
        code = igrid[y][x]
        if code not in STATIC_CODES:
            obs_by_code[code].append((y, x, cls))
            if context_match:
                ctx = cell_context(igrid, y, x)
                obs_by_ctx[ctx].append((y, x, cls))

    smoothed = [row[:] for row in tensor]  # shallow copy rows

    for y in range(H):
        new_row = []
        for x in range(W):
            code = igrid[y][x]

            # Only smooth unobserved dynamic cells
            if code in STATIC_CODES or (y, x) in all_obs:
                new_row.append(tensor[y][x])
                continue

            # Find nearby observed cells with same terrain code (or context)
            if context_match:
                ctx = cell_context(igrid, y, x)
                candidates = obs_by_ctx.get(ctx, [])
            else:
                candidates = obs_by_code.get(code, [])

            # Filter by radius and weight by inverse distance
            local_dist = [0.0] * N_CLASSES
            total_w = 0.0
            for oy, ox, cls in candidates:
                dist = abs(y - oy) + abs(x - ox)  # Manhattan distance
                if dist <= radius:
                    w = 1.0 / max(dist, 1)  # inverse distance weight
                    local_dist[cls] += w
                    total_w += w

            if total_w < 1e-12:
                new_row.append(tensor[y][x])
                continue

            # Normalize local observation distribution
            local_dist = [v / total_w for v in local_dist]

            # Blend: (1 - weight) * prior + weight * local_obs
            d = tensor[y][x]
            blended = [(1 - weight) * d[i] + weight * local_dist[i] for i in range(N_CLASSES)]
            blended = [max(v, FLOOR_DYN) for v in blended]
            t = sum(blended)
            new_row.append([v / t for v in blended])

        smoothed[y] = new_row

    return smoothed


def run_test(igrid, gt, cond_matrix, radius=0, weight=0.0, context_match=False):
    all_obs = get_observations(igrid, gt)
    tensor = build_base_tensor(igrid, all_obs, cond_matrix)

    if radius > 0 and weight > 0:
        tensor = apply_spatial_smoothing(igrid, tensor, all_obs, radius, weight, context_match)

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

    print(f"Spatial smoothing test: {len(files)} files, {len(rounds)} rounds, {N_MC} MC")
    print()

    strategies = [
        ("A. No smooth (baseline)", 0, 0.0, False),
        ("B. r=3  w=0.03", 3, 0.03, False),
        ("C. r=5  w=0.03", 5, 0.03, False),
        ("D. r=3  w=0.05", 3, 0.05, False),
        ("E. r=5  w=0.05", 5, 0.05, False),
        ("F. r=3  w=0.10", 3, 0.10, False),
        ("G. r=5  w=0.10", 5, 0.10, False),
        ("H. r=3  w=0.03 ctx", 3, 0.03, True),
        ("I. r=5  w=0.05 ctx", 5, 0.05, True),
        ("J. r=8  w=0.03", 8, 0.03, False),
        ("K. r=8  w=0.05", 8, 0.05, False),
    ]

    # Print header
    print(f"  {'round':<12}", end="")
    for label, _, _, _ in strategies:
        short = label[3:18]
        print(f" {short:>15}", end="")
    print()
    print("  " + "-" * (12 + 16 * len(strategies)))

    results = {label: defaultdict(list) for label, _, _, _ in strategies}

    for round_id, round_files in sorted(rounds.items()):
        cond_loo = build_conditional_loo(files, round_id, all_data)

        for fname in round_files:
            d = all_data[fname]
            gt, igrid = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not igrid:
                continue

            for _ in range(N_MC):
                for label, radius, weight, ctx in strategies:
                    score = run_test(igrid, gt, cond_loo, radius, weight, ctx)
                    results[label][round_id].append(score)

        short = round_id[:8]
        print(f"  {short:<12}", end="", flush=True)
        for label, _, _, _ in strategies:
            avg = sum(results[label][round_id]) / len(results[label][round_id])
            print(f" {avg:15.1f}", end="")
        print()

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    baseline = None
    print(f"\n  {'Strategy':<28} {'Avg':>6}  {'vs A':>6}")
    print(f"  {'-' * 28} {'-' * 6}  {'-' * 6}")
    for label, _, _, _ in strategies:
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
