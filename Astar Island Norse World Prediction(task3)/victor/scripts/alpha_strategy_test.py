"""
alpha_strategy_test.py - Test adaptive alpha based on observation count/agreement.

Key insight: with alpha=0.05, observations barely matter. But two independent
observations that AGREE should warrant much higher confidence.

Strategies:
  A. Current: 5+5, flat alpha=0.05
  B. 9+1 fullgrid, flat alpha=0.05
  C. 9+1 fullgrid, adaptive alpha (higher when 2 obs agree)
  D. 9+1 fullgrid, high flat alpha=0.15
  E. 9+1 fullgrid, adaptive alpha + higher base (0.10)
  F. 5+5, adaptive alpha (same tiles as current but smarter alpha)

Run from victor/ folder:
    python -m scripts.alpha_strategy_test
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
N_MC = 8
random.seed(42)

ALL_ANCHORS = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
ANCHOR_CELLS = {a: set(_covered_cells(*a)) for a in ALL_ANCHORS}

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


def select_entropy_tiles(igrid, prior_tensor, observed_cells, n=1):
    """Select tiles covering highest-entropy unobserved cells."""
    H, W = len(igrid), len(igrid[0])
    remaining = {}
    for y in range(H):
        for x in range(W):
            if (y, x) not in observed_cells:
                remaining[(y, x)] = _entropy(prior_tensor[y][x])

    sel = []
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


def select_reobserve_tiles(igrid, prior_tensor, observed_cells, n=1):
    """Select tiles to RE-OBSERVE: cover highest-entropy ALREADY-observed cells.
    Goal: get 2nd independent observation for high-uncertainty cells."""
    H, W = len(igrid), len(igrid[0])
    remaining = {}
    for y in range(H):
        for x in range(W):
            if (y, x) in observed_cells:
                remaining[(y, x)] = _entropy(prior_tensor[y][x])

    sel = []
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


def calibrate_conditional(igrid, all_obs, cond_matrix, n_hist=50):
    """Calibrate conditional matrix. all_obs: {(y,x): [list of obs classes]}"""
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


def build_tensor(igrid, all_obs, cond_matrix, alpha_single=0.05, alpha_agree=0.05, alpha_disagree=0.05):
    """
    Build tensor. all_obs: {(y,x): [list of obs classes]}.
    alpha_single: alpha for cells with 1 observation
    alpha_agree: alpha for cells with 2+ observations that ALL agree
    alpha_disagree: alpha for cells with 2+ observations that disagree
    """
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

                    # Pick alpha based on observation count and agreement
                    if len(obs_vals) == 1:
                        alpha = alpha_single
                    elif len(set(obs_vals)) == 1:
                        # All observations agree
                        alpha = alpha_agree
                    else:
                        # Observations disagree — cell is genuinely stochastic
                        alpha = alpha_disagree

                    d = [(1 - alpha) * prior[i] + alpha * oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]
                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        tensor.append(row)
    return tensor


def get_observations(igrid, gt, p1_tiles, p2_tiles):
    """Run two phases with independent observations. Returns {(y,x): [obs_classes]}."""
    H, W = len(igrid), len(igrid[0])

    all_obs = {}

    # Phase 1
    obs_sample1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p1_cells = set()
    for a in p1_tiles:
        p1_cells |= ANCHOR_CELLS[a]
    for (y, x) in p1_cells:
        all_obs[(y, x)] = [obs_sample1[(y, x)]]

    # Phase 2 (independent draw)
    obs_sample2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p2_cells = set()
    for a in p2_tiles:
        p2_cells |= ANCHOR_CELLS[a]
    for (y, x) in p2_cells:
        if (y, x) in all_obs:
            all_obs[(y, x)].append(obs_sample2[(y, x)])
        else:
            all_obs[(y, x)] = [obs_sample2[(y, x)]]

    return all_obs


def build_prelim_tensor(igrid, all_obs, cond_matrix):
    """Quick preliminary tensor for entropy-based tile selection."""
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
                d = blended.get(ctx, cond_matrix.get(ctx, [1.0 / N_CLASSES] * N_CLASSES))[:]
                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        tensor.append(row)
    return tensor


def run_a(igrid, gt, cond_matrix):
    """Current: 5 sett + 5 spread, flat alpha=0.05."""
    p1 = select_settlement_tiles(igrid, 5)
    p2 = list(SPREAD_ANCHORS)
    all_obs = get_observations(igrid, gt, p1, p2)
    tensor = build_tensor(igrid, all_obs, cond_matrix, 0.05, 0.05, 0.05)
    return score_tensor(tensor, gt)


def run_b(igrid, gt, cond_matrix):
    """9+1 fullgrid + entropy, flat alpha=0.05."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p1_cells = set()
    for a in FULL_GRID_ANCHORS:
        p1_cells |= ANCHOR_CELLS[a]
    all_obs = {(y, x): [obs1[(y, x)]] for (y, x) in p1_cells}

    prelim = build_prelim_tensor(igrid, all_obs, cond_matrix)
    p2 = select_reobserve_tiles(igrid, prelim, p1_cells, n=1)

    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    for a in p2:
        for (y, x) in ANCHOR_CELLS[a]:
            if (y, x) in all_obs:
                all_obs[(y, x)].append(obs2[(y, x)])

    tensor = build_tensor(igrid, all_obs, cond_matrix, 0.05, 0.05, 0.05)
    return score_tensor(tensor, gt)


def run_c(igrid, gt, cond_matrix):
    """9+1 fullgrid + reobserve, ADAPTIVE alpha: agree=0.30, disagree=0.02, single=0.05."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p1_cells = set()
    for a in FULL_GRID_ANCHORS:
        p1_cells |= ANCHOR_CELLS[a]
    all_obs = {(y, x): [obs1[(y, x)]] for (y, x) in p1_cells}

    prelim = build_prelim_tensor(igrid, all_obs, cond_matrix)
    p2 = select_reobserve_tiles(igrid, prelim, p1_cells, n=1)

    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    for a in p2:
        for (y, x) in ANCHOR_CELLS[a]:
            if (y, x) in all_obs:
                all_obs[(y, x)].append(obs2[(y, x)])

    tensor = build_tensor(igrid, all_obs, cond_matrix,
                          alpha_single=0.05, alpha_agree=0.30, alpha_disagree=0.02)
    return score_tensor(tensor, gt)


def run_d(igrid, gt, cond_matrix):
    """9+1 fullgrid, higher flat alpha=0.15."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p1_cells = set()
    for a in FULL_GRID_ANCHORS:
        p1_cells |= ANCHOR_CELLS[a]
    all_obs = {(y, x): [obs1[(y, x)]] for (y, x) in p1_cells}

    prelim = build_prelim_tensor(igrid, all_obs, cond_matrix)
    p2 = select_reobserve_tiles(igrid, prelim, p1_cells, n=1)

    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    for a in p2:
        for (y, x) in ANCHOR_CELLS[a]:
            if (y, x) in all_obs:
                all_obs[(y, x)].append(obs2[(y, x)])

    tensor = build_tensor(igrid, all_obs, cond_matrix, 0.15, 0.15, 0.15)
    return score_tensor(tensor, gt)


def run_e(igrid, gt, cond_matrix):
    """9+1 fullgrid, adaptive: single=0.10, agree=0.40, disagree=0.03."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    p1_cells = set()
    for a in FULL_GRID_ANCHORS:
        p1_cells |= ANCHOR_CELLS[a]
    all_obs = {(y, x): [obs1[(y, x)]] for (y, x) in p1_cells}

    prelim = build_prelim_tensor(igrid, all_obs, cond_matrix)
    p2 = select_reobserve_tiles(igrid, prelim, p1_cells, n=1)

    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    for a in p2:
        for (y, x) in ANCHOR_CELLS[a]:
            if (y, x) in all_obs:
                all_obs[(y, x)].append(obs2[(y, x)])

    tensor = build_tensor(igrid, all_obs, cond_matrix,
                          alpha_single=0.10, alpha_agree=0.40, alpha_disagree=0.03)
    return score_tensor(tensor, gt)


def run_f(igrid, gt, cond_matrix):
    """5+5 sett+spread, adaptive alpha: agree=0.30, disagree=0.02, single=0.05."""
    p1 = select_settlement_tiles(igrid, 5)
    p2 = list(SPREAD_ANCHORS)
    all_obs = get_observations(igrid, gt, p1, p2)
    tensor = build_tensor(igrid, all_obs, cond_matrix,
                          alpha_single=0.05, alpha_agree=0.30, alpha_disagree=0.02)
    return score_tensor(tensor, gt)


def run_g(igrid, gt, cond_matrix):
    """5+5 sett+spread, adaptive: single=0.10, agree=0.40, disagree=0.03."""
    p1 = select_settlement_tiles(igrid, 5)
    p2 = list(SPREAD_ANCHORS)
    all_obs = get_observations(igrid, gt, p1, p2)
    tensor = build_tensor(igrid, all_obs, cond_matrix,
                          alpha_single=0.10, alpha_agree=0.40, alpha_disagree=0.03)
    return score_tensor(tensor, gt)


def run_h(igrid, gt, cond_matrix):
    """5 sett + 5 reobserve (targeted), adaptive: single=0.05, agree=0.30, disagree=0.02."""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

    p1 = select_settlement_tiles(igrid, 5)
    p1_cells = set()
    for a in p1:
        p1_cells |= ANCHOR_CELLS[a]
    all_obs = {(y, x): [obs1[(y, x)]] for (y, x) in p1_cells}

    # Phase 2: reobserve highest-entropy cells from phase 1
    prelim = build_prelim_tensor(igrid, all_obs, cond_matrix)
    p2 = select_reobserve_tiles(igrid, prelim, p1_cells, n=5)

    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    for a in p2:
        for (y, x) in ANCHOR_CELLS[a]:
            if (y, x) in all_obs:
                all_obs[(y, x)].append(obs2[(y, x)])
            else:
                all_obs[(y, x)] = [obs2[(y, x)]]

    tensor = build_tensor(igrid, all_obs, cond_matrix,
                          alpha_single=0.05, alpha_agree=0.30, alpha_disagree=0.02)
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

    print(f"Alpha strategy test: {len(files)} files, {len(rounds)} rounds, {N_MC} MC")
    print()

    strategies = [
        ("A. 5+5 flat a=.05", run_a),
        ("B. 9+1 flat a=.05", run_b),
        ("C. 9+1 adapt .05/.30/.02", run_c),
        ("D. 9+1 flat a=.15", run_d),
        ("E. 9+1 adapt .10/.40/.03", run_e),
        ("F. 5+5 adapt .05/.30/.02", run_f),
        ("G. 5+5 adapt .10/.40/.03", run_g),
        ("H. 5+5reobs adapt", run_h),
    ]

    # Print header
    print(f"  {'round':<12}", end="")
    for label, _ in strategies:
        short = label[3:17]
        print(f" {short:>14}", end="")
    print()
    print("  " + "-" * (12 + 15 * len(strategies)))

    results = {label: defaultdict(list) for label, _ in strategies}

    for round_id, round_files in sorted(rounds.items()):
        cond_loo = build_conditional_loo(files, round_id, all_data)

        for fname in round_files:
            d = all_data[fname]
            gt, igrid = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not igrid:
                continue

            for _ in range(N_MC):
                for label, fn in strategies:
                    score = fn(igrid, gt, cond_loo)
                    results[label][round_id].append(score)

        short = round_id[:8]
        print(f"  {short:<12}", end="", flush=True)
        for label, _ in strategies:
            avg = sum(results[label][round_id]) / len(results[label][round_id])
            print(f" {avg:14.1f}", end="")
        print()

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  {'Strategy':<30} {'Avg':>6}  {'vs A':>6}")
    print(f"  {'-' * 30} {'-' * 6}  {'-' * 6}")
    baseline = None
    for label, _ in strategies:
        all_scores = []
        for rid in rounds:
            all_scores.extend(results[label][rid])
        avg = sum(all_scores) / len(all_scores)
        if baseline is None:
            baseline = avg
        delta = avg - baseline
        print(f"  {label:<30} {avg:6.2f}  {delta:+6.2f}")


if __name__ == "__main__":
    main()
