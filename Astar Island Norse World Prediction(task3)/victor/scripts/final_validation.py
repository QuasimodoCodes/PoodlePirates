"""
final_validation.py - Validate the full pipeline improvements.

Compare:
  A. Old model: flat matrix + hand-tuned boosts, no temperature
  B. New model: conditional matrix + temperature=1.10

Run from victor/ folder:
    python -m scripts.final_validation
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N_CLASSES = 6
FLOOR_STATIC = 1e-5
FLOOR_DYN = 0.005
ALPHA = 0.05
N_MC = 10  # more MC for final validation
random.seed(42)

ALL_ANCHORS = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
ANCHOR_CELLS = {a: set(_covered_cells(*a)) for a in ALL_ANCHORS}

TERRAIN_FLOOR = {1: 0.008, 2: 0.008, 3: 0.006, 4: 0.003, 11: 0.004, 0: 0.005}


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
            if dy == 0 and dx == 0: continue
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

def apply_temperature(dist, temp):
    if temp == 1.0: return dist
    log_dist = [math.log(max(p, 1e-12)) / temp for p in dist]
    max_log = max(log_dist)
    exp_dist = [math.exp(v - max_log) for v in log_dist]
    total = sum(exp_dist)
    return [v / total for v in exp_dist]


# ── Matrix builders ──

def build_flat_loo(all_files, exclude_id, all_data):
    acc = defaultdict(list)
    for f in all_files:
        if f.split("_seed")[0] == exclude_id: continue
        d = all_data[f]
        gt, ig = d.get("ground_truth"), d.get("initial_grid")
        if not gt or not ig: continue
        for y in range(len(ig)):
            for x in range(len(ig[0])):
                c = ig[y][x]
                if c not in STATIC_CODES:
                    acc[c].append(gt[y][x])
    m = {}
    for c, s in acc.items():
        n = len(s)
        m[c] = [sum(v[i] for v in s)/n for i in range(N_CLASSES)]
    for c in [0,1,2,3,4,5,10,11]:
        if c not in m: m[c] = [1/N_CLASSES]*N_CLASSES
    return m

def build_conditional_loo(all_files, exclude_id, all_data):
    acc = defaultdict(list)
    for f in all_files:
        if f.split("_seed")[0] == exclude_id: continue
        d = all_data[f]
        gt, ig = d.get("ground_truth"), d.get("initial_grid")
        if not gt or not ig: continue
        for y in range(len(ig)):
            for x in range(len(ig[0])):
                c = ig[y][x]
                if c not in STATIC_CODES:
                    ctx = cell_context(ig, y, x)
                    acc[ctx].append(gt[y][x])
    m = {}
    for ctx, s in acc.items():
        n = len(s)
        m[ctx] = [sum(v[i] for v in s)/n for i in range(N_CLASSES)]
    return m


# ── Observation + calibration ──

def select_settlement_tiles(igrid, n=5):
    setts = {(y, x) for y in range(len(igrid)) for x in range(len(igrid[0])) if igrid[y][x] == 1}
    if not setts: return SPREAD_ANCHORS[:n]
    covered, sel = set(), []
    for _ in range(n):
        best, bc = None, -1
        for a in ALL_ANCHORS:
            c = len((ANCHOR_CELLS[a] & setts) - covered)
            if c > bc: bc, best = c, a
        if not best or bc <= 0: break
        sel.append(best)
        covered |= (ANCHOR_CELLS[best] & setts)
    return sel

def get_observations(igrid, gt):
    H, W = len(igrid), len(igrid[0])
    all_obs = {}
    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    for a in select_settlement_tiles(igrid, 5):
        for (y, x) in ANCHOR_CELLS[a]:
            all_obs[(y, x)] = [obs1[(y, x)]]
    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    for a in SPREAD_ANCHORS:
        for (y, x) in ANCHOR_CELLS[a]:
            if (y, x) in all_obs: all_obs[(y, x)].append(obs2[(y, x)])
            else: all_obs[(y, x)] = [obs2[(y, x)]]
    return all_obs

def calibrate_flat(igrid, all_obs, matrix, n_hist=50):
    rc = defaultdict(list)
    for (y, x), vals in all_obs.items():
        c = igrid[y][x]
        if c not in STATIC_CODES:
            for v in vals: rc[c].append(v)
    bl = {}
    for c in [0,1,2,3,4,5,10,11]:
        h = matrix.get(c, [1/N_CLASSES]*N_CLASSES)
        if c in STATIC_CODES: bl[c] = h[:]; continue
        ol = rc.get(c, [])
        nr = len(ol)
        if nr == 0: bl[c] = h[:]; continue
        rf = [0.0]*N_CLASSES
        for v in ol: rf[v] += 1.0/nr
        t = nr + n_hist
        bl[c] = [(nr*rf[i]+n_hist*h[i])/t for i in range(N_CLASSES)]
    return bl

def calibrate_conditional(igrid, all_obs, matrix, n_hist=50):
    rc = defaultdict(list)
    for (y, x), vals in all_obs.items():
        c = igrid[y][x]
        if c not in STATIC_CODES:
            ctx = cell_context(igrid, y, x)
            for v in vals: rc[ctx].append(v)
    bl = {}
    for ctx in set(list(matrix.keys()) + list(rc.keys())):
        h = matrix.get(ctx, [1/N_CLASSES]*N_CLASSES)
        ol = rc.get(ctx, [])
        nr = len(ol)
        if nr == 0: bl[ctx] = h[:]; continue
        rf = [0.0]*N_CLASSES
        for v in ol: rf[v] += 1.0/nr
        t = nr + n_hist
        bl[ctx] = [(nr*rf[i]+n_hist*h[i])/t for i in range(N_CLASSES)]
    return bl


# ── Model A: Old (flat + boosts, no temp) ──

def run_old(igrid, gt, flat_matrix):
    H, W = len(igrid), len(igrid[0])
    all_obs = get_observations(igrid, gt)
    bl = calibrate_flat(igrid, all_obs, flat_matrix)

    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                d = [FLOOR_STATIC]*N_CLASSES
                d[pc] = 1.0 - FLOOR_STATIC*5
            else:
                prior = bl.get(code, [1/N_CLASSES]*N_CLASSES)[:]
                # Settlement neighbor boost
                if code in (4, 11):
                    nc = count_neighbors(igrid, y, x, 1)
                    if nc > 0:
                        boost = min(nc * 0.01, 0.05)
                        prior[1] += boost
                        prior[0] += boost * 0.7
                        tp = sum(prior)
                        prior = [p/tp for p in prior]
                # Ocean boost
                if code in (1, 4, 11):
                    oc = count_neighbors(igrid, y, x, 10)
                    if oc > 0:
                        boost = min(oc * 0.005, 0.03)
                        prior[2] += boost
                        if code in (4, 11):
                            prior[1] = max(prior[1] - boost*0.5, 0.001)
                        tp = sum(prior)
                        prior = [p/tp for p in prior]
                if (y, x) in all_obs:
                    vals = all_obs[(y, x)]
                    oh = [0.0]*N_CLASSES
                    for v in vals: oh[v] += 1.0/len(vals)
                    d = [(1-ALPHA)*prior[i]+ALPHA*oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]
                floor = TERRAIN_FLOOR.get(code, FLOOR_DYN)
                d = [max(v, floor) for v in d]
            t = sum(d)
            row.append([v/t for v in d])
        tensor.append(row)
    return score_tensor(tensor, gt)


# ── Model B: New (conditional + temp=1.10) ──

def run_new(igrid, gt, cond_matrix):
    H, W = len(igrid), len(igrid[0])
    all_obs = get_observations(igrid, gt)
    bl = calibrate_conditional(igrid, all_obs, cond_matrix)

    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                d = [FLOOR_STATIC]*N_CLASSES
                d[pc] = 1.0 - FLOOR_STATIC*5
            else:
                ctx = cell_context(igrid, y, x)
                prior = bl.get(ctx, cond_matrix.get(ctx, [1/N_CLASSES]*N_CLASSES))[:]
                if (y, x) in all_obs:
                    vals = all_obs[(y, x)]
                    oh = [0.0]*N_CLASSES
                    for v in vals: oh[v] += 1.0/len(vals)
                    d = [(1-ALPHA)*prior[i]+ALPHA*oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]
                d = apply_temperature(d, 1.10)
                floor = TERRAIN_FLOOR.get(code, FLOOR_DYN)
                d = [max(v, floor) for v in d]
            t = sum(d)
            row.append([v/t for v in d])
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

    print(f"Final validation: {len(files)} files, {len(rounds)} rounds, {N_MC} MC trials")
    print(f"  Old: flat matrix + hand-tuned boosts, alpha=0.05, no temp")
    print(f"  New: conditional matrix + temp=1.10, alpha=0.05")
    print()
    print(f"  {'Round':<12} {'Old':>8} {'New':>8} {'Delta':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

    old_all, new_all = [], []

    for round_id, round_files in sorted(rounds.items()):
        flat_loo = build_flat_loo(files, round_id, all_data)
        cond_loo = build_conditional_loo(files, round_id, all_data)

        old_scores, new_scores = [], []
        for fname in round_files:
            d = all_data[fname]
            gt, igrid = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not igrid: continue
            for _ in range(N_MC):
                old_scores.append(run_old(igrid, gt, flat_loo))
                new_scores.append(run_new(igrid, gt, cond_loo))

        old_avg = sum(old_scores) / len(old_scores)
        new_avg = sum(new_scores) / len(new_scores)
        delta = new_avg - old_avg
        old_all.extend(old_scores)
        new_all.extend(new_scores)

        marker = " <<<" if abs(delta) > 1.0 else ""
        print(f"  {round_id[:8]:<12} {old_avg:8.2f} {new_avg:8.2f} {delta:+8.2f}{marker}")

    old_total = sum(old_all) / len(old_all)
    new_total = sum(new_all) / len(new_all)
    delta_total = new_total - old_total

    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'AVERAGE':<12} {old_total:8.2f} {new_total:8.2f} {delta_total:+8.2f}")
    print()

    # Worst round comparison
    old_by_round = defaultdict(list)
    new_by_round = defaultdict(list)
    for round_id in rounds:
        for fname in rounds[round_id]:
            d = all_data[fname]
            gt, igrid = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not igrid: continue

    print(f"  Improvement: {delta_total:+.2f} pts average across {len(rounds)} rounds")
    if delta_total > 0:
        print(f"  New model is better!")
    else:
        print(f"  Old model is better — keeping current pipeline.")


if __name__ == "__main__":
    main()
