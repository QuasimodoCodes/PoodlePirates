"""
adaptive_nhist_test.py - Test per-bucket adaptive N_HIST.

When round observations diverge heavily from historical matrix for a specific
context bucket, lower N_HIST for that bucket to trust round data more.

This is different from our old global adaptive N_HIST (which hurt us) because:
- It's PER CONTEXT BUCKET, not global
- Only surprised buckets get lower N_HIST
- Normal buckets keep the conservative N_HIST=50

Run from victor/ folder:
    python -m scripts.adaptive_nhist_test
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
N_MC = 8
random.seed(42)

ALL_ANCHORS = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
ANCHOR_CELLS = {a: set(_covered_cells(*a)) for a in ALL_ANCHORS}


def kl_divergence(p, q):
    return sum(pi * math.log(pi / max(qi, 1e-12)) for pi, qi in zip(p, q) if pi > 1e-12)

def score_tensor(pred, gt):
    wkl = te = 0.0
    for y in range(40):
        for x in range(40):
            e = _entropy(gt[y][x])
            wkl += e * kl_divergence(gt[y][x], pred[y][x])
            te += e
    return max(0, min(100, 100 * math.exp(-3 * (wkl / te)))) if te > 1e-12 else 100

def sample_observation(gt_dist):
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(gt_dist):
        cumsum += p
        if r < cumsum: return i
    return 5

def count_neighbors(igrid, y, x, target_code, radius=2):
    c = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < 40 and 0 <= nx < 40 and igrid[ny][nx] == target_code: c += 1
    return c

def cell_context(igrid, y, x):
    code = igrid[y][x]
    sn = count_neighbors(igrid, y, x, 1)
    on = count_neighbors(igrid, y, x, 10)
    sb = "sett_hi" if sn >= 3 else ("sett_lo" if sn >= 1 else "sett_no")
    ob = "ocean" if on >= 1 else "inland"
    return (code, sb, ob)

def temp_scale(d, t):
    ld = [math.log(max(p, 1e-12)) / t for p in d]
    mx = max(ld)
    ed = [math.exp(v - mx) for v in ld]
    s = sum(ed)
    return [v / s for v in ed]


def build_conditional_loo(all_files, exclude_id, all_data):
    acc = defaultdict(list)
    for f in all_files:
        if f.split("_seed")[0] == exclude_id: continue
        d = all_data[f]
        gt, ig = d.get("ground_truth"), d.get("initial_grid")
        if not gt or not ig: continue
        for y in range(40):
            for x in range(40):
                c = ig[y][x]
                if c not in STATIC_CODES:
                    ctx = cell_context(ig, y, x)
                    acc[ctx].append(gt[y][x])
    m = {}
    for ctx, samples in acc.items():
        n = len(samples)
        m[ctx] = [sum(s[i] for s in samples) / n for i in range(N_CLASSES)]
    return m


def select_settlement_tiles(igrid, n=5):
    setts = {(y, x) for y in range(40) for x in range(40) if igrid[y][x] == 1}
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
    all_obs = {}
    obs1 = {(y, x): sample_observation(gt[y][x]) for y in range(40) for x in range(40)}
    for a in select_settlement_tiles(igrid, 5):
        for yx in ANCHOR_CELLS[a]:
            all_obs[yx] = [obs1[yx]]
    obs2 = {(y, x): sample_observation(gt[y][x]) for y in range(40) for x in range(40)}
    for a in SPREAD_ANCHORS:
        for yx in ANCHOR_CELLS[a]:
            if yx in all_obs: all_obs[yx].append(obs2[yx])
            else: all_obs[yx] = [obs2[yx]]
    return all_obs


def compute_surprise(igrid, all_obs, cond_matrix):
    """Compute per-bucket surprise: symmetric KL between round obs and historical."""
    round_counts = defaultdict(list)
    for (y, x), vals in all_obs.items():
        code = igrid[y][x]
        if code in STATIC_CODES: continue
        ctx = cell_context(igrid, y, x)
        for v in vals:
            round_counts[ctx].append(v)

    surprise = {}
    for ctx, obs_list in round_counts.items():
        if len(obs_list) < 3:  # need minimum samples
            surprise[ctx] = 0.0
            continue
        hist = cond_matrix.get(ctx, [1 / N_CLASSES] * N_CLASSES)
        # Round frequency
        rf = [0.0] * N_CLASSES
        for v in obs_list:
            rf[v] += 1.0 / len(obs_list)
        # Symmetric KL as surprise measure
        kl_fwd = sum(rf[i] * math.log(max(rf[i], 1e-12) / max(hist[i], 1e-12))
                      for i in range(N_CLASSES) if rf[i] > 1e-12)
        kl_rev = sum(hist[i] * math.log(max(hist[i], 1e-12) / max(rf[i], 1e-12))
                      for i in range(N_CLASSES) if hist[i] > 1e-12)
        surprise[ctx] = (kl_fwd + kl_rev) / 2

    return surprise, round_counts


def calibrate_adaptive(igrid, all_obs, cond_matrix, n_hist_normal=50,
                       n_hist_surprised=15, surprise_threshold=0.3):
    """Calibrate with per-bucket adaptive N_HIST."""
    surprise, round_counts = compute_surprise(igrid, all_obs, cond_matrix)

    blended = {}
    n_surprised = 0
    for ctx in set(list(cond_matrix.keys()) + list(round_counts.keys())):
        hist = cond_matrix.get(ctx, [1 / N_CLASSES] * N_CLASSES)
        ol = round_counts.get(ctx, [])
        nr = len(ol)
        if nr == 0:
            blended[ctx] = hist[:]
            continue

        # Adaptive N_HIST based on surprise
        s = surprise.get(ctx, 0.0)
        if s > surprise_threshold and nr >= 5:
            n_hist = n_hist_surprised
            n_surprised += 1
        else:
            n_hist = n_hist_normal

        rf = [0.0] * N_CLASSES
        for v in ol:
            rf[v] += 1.0 / nr
        t = nr + n_hist
        blended[ctx] = [(nr * rf[i] + n_hist * hist[i]) / t for i in range(N_CLASSES)]

    return blended, n_surprised


def calibrate_fixed(igrid, all_obs, cond_matrix, n_hist=50):
    """Standard fixed N_HIST calibration."""
    round_counts = defaultdict(list)
    for (y, x), vals in all_obs.items():
        code = igrid[y][x]
        if code in STATIC_CODES: continue
        ctx = cell_context(igrid, y, x)
        for v in vals:
            round_counts[ctx].append(v)

    blended = {}
    for ctx in set(list(cond_matrix.keys()) + list(round_counts.keys())):
        hist = cond_matrix.get(ctx, [1 / N_CLASSES] * N_CLASSES)
        ol = round_counts.get(ctx, [])
        nr = len(ol)
        if nr == 0:
            blended[ctx] = hist[:]
            continue
        rf = [0.0] * N_CLASSES
        for v in ol:
            rf[v] += 1.0 / nr
        t = nr + n_hist
        blended[ctx] = [(nr * rf[i] + n_hist * hist[i]) / t for i in range(N_CLASSES)]
    return blended


def build_tensor(igrid, all_obs, blended, cond_matrix, temperature=1.10):
    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                d = [FLOOR_STATIC] * N_CLASSES
                d[pc] = 1.0 - FLOOR_STATIC * 5
            else:
                ctx = cell_context(igrid, y, x)
                prior = blended.get(ctx, cond_matrix.get(ctx, [1 / N_CLASSES] * N_CLASSES))[:]
                if (y, x) in all_obs:
                    vals = all_obs[(y, x)]
                    oh = [0.0] * N_CLASSES
                    for v in vals:
                        oh[v] += 1.0 / len(vals)
                    d = [(1 - ALPHA) * prior[i] + ALPHA * oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]
                d = temp_scale(d, temperature)
                fl = {1: 0.008, 2: 0.008, 3: 0.006, 4: 0.003, 11: 0.004, 0: 0.005}.get(code, 0.005)
                d = [max(v, fl) for v in d]
            t = sum(d)
            row.append([v / t for v in d])
        tensor.append(row)
    return tensor


def run_fixed(igrid, gt, cond_matrix, n_hist=50):
    all_obs = get_observations(igrid, gt)
    bl = calibrate_fixed(igrid, all_obs, cond_matrix, n_hist)
    tensor = build_tensor(igrid, all_obs, bl, cond_matrix)
    return score_tensor(tensor, gt)


def run_adaptive(igrid, gt, cond_matrix, n_normal=50, n_surprised=15, threshold=0.3):
    all_obs = get_observations(igrid, gt)
    bl, n_s = calibrate_adaptive(igrid, all_obs, cond_matrix, n_normal, n_surprised, threshold)
    tensor = build_tensor(igrid, all_obs, bl, cond_matrix)
    return score_tensor(tensor, gt), n_s


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

    print(f"Adaptive N_HIST test: {len(files)} files, {len(rounds)} rounds, {N_MC} MC")
    print()

    strategies = [
        ("A. Fixed N=50", "fixed", 50, 50, 0),
        ("B. Fixed N=30", "fixed", 30, 30, 0),
        ("C. Adapt 50/15 t=0.20", "adapt", 50, 15, 0.20),
        ("D. Adapt 50/15 t=0.30", "adapt", 50, 15, 0.30),
        ("E. Adapt 50/10 t=0.30", "adapt", 50, 10, 0.30),
        ("F. Adapt 50/15 t=0.50", "adapt", 50, 15, 0.50),
        ("G. Adapt 50/20 t=0.20", "adapt", 50, 20, 0.20),
        ("H. Adapt 50/5  t=0.30", "adapt", 50, 5, 0.30),
    ]

    results = {s[0]: defaultdict(list) for s in strategies}
    surprised_counts = {s[0]: [] for s in strategies}

    for round_id, round_files in sorted(rounds.items()):
        cond_loo = build_conditional_loo(files, round_id, all_data)

        for fname in round_files:
            d = all_data[fname]
            gt, igrid = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not igrid: continue

            for _ in range(N_MC):
                for label, mode, n_norm, n_surp, thresh in strategies:
                    if mode == "fixed":
                        score = run_fixed(igrid, gt, cond_loo, n_norm)
                        results[label][round_id].append(score)
                    else:
                        score, n_s = run_adaptive(igrid, gt, cond_loo, n_norm, n_surp, thresh)
                        results[label][round_id].append(score)
                        surprised_counts[label].append(n_s)

        short = round_id[:8]
        print(f"  {short}", end="", flush=True)
        for label, _, _, _, _ in strategies:
            avg = sum(results[label][round_id]) / len(results[label][round_id])
            print(f"  {avg:5.1f}", end="")
        print()

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    baseline = None
    print(f"\n  {'Strategy':<28} {'Avg':>6}  {'vs A':>6}  {'avg surprised':>14}")
    print(f"  {'-' * 28} {'-' * 6}  {'-' * 6}  {'-' * 14}")
    for label, mode, _, _, _ in strategies:
        all_scores = []
        for rid in rounds:
            all_scores.extend(results[label][rid])
        avg = sum(all_scores) / len(all_scores)
        if baseline is None:
            baseline = avg
        delta = avg - baseline
        sc = surprised_counts[label]
        avg_s = sum(sc) / len(sc) if sc else 0
        print(f"  {label:<28} {avg:6.2f}  {delta:+6.2f}  {avg_s:14.1f}")


if __name__ == "__main__":
    main()
