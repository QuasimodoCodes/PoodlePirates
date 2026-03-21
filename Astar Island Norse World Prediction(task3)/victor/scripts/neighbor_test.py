"""Test all neighbor interaction effects."""
import os, sys, json, math, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N_CLASSES = 6; FLOOR_STATIC = 1e-5; FLOOR_DYN = 0.005; N_MC = 5
random.seed(42)

def kl_divergence(p, q):
    return sum(pi * math.log(pi / max(qi, 1e-12)) for pi, qi in zip(p, q) if pi > 1e-12)

def score_tensor(pred, gt):
    H, W = len(gt), len(gt[0])
    wkl = te = 0.0
    for y in range(H):
        for x in range(W):
            e = _entropy(gt[y][x]); wkl += e * kl_divergence(gt[y][x], pred[y][x]); te += e
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

ALL_ANCHORS = [(ax, ay) for ay in range(40-TILE_H+1) for ax in range(40-TILE_W+1)]
ANCHOR_CELLS = {a: set(_covered_cells(*a)) for a in ALL_ANCHORS}

def build_loo(files, exclude_id, all_data):
    acc = defaultdict(list)
    for f in files:
        if f.split("_seed")[0] == exclude_id:
            continue
        d = all_data[f]
        gt, ig = d.get("ground_truth"), d.get("initial_grid")
        if not gt or not ig:
            continue
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
        if c not in m:
            m[c] = [1/N_CLASSES]*N_CLASSES
    return m

def select_sett_tiles(igrid, n=5):
    setts = {(y,x) for y in range(len(igrid)) for x in range(len(igrid[0])) if igrid[y][x]==1}
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

def calibrate(igrid, obs_codes, hist, n_hist=50):
    bl = {}
    for code in [0,1,2,3,4,5,10,11]:
        h = hist.get(code, [1/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES:
            bl[code] = h[:]
            continue
        ol = obs_codes.get(code, [])
        nr = len(ol)
        if nr == 0:
            bl[code] = h[:]
            continue
        rf = [0.0]*N_CLASSES
        for c in ol:
            rf[c] += 1.0/nr
        t = nr + n_hist
        bl[code] = [(nr*rf[i]+n_hist*h[i])/t for i in range(N_CLASSES)]
    return bl

def count_neighbors(igrid, y, x, target_code, radius=2):
    H, W = len(igrid), len(igrid[0])
    c = 0
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and igrid[ny][nx] == target_code:
                c += 1
    return c

def run_test(igrid, gt, matrix, use_sett=True, use_ocean=True, use_sett_cluster=True):
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}

    p1 = select_sett_tiles(igrid)
    obs_p1 = set()
    for ax, ay in p1:
        obs_p1 |= tile_cells(ax, ay)
    obs_p2 = set()
    for ax, ay in SPREAD_ANCHORS:
        obs_p2 |= tile_cells(ax, ay)

    oc = defaultdict(list)
    for y, x in obs_p1:
        c = igrid[y][x]
        if c not in STATIC_CODES:
            oc[c].append(obs1[(y, x)])
    for y, x in obs_p2:
        c = igrid[y][x]
        if c not in STATIC_CODES:
            oc[c].append(obs2[(y, x)])

    bl = calibrate(igrid, oc, matrix)
    obs_both = obs_p1 & obs_p2

    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                d = [FLOOR_STATIC]*N_CLASSES
                d[pc] = 1.0 - FLOOR_STATIC * 5
            else:
                prior = bl.get(code, [1/N_CLASSES]*N_CLASSES)[:]

                # 1. Settlement neighbor boost (Forest/Plains near settlements)
                if use_sett and code in (4, 11):
                    nc = count_neighbors(igrid, y, x, 1)
                    if nc > 0:
                        boost = min(nc * 0.01, 0.05)
                        prior[1] += boost        # P(Settlement)
                        prior[0] += boost * 0.7  # P(Empty)
                        tp = sum(prior)
                        prior = [p/tp for p in prior]

                # 2. Ocean proximity (Forest/Plains/Settlement near ocean)
                if use_ocean and code in (4, 11, 1):
                    oc_n = count_neighbors(igrid, y, x, 10)
                    if oc_n > 0:
                        boost = min(oc_n * 0.005, 0.03)
                        prior[2] += boost  # P(Port)
                        if code in (4, 11):
                            prior[1] = max(prior[1] - boost * 0.5, 0.001)
                        tp = sum(prior)
                        prior = [p/tp for p in prior]

                # 3. Settlement cluster collapse
                if use_sett_cluster and code == 1:
                    sn = count_neighbors(igrid, y, x, 1)
                    if sn >= 2:
                        boost = min(sn * 0.005, 0.025)
                        prior[1] = max(prior[1] - boost, 0.01)
                        prior[0] += boost * 0.7
                        tp = sum(prior)
                        prior = [p/tp for p in prior]

                if (y,x) in obs_both:
                    oh = [0.0]*N_CLASSES
                    oh[obs1[(y,x)]] += 0.5
                    oh[obs2[(y,x)]] += 0.5
                    d = [0.95*prior[i]+0.05*oh[i] for i in range(N_CLASSES)]
                elif (y,x) in obs_p1:
                    oh = [0.0]*N_CLASSES
                    oh[obs1[(y,x)]] = 1.0
                    d = [0.95*prior[i]+0.05*oh[i] for i in range(N_CLASSES)]
                elif (y,x) in obs_p2:
                    oh = [0.0]*N_CLASSES
                    oh[obs2[(y,x)]] = 1.0
                    d = [0.95*prior[i]+0.05*oh[i] for i in range(N_CLASSES)]
                else:
                    d = prior[:]
                d = [max(v, FLOOR_DYN) for v in d]
            t = sum(d)
            row.append([v/t for v in d])
        tensor.append(row)
    return score_tensor(tensor, gt)


def main():
    hdir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(hdir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files:
        rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(hdir, f)) as fh:
            all_data[f] = json.load(fh)

    print(f"Neighbor interaction test: {len(files)} files, {len(rounds)} rounds, {N_MC} MC")
    print()

    configs = [
        ("A. No neighbors",            False, False, False),
        ("B. Sett boost only",          True,  False, False),
        ("C. +Ocean proximity",         True,  True,  False),
        ("D. +Sett cluster collapse",   True,  False, True),
        ("E. All three combined",       True,  True,  True),
        ("F. Ocean only",               False, True,  False),
        ("G. Cluster collapse only",    False, False, True),
    ]

    results = {}
    for label, s, o, sc in configs:
        all_scores = []
        round_scores = defaultdict(list)
        for rid, rfiles in sorted(rounds.items()):
            loo = build_loo(files, rid, all_data)
            for fname in rfiles:
                d = all_data[fname]
                gt, ig = d.get("ground_truth"), d.get("initial_grid")
                if not gt or not ig:
                    continue
                ts = [run_test(ig, gt, loo, s, o, sc) for _ in range(N_MC)]
                avg = sum(ts)/len(ts)
                all_scores.append(avg)
                round_scores[rid].append(avg)
        overall = sum(all_scores)/len(all_scores)
        results[label] = (overall, round_scores)
        base_delta = overall - 75.61
        print(f"  {label:<30} avg={overall:.2f}  ({base_delta:+.2f} vs no-neighbor)")

    # Per-round detail
    print()
    print("  Per-round breakdown:")
    labels = [l for l, _, _, _ in configs]
    print(f"  {'Round':<12}", end="")
    for label in labels:
        short = label.split(". ")[1][:10]
        print(f" {short:>10}", end="")
    print()
    print("  " + "-" * (12 + 11 * len(labels)))

    for round_id in sorted(list(rounds.keys())):
        short_id = round_id[:8]
        print(f"  {short_id:<12}", end="")
        for label in labels:
            scores = results[label][1][round_id]
            avg = sum(scores)/len(scores)
            print(f" {avg:10.2f}", end="")
        print()


if __name__ == "__main__":
    main()
