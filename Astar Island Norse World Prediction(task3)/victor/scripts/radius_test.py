"""
radius_test.py — Test whether expanding neighbor radius improves predictions.

Current: r=2 (5x5, 24 neighbors)
Test: r=3 (7x7, 48 neighbors), r=4 (9x9, 80 neighbors), r=5 (11x11, 120 neighbors)
Also test: different bin thresholds for wider radii

python -m scripts.radius_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.005; ALPHA = 0.05; TEMP = 1.10
TF = {1: .008, 2: .008, 3: .006, 4: .003, 11: .004, 0: .005}
random.seed(42)

ALL_A = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
AC = {a: set(_covered_cells(*a)) for a in ALL_A}


def kl(p, q):
    return sum(pi * math.log(pi / max(qi, 1e-12)) for pi, qi in zip(p, q) if pi > 1e-12)

def score_t(pred, gt):
    wkl = te = 0.0
    for y in range(40):
        for x in range(40):
            e = _entropy(gt[y][x]); wkl += e * kl(gt[y][x], pred[y][x]); te += e
    return max(0, min(100, 100 * math.exp(-3 * (wkl / te)))) if te > 1e-12 else 100

def so(d):
    r = random.random(); cs = 0.0
    for i, p in enumerate(d):
        cs += p
        if r < cs: return i
    return 5

def cn(ig, y, x, tc, r=2):
    c = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < 40 and 0 <= nx < 40 and ig[ny][nx] == tc: c += 1
    return c

def ctx_r(ig, y, x, radius, sett_thresholds, ocean_thresh=1):
    """Context with configurable radius and thresholds."""
    code = ig[y][x]
    sn = cn(ig, y, x, 1, r=radius)
    on = cn(ig, y, x, 10, r=radius)
    hi, lo = sett_thresholds
    sb = "sett_hi" if sn >= hi else ("sett_lo" if sn >= lo else "sett_no")
    ob = "ocean" if on >= ocean_thresh else "inland"
    return (code, sb, ob)

def ts(d, t):
    ld = [math.log(max(p, 1e-12)) / t for p in d]; mx = max(ld)
    ed = [math.exp(v - mx) for v in ld]; s = sum(ed)
    return [v / s for v in ed]

def sel_sett(ig, n=5):
    ss = {(y, x) for y in range(40) for x in range(40) if ig[y][x] == 1}
    if not ss: return SPREAD_ANCHORS[:n]
    cov, sel = set(), []
    for _ in range(n):
        b, bc = None, -1
        for a in ALL_A:
            c = len((AC[a] & ss) - cov)
            if c > bc: bc, b = c, a
        if not b or bc <= 0: break
        sel.append(b); cov |= (AC[b] & ss)
    return sel

def surp(ol, h):
    nr = len(ol)
    if nr < 3: return 0.0
    rf = [0.0] * N
    for v in ol: rf[v] += 1.0 / nr
    kf = sum(rf[i] * math.log(max(rf[i], 1e-12) / max(h[i], 1e-12)) for i in range(N) if rf[i] > 1e-12)
    kr = sum(h[i] * math.log(max(h[i], 1e-12) / max(rf[i], 1e-12)) for i in range(N) if h[i] > 1e-12)
    return (kf + kr) / 2

def build_prediction(ig, gt, cm, ctx_fn):
    obs1 = {(y, x): so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2 = {(y, x): so(gt[y][x]) for y in range(40) for x in range(40)}
    ao = {}
    for a in sel_sett(ig, 5):
        for yx in AC[a]: ao[yx] = [obs1[yx]]
    for a in SPREAD_ANCHORS:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx] = [obs2[yx]]

    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_fn(ig, yx[0], yx[1])
            for v in vals: rc[ct].append(v)
    bl = {}
    for ct in set(list(cm.keys()) + list(rc.keys())):
        h = cm.get(ct, [1 / N] * N); ol = rc.get(ct, []); nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surp(ol, h)
        nh = 5 if s > .3 and nr >= 5 else 50
        rf = [0.0] * N
        for v in ol: rf[v] += 1.0 / nr
        t = nr + nh
        bl[ct] = [(nr * rf[i] + nh * h[i]) / t for i in range(N)]

    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS] * N; d[pc] = 1 - 5 * FS
            else:
                ct = ctx_fn(ig, y, x)
                prior = bl.get(ct, cm.get(ct, [1 / N] * N))[:]
                if (y, x) in ao:
                    vals = ao[(y, x)]; oh = [0.0] * N
                    for v in vals: oh[v] += 1.0 / len(vals)
                    d = [(1 - ALPHA) * prior[i] + ALPHA * oh[i] for i in range(N)]
                else:
                    d = prior[:]
                d = ts(d, TEMP)
                fl = TF.get(code, FD); d = [max(v, fl) for v in d]
            t = sum(d); row.append([v / t for v in d])
        tensor.append(row)
    return tensor


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    round_ids = sorted(rounds.keys())
    n_rounds = len(rounds)
    print(f"Radius test: {len(files)} files, {n_rounds} rounds")
    print("=" * 70)

    # Configurations to test:
    # (name, radius, (sett_hi_thresh, sett_lo_thresh), ocean_thresh)
    configs = [
        ("r=2 (current)",   2, (3, 1), 1),
        ("r=3 same-bins",   3, (3, 1), 1),
        ("r=3 scaled",      3, (5, 2), 2),
        ("r=4 same-bins",   4, (3, 1), 1),
        ("r=4 scaled",      4, (8, 3), 3),
        ("r=5 scaled",      5, (12, 4), 4),
        # Also test: just ocean radius wider (settlements stay r=2)
        ("r=2s+r=4o",       None, None, None),  # special: mixed
    ]

    for cfg_name, radius, sett_thresh, ocean_thresh in configs:
        if cfg_name == "r=2s+r=4o":
            # Mixed: settlement context at r=2, ocean context at r=4
            def ctx_fn(ig, y, x):
                code = ig[y][x]
                sn = cn(ig, y, x, 1, r=2)
                on = cn(ig, y, x, 10, r=4)
                sb = "sett_hi" if sn >= 3 else ("sett_lo" if sn >= 1 else "sett_no")
                ob = "ocean" if on >= 3 else "inland"
                return (code, sb, ob)
        else:
            _r, _st, _ot = radius, sett_thresh, ocean_thresh
            def ctx_fn(ig, y, x, _r=_r, _st=_st, _ot=_ot):
                return ctx_r(ig, y, x, _r, _st, _ot)

        all_scores = []
        n_buckets_list = []

        for test_rid in round_ids:
            # Build conditional matrix from all other rounds
            cond_acc = defaultdict(list)
            for f in files:
                if f.split("_seed")[0] == test_rid: continue
                d = all_data[f]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
                if not gt or not ig: continue
                for y in range(40):
                    for x in range(40):
                        c = ig[y][x]
                        if c not in STATIC_CODES:
                            cond_acc[ctx_fn(ig, y, x)].append(gt[y][x])
            cm = {c: [sum(s[i] for s in ss) / len(ss) for i in range(N)] for c, ss in cond_acc.items()}
            n_buckets_list.append(len(cm))

            for fname in rounds[test_rid]:
                d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
                if not gt or not ig: continue
                tensor = build_prediction(ig, gt, cm, ctx_fn)
                all_scores.append(score_t(tensor, gt))

        avg = sum(all_scores) / len(all_scores)
        std = (sum((s - avg) ** 2 for s in all_scores) / len(all_scores)) ** 0.5
        avg_buckets = sum(n_buckets_list) / len(n_buckets_list)
        print(f"  {cfg_name:<20}  avg={avg:.2f}  std={std:.1f}  buckets={avg_buckets:.0f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
