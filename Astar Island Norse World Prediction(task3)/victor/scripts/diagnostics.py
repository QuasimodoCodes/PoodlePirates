"""
diagnostics.py - Model diagnostics: calibration, learning curve, per-terrain analysis.

1. Calibration plot: predicted probability vs actual frequency (binned)
   - Overconfident = line below diagonal (predicting 80% but only right 60%)
   - Underconfident = line above diagonal
   - Perfect = on the diagonal

2. Learning curve: LOO score vs number of training rounds

3. Per-terrain and per-context breakdown: where do we lose most points?

python -m scripts.diagnostics
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.005; ALPHA = 0.05
TF = {1: .008, 2: .008, 3: .006, 4: .003, 11: .004, 0: .005}
CLS_NAMES = ["Empty", "Settl", "Port", "Ruin", "Forest", "Mtn"]
CODE_NAMES = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin",
              4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains"}
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

def ctx(ig, y, x):
    code = ig[y][x]; sn = cn(ig, y, x, 1); on = cn(ig, y, x, 10)
    sb = "sett_hi" if sn >= 3 else ("sett_lo" if sn >= 1 else "sett_no")
    ob = "ocean" if on >= 1 else "inland"
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

def build_prediction(ig, gt, cm):
    """Build prediction tensor using our full new model. Returns tensor + observations."""
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
            ct = ctx(ig, yx[0], yx[1])
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
                ct = ctx(ig, y, x)
                prior = bl.get(ct, cm.get(ct, [1 / N] * N))[:]
                if (y, x) in ao:
                    vals = ao[(y, x)]; oh = [0.0] * N
                    for v in vals: oh[v] += 1.0 / len(vals)
                    d = [(1 - ALPHA) * prior[i] + ALPHA * oh[i] for i in range(N)]
                else:
                    d = prior[:]
                d = ts(d, 1.10)
                fl = TF.get(code, FD); d = [max(v, fl) for v in d]
            t = sum(d); row.append([v / t for v in d])
        tensor.append(row)
    return tensor, ao


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    n_rounds = len(rounds)
    print(f"Model Diagnostics: {len(files)} files, {n_rounds} rounds")
    print("=" * 70)

    # ── 1. CALIBRATION ANALYSIS ──────────────────────────────────────
    print("\n1. CALIBRATION ANALYSIS")
    print("   (predicted probability vs actual ground truth frequency)")
    print()

    # Bins: 0-10%, 10-20%, ..., 90-100%
    n_bins = 10
    bin_pred_sum = [0.0] * n_bins  # sum of predicted probabilities
    bin_gt_sum = [0.0] * n_bins    # sum of actual GT probabilities
    bin_count = [0] * n_bins       # count of (cell, class) pairs in each bin

    # Per-terrain calibration
    terrain_bins = {code: {"pred": [0.0] * n_bins, "gt": [0.0] * n_bins, "n": [0] * n_bins}
                    for code in [1, 4, 11]}

    # Per-terrain score contribution
    terrain_wkl = defaultdict(float)
    terrain_te = defaultdict(float)

    for rid, rfiles in sorted(rounds.items()):
        cond_acc = defaultdict(list)
        for f in files:
            if f.split("_seed")[0] == rid: continue
            d = all_data[f]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in STATIC_CODES:
                        cond_acc[ctx(ig, y, x)].append(gt[y][x])
        cm = {c: [sum(s[i] for s in ss) / len(ss) for i in range(N)] for c, ss in cond_acc.items()}

        for fname in rfiles:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue

            tensor, _ = build_prediction(ig, gt, cm)

            for y in range(40):
                for x in range(40):
                    code = ig[y][x]
                    if code in STATIC_CODES: continue

                    pred_dist = tensor[y][x]
                    gt_dist = gt[y][x]
                    e = _entropy(gt_dist)
                    k = kl(gt_dist, pred_dist)
                    terrain_wkl[code] += e * k
                    terrain_te[code] += e

                    for cls in range(N):
                        p = pred_dist[cls]
                        g = gt_dist[cls]
                        b = min(int(p * n_bins), n_bins - 1)
                        bin_pred_sum[b] += p
                        bin_gt_sum[b] += g
                        bin_count[b] += 1

                        if code in terrain_bins:
                            terrain_bins[code]["pred"][b] += p
                            terrain_bins[code]["gt"][b] += g
                            terrain_bins[code]["n"][b] += 1

    # Print calibration table
    print(f"  {'Bin':>10} {'Avg Pred':>10} {'Avg GT':>10} {'Gap':>8} {'n':>8}  {'Visual'}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 8}  {'-' * 30}")
    for b in range(n_bins):
        if bin_count[b] == 0: continue
        avg_p = bin_pred_sum[b] / bin_count[b]
        avg_g = bin_gt_sum[b] / bin_count[b]
        gap = avg_g - avg_p
        lo = b * 10; hi = (b + 1) * 10
        # Visual: | for prediction, * for ground truth
        bar_p = int(avg_p * 50)
        bar_g = int(avg_g * 50)
        visual = ""
        for i in range(max(bar_p, bar_g) + 1):
            if i == bar_p and i == bar_g: visual += "X"
            elif i == bar_p: visual += "|"
            elif i == bar_g: visual += "*"
            else: visual += "."
        direction = "OVER" if gap < -0.02 else ("UNDER" if gap > 0.02 else "OK")
        print(f"  {lo:>3}-{hi:<3}%   {avg_p:10.4f} {avg_g:10.4f} {gap:+8.4f} {bin_count[b]:8d}  {direction}")

    # ── 2. PER-TERRAIN SCORE BREAKDOWN ───────────────────────────────
    print("\n2. PER-TERRAIN SCORE CONTRIBUTION")
    print("   (where are we losing the most points?)")
    print()
    total_wkl = sum(terrain_wkl.values())
    total_te = sum(terrain_te.values())
    print(f"  {'Terrain':<12} {'Entropy%':>10} {'wKL%':>10} {'Avg wKL':>10} {'Score if solo':>14}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 14}")
    for code in [11, 4, 1, 2, 3, 0]:
        te = terrain_te.get(code, 0)
        wk = terrain_wkl.get(code, 0)
        if te < 1e-12: continue
        avg_wkl = wk / te
        solo_score = max(0, min(100, 100 * math.exp(-3 * avg_wkl)))
        name = CODE_NAMES.get(code, f"code{code}")
        print(f"  {name:<12} {te / total_te * 100:9.1f}% {wk / total_wkl * 100:9.1f}% {avg_wkl:10.4f} {solo_score:13.1f}")

    overall_wkl = total_wkl / total_te
    overall_score = max(0, min(100, 100 * math.exp(-3 * overall_wkl)))
    print(f"  {'OVERALL':<12} {'100.0%':>10} {'100.0%':>10} {overall_wkl:10.4f} {overall_score:13.1f}")

    # ── 3. PER-TERRAIN CALIBRATION ───────────────────────────────────
    print("\n3. PER-TERRAIN CALIBRATION (high bins only)")
    print()
    for code in [11, 4, 1]:
        name = CODE_NAMES.get(code, f"code{code}")
        tb = terrain_bins[code]
        print(f"  {name}:")
        for b in range(n_bins):
            if tb["n"][b] < 10: continue
            avg_p = tb["pred"][b] / tb["n"][b]
            avg_g = tb["gt"][b] / tb["n"][b]
            gap = avg_g - avg_p
            lo = b * 10; hi = (b + 1) * 10
            direction = "OVER" if gap < -0.02 else ("UNDER" if gap > 0.02 else "ok")
            print(f"    {lo:>3}-{hi:<3}%  pred={avg_p:.4f}  gt={avg_g:.4f}  gap={gap:+.4f}  n={tb['n'][b]:>6}  {direction}")
        print()

    # ── 4. LEARNING CURVE ────────────────────────────────────────────
    print("4. LEARNING CURVE")
    print("   (LOO score with increasing training data)")
    print()

    round_ids = sorted(rounds.keys())
    # Test: train on 3,4,5,...,11 rounds, test on held-out
    for n_train in [3, 5, 7, 9, 11]:
        if n_train >= n_rounds: continue
        scores = []
        # Use first n_train rounds as training, rest as test
        for test_idx in range(n_rounds):
            test_rid = round_ids[test_idx]
            # Use n_train rounds (excluding test) for training
            train_rids = [r for i, r in enumerate(round_ids) if i != test_idx][:n_train]
            train_files = []
            for r in train_rids:
                train_files.extend(rounds[r])

            cond_acc = defaultdict(list)
            for f in train_files:
                d = all_data[f]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
                if not gt or not ig: continue
                for y in range(40):
                    for x in range(40):
                        c = ig[y][x]
                        if c not in STATIC_CODES:
                            cond_acc[ctx(ig, y, x)].append(gt[y][x])
            cm = {c: [sum(s[i] for s in ss) / len(ss) for i in range(N)] for c, ss in cond_acc.items()}

            for fname in rounds[test_rid]:
                d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
                if not gt or not ig: continue
                tensor, _ = build_prediction(ig, gt, cm)
                scores.append(score_t(tensor, gt))

        avg = sum(scores) / len(scores) if scores else 0
        bar = "#" * int(avg / 2)
        print(f"  {n_train:>2} train rounds: avg={avg:.2f}  {bar}")

    # Also show full LOO (n-1 training)
    scores = []
    for test_rid in round_ids:
        cond_acc = defaultdict(list)
        for f in files:
            if f.split("_seed")[0] == test_rid: continue
            d = all_data[f]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in STATIC_CODES:
                        cond_acc[ctx(ig, y, x)].append(gt[y][x])
        cm = {c: [sum(s[i] for s in ss) / len(ss) for i in range(N)] for c, ss in cond_acc.items()}
        for fname in rounds[test_rid]:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            tensor, _ = build_prediction(ig, gt, cm)
            scores.append(score_t(tensor, gt))
    avg = sum(scores) / len(scores) if scores else 0
    bar = "#" * int(avg / 2)
    print(f"  {n_rounds - 1:>2} train rounds: avg={avg:.2f}  {bar}  (full LOO)")

    # ── 5. OBSERVED vs UNOBSERVED CELLS ──────────────────────────────
    print("\n5. OBSERVED vs UNOBSERVED CELL PERFORMANCE")
    print()
    obs_wkl = unobs_wkl = obs_te = unobs_te = 0.0
    obs_n = unobs_n = 0
    for rid, rfiles in sorted(rounds.items()):
        cond_acc = defaultdict(list)
        for f in files:
            if f.split("_seed")[0] == rid: continue
            d = all_data[f]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in STATIC_CODES:
                        cond_acc[ctx(ig, y, x)].append(gt[y][x])
        cm = {c: [sum(s[i] for s in ss) / len(ss) for i in range(N)] for c, ss in cond_acc.items()}

        for fname in rfiles:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            tensor, ao = build_prediction(ig, gt, cm)
            for y in range(40):
                for x in range(40):
                    code = ig[y][x]
                    if code in STATIC_CODES: continue
                    e = _entropy(gt[y][x])
                    k = kl(gt[y][x], tensor[y][x])
                    if (y, x) in ao:
                        obs_wkl += e * k; obs_te += e; obs_n += 1
                    else:
                        unobs_wkl += e * k; unobs_te += e; unobs_n += 1

    obs_avg = obs_wkl / obs_te if obs_te > 0 else 0
    unobs_avg = unobs_wkl / unobs_te if unobs_te > 0 else 0
    obs_score = max(0, min(100, 100 * math.exp(-3 * obs_avg)))
    unobs_score = max(0, min(100, 100 * math.exp(-3 * unobs_avg)))
    print(f"  Observed cells:   n={obs_n:>7}  avg_wKL={obs_avg:.4f}  equiv_score={obs_score:.1f}")
    print(f"  Unobserved cells: n={unobs_n:>7}  avg_wKL={unobs_avg:.4f}  equiv_score={unobs_score:.1f}")
    print(f"  Gap: {unobs_avg - obs_avg:.4f} wKL  ({obs_score - unobs_score:.1f} pts)")


if __name__ == "__main__":
    main()
