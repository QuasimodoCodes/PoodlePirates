"""
calibration_fix_test2.py - Fine-grained sweep around promising calibration fixes.

From test1: G (t=1.10, g=0.95) = +0.10, K (t=1.05, g=0.90) = +0.15
Zoom in around these regions.

python -m scripts.calibration_fix_test2
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.005; ALPHA = 0.05; N_MC = 10
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

def ctx(ig, y, x):
    code = ig[y][x]; sn = cn(ig, y, x, 1); on = cn(ig, y, x, 10)
    sb = "sett_hi" if sn >= 3 else ("sett_lo" if sn >= 1 else "sett_no")
    ob = "ocean" if on >= 1 else "inland"
    return (code, sb, ob)

def ts(d, t):
    ld = [math.log(max(p, 1e-12)) / t for p in d]; mx = max(ld)
    ed = [math.exp(v - mx) for v in ld]; s = sum(ed)
    return [v / s for v in ed]

def power_recal(d, gamma):
    powered = [max(p, 1e-12) ** gamma for p in d]
    s = sum(powered)
    return [v / s for v in powered]

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

def get_obs(ig, gt):
    obs1 = {(y, x): so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2 = {(y, x): so(gt[y][x]) for y in range(40) for x in range(40)}
    ao = {}
    for a in sel_sett(ig, 5):
        for yx in AC[a]: ao[yx] = [obs1[yx]]
    for a in SPREAD_ANCHORS:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx] = [obs2[yx]]
    return ao

def run_model(ig, gt, cm, temp=1.10, gamma=None):
    ao = get_obs(ig, gt)
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
                if temp != 1.0:
                    d = ts(d, temp)
                if gamma is not None:
                    d = power_recal(d, gamma)
                fl = TF.get(code, FD); d = [max(v, fl) for v in d]
            t = sum(d); row.append([v / t for v in d])
        tensor.append(row)
    return score_t(tensor, gt)


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    strategies = [
        # (label, temp, gamma)
        ("A. t=1.10 (baseline)", 1.10, None),
        ("B. t=1.10 g=0.97",    1.10, 0.97),
        ("C. t=1.10 g=0.95",    1.10, 0.95),
        ("D. t=1.10 g=0.93",    1.10, 0.93),
        ("E. t=1.08 g=0.95",    1.08, 0.95),
        ("F. t=1.07 g=0.93",    1.07, 0.93),
        ("G. t=1.05 g=0.92",    1.05, 0.92),
        ("H. t=1.05 g=0.90",    1.05, 0.90),
        ("I. t=1.05 g=0.88",    1.05, 0.88),
        ("J. t=1.08",           1.08, None),
        ("K. t=1.12",           1.12, None),
        ("L. t=1.15",           1.15, None),
    ]

    print(f"Calibration fix test 2: {len(files)} files, {len(rounds)} rounds, {N_MC} MC")
    print()

    labels_short = [chr(65 + i) for i in range(len(strategies))]
    print(f"  {'Round':<10}", end="")
    for l in labels_short:
        print(f" {l:>6}", end="")
    print()
    print(f"  {'-'*10}", end="")
    for _ in labels_short:
        print(f" {'-'*6}", end="")
    print()

    results = {s[0]: [] for s in strategies}

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

        round_scores = {s[0]: [] for s in strategies}
        for fname in rfiles:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            for _ in range(N_MC):
                for label, temp, gamma in strategies:
                    sc = run_model(ig, gt, cm, temp, gamma)
                    round_scores[label].append(sc)

        short = rid[:8]
        print(f"  {short:<10}", end="")
        for label, _, _ in strategies:
            avg = sum(round_scores[label]) / len(round_scores[label])
            results[label].extend(round_scores[label])
            print(f" {avg:6.1f}", end="")
        print()

    print()
    print("=" * 90)
    baseline_avg = None
    print(f"\n  {'Strategy':<28} {'Avg':>7} {'vs A':>7}")
    print(f"  {'-'*28} {'-'*7} {'-'*7}")
    for label, temp, gamma in strategies:
        avg = sum(results[label]) / len(results[label])
        if baseline_avg is None:
            baseline_avg = avg
        delta = avg - baseline_avg
        marker = " <<<" if delta > 0.2 else (" ***" if delta > 0.05 else "")
        print(f"  {label:<28} {avg:7.2f} {delta:+7.2f}{marker}")


if __name__ == "__main__":
    main()
