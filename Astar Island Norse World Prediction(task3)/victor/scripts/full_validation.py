"""Full end-to-end validation: old model vs new model with all improvements.
python -m scripts.full_validation
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

def score(pred, gt):
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

def run_old(ig, gt, flat):
    ao = get_obs(ig, gt)
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            for v in vals: rc[c].append(v)
    bl = {}
    for c in [0, 1, 2, 3, 4, 5, 10, 11]:
        h = flat.get(c, [1 / N] * N)
        if c in STATIC_CODES: bl[c] = h[:]; continue
        ol = rc.get(c, []); nr = len(ol)
        if nr == 0: bl[c] = h[:]; continue
        rf = [0.0] * N
        for v in ol: rf[v] += 1.0 / nr
        t = nr + 50
        bl[c] = [(nr * rf[i] + 50 * h[i]) / t for i in range(N)]
    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS] * N; d[pc] = 1 - 5 * FS
            else:
                prior = bl.get(code, [1 / N] * N)[:]
                if code in (4, 11):
                    nc = cn(ig, y, x, 1)
                    if nc > 0:
                        b = min(nc * .01, .05); prior[1] += b; prior[0] += b * .7
                        tp = sum(prior); prior = [p / tp for p in prior]
                if code in (1, 4, 11):
                    oc = cn(ig, y, x, 10)
                    if oc > 0:
                        b = min(oc * .005, .03); prior[2] += b
                        if code in (4, 11): prior[1] = max(prior[1] - b * .5, .001)
                        tp = sum(prior); prior = [p / tp for p in prior]
                if (y, x) in ao:
                    vals = ao[(y, x)]; oh = [0.0] * N
                    for v in vals: oh[v] += 1.0 / len(vals)
                    d = [(1 - ALPHA) * prior[i] + ALPHA * oh[i] for i in range(N)]
                else:
                    d = prior[:]
                fl = TF.get(code, FD); d = [max(v, fl) for v in d]
            t = sum(d); row.append([v / t for v in d])
        tensor.append(row)
    return score(tensor, gt)

def run_new(ig, gt, cm):
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
                d = ts(d, 1.10)
                fl = TF.get(code, FD); d = [max(v, fl) for v in d]
            t = sum(d); row.append([v / t for v in d])
        tensor.append(row)
    return score(tensor, gt)

def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    print(f"Full validation: {len(files)} files, {len(rounds)} rounds, {N_MC} MC")
    print("  Old: flat matrix + hand-tuned boosts, fixed N_HIST=50, no temperature")
    print("  New: conditional matrix + adaptive N_HIST(50/5) + temp=1.10")
    print()
    print(f"  {'Round':<12} {'Old':>8} {'New':>8} {'Delta':>8}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8}")

    oa, na = [], []
    for rid, rfiles in sorted(rounds.items()):
        flat_acc = defaultdict(list)
        cond_acc = defaultdict(list)
        for f in files:
            if f.split("_seed")[0] == rid: continue
            d = all_data[f]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in STATIC_CODES:
                        flat_acc[c].append(gt[y][x])
                        cond_acc[ctx(ig, y, x)].append(gt[y][x])
        fm = {c: [sum(s[i] for s in ss) / len(ss) for i in range(N)] for c, ss in flat_acc.items()}
        for c in [0, 1, 2, 3, 4, 5, 10, 11]:
            if c not in fm: fm[c] = [1 / N] * N
        cm = {c: [sum(s[i] for s in ss) / len(ss) for i in range(N)] for c, ss in cond_acc.items()}

        os_, ns_ = [], []
        for fname in rfiles:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            for _ in range(N_MC):
                os_.append(run_old(ig, gt, fm))
                ns_.append(run_new(ig, gt, cm))
        o = sum(os_) / len(os_); n = sum(ns_) / len(ns_)
        oa.extend(os_); na.extend(ns_)
        mk = " <<<" if abs(n - o) > 1 else ""
        print(f"  {rid[:8]:<12} {o:8.2f} {n:8.2f} {n - o:+8.2f}{mk}")

    ot = sum(oa) / len(oa); nt = sum(na) / len(na)
    print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8}")
    print(f"  {'AVERAGE':<12} {ot:8.2f} {nt:8.2f} {nt - ot:+8.2f}")
    print(f"\n  Total improvement: {nt - ot:+.2f} pts across {len(rounds)} rounds")

if __name__ == "__main__":
    main()
