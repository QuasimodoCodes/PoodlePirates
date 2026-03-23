"""
round_weighting_test.py — Test if weighting recent rounds more helps.

Hypothesis: more recent rounds may be more predictive of future rounds
(simulation parameters might drift over time).

Strategies:
  PROD:     Equal weight to all training rounds
  RECENT_2: 2x weight to last 5 rounds
  RECENT_3: 3x weight to last 5 rounds
  DECAY:    Exponential decay (0.95^age)
  HALF:     Only use last 50% of rounds
  KNN_SETT: KNN for Settlement only, buckets for rest (K=100)

python -m scripts.round_weighting_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.001; TEMP = 1.10; ALPHA = 0.05
N_HIST = 50; N_HIST_SURP = 20; SURP_THRESH = 0.30

ALL_A = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
AC = {a: set(_covered_cells(*a)) for a in ALL_A}


def bfs_dist(ig, tc):
    H, W = 40, 40
    dist = [[99]*W for _ in range(H)]; q = []
    for y in range(H):
        for x in range(W):
            if ig[y][x] == tc: dist[y][x] = 0; q.append((y, x))
    head = 0
    while head < len(q):
        cy, cx = q[head]; head += 1
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy+dy, cx+dx
            if 0 <= ny < H and 0 <= nx < W and dist[ny][nx] > dist[cy][cx]+1:
                dist[ny][nx] = dist[cy][cx]+1; q.append((ny, nx))
    return dist

def bin_od(d):
    if d <= 1: return "od0"
    if d <= 4: return "od1"
    if d <= 10: return "od2"
    return "od3"
def bin_sd(d):
    if d <= 2: return "sd_close"
    if d <= 5: return "sd_mid"
    if d <= 10: return "sd_far"
    return "sd_void"
def kl_div(p, q):
    return sum(pi * math.log(pi / max(qi, 1e-12)) for pi, qi in zip(p, q) if pi > 1e-12)
def score_t(pred, gt):
    wkl = te = 0.0
    for y in range(40):
        for x in range(40):
            e = _entropy(gt[y][x]); wkl += e * kl_div(gt[y][x], pred[y][x]); te += e
    return max(0, min(100, 100 * math.exp(-3 * (wkl / te)))) if te > 1e-12 else 100
def so(d):
    r = random.random(); cs = 0.0
    for i, p in enumerate(d):
        cs += p
        if r < cs: return i
    return 5
def cn(ig, y, x, tc, r=2):
    c = 0
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dy == 0 and dx == 0: continue
            ny, nx = y+dy, x+dx
            if 0 <= ny < 40 and 0 <= nx < 40 and ig[ny][nx] == tc: c += 1
    return c
def ts(d, t):
    if t == 1.0: s = sum(d); return [v/s for v in d]
    ld = [math.log(max(p, 1e-12))/t for p in d]; mx = max(ld)
    ed = [math.exp(v - mx) for v in ld]; s = sum(ed)
    return [v/s for v in ed]
def surprise(ol, h):
    nr = len(ol)
    if nr < 3: return 0.0
    rf = [0.0]*N
    for v in ol: rf[v] += 1.0/nr
    kf = sum(rf[i]*math.log(max(rf[i],1e-12)/max(h[i],1e-12)) for i in range(N) if rf[i] > 1e-12)
    kr = sum(h[i]*math.log(max(h[i],1e-12)/max(rf[i],1e-12)) for i in range(N) if h[i] > 1e-12)
    return (kf + kr) / 2
def compute_shift(rc, cm):
    rt = [0.0]*N; no = 0
    for ct, ol in rc.items():
        for cls in ol: rt[cls] += 1; no += 1
    if no == 0: return [1.0]*N
    rf = [c/no for c in rt]
    ht = [0.0]*N; nb = 0
    for ct, d in cm.items():
        for i in range(N): ht[i] += d[i]
        nb += 1
    if nb == 0: return [1.0]*N
    hf = [h/nb for h in ht]
    return [math.sqrt(rf[i]/hf[i]) if hf[i] > 1e-8 else 1.0 for i in range(N)]
def apply_shift(d, R):
    s = [max(d[i]*R[i], 1e-12) for i in range(N)]
    t = sum(s); return [v/t for v in s]
def ctx_fn(ig, y, x, odm, sdm):
    code = ig[y][x]; sn = cn(ig, y, x, 1)
    sb = "sh" if sn >= 3 else ("sl" if sn >= 1 else "sn")
    if code in (1, 4, 11) and odm and sdm:
        return (code, bin_sd(sdm[y][x]), bin_od(odm[y][x]))
    else:
        on = cn(ig, y, x, 10)
        return (code, sb, "oc" if on >= 1 else "in")
def sel_sett(ig, n=5):
    ss = {(y,x) for y in range(40) for x in range(40) if ig[y][x] == 1}
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
def get_obs(ig, gt):
    obs1 = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2 = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
    ao = {}
    for a in sel_sett(ig, 5):
        for yx in AC[a]: ao[yx] = [obs1[yx]]
    for a in SPREAD_ANCHORS:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx] = [obs2[yx]]
    return ao


def build_weighted_cm(files, all_data, odm_c, sdm_c, exclude_round, round_order, weight_fn):
    """Build CM with per-round weights."""
    ca = defaultdict(lambda: [[], []])  # [samples, weights]
    n_rounds = len(round_order)

    for f in files:
        rid = f.split("_seed")[0]
        if rid == exclude_round: continue
        d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
        if not gt or not ig: continue
        odm = odm_c.get(f); sdm = sdm_c.get(f)

        # Round index (0 = oldest)
        ridx = round_order.index(rid) if rid in round_order else 0
        w = weight_fn(ridx, n_rounds)

        for y in range(40):
            for x in range(40):
                c = ig[y][x]
                if c not in STATIC_CODES:
                    ct = ctx_fn(ig, y, x, odm, sdm)
                    ca[ct][0].append(gt[y][x])
                    ca[ct][1].append(w)

    cm = {}
    for ct, (samples, weights) in ca.items():
        tw = sum(weights)
        if tw < 1e-12: continue
        cm[ct] = [sum(w * s[i] for s, w in zip(samples, weights)) / tw for i in range(N)]
    return cm


def knn_sett_predict(query_feat, train_sett, K=100):
    """KNN for settlement cells only."""
    _, qod, qsd, qfn, qsn, qon = query_feat
    scored = []
    for feat, gt_dist in train_sett:
        _, cod, csd, cfn, csn, con = feat
        d = (qod - cod)**2 + (qsd - csd)**2 + 0.3*(qfn - cfn)**2 + 0.5*(qsn - csn)**2 + 0.3*(qon - con)**2
        scored.append((d, gt_dist))
    scored.sort(key=lambda x: x[0])
    neighbors = scored[:K]
    if not neighbors: return [1.0/N]*N
    if neighbors[0][0] < 1e-12:
        exact = [gt for d, gt in neighbors if d < 1e-12]
        return [sum(g[i] for g in exact)/len(exact) for i in range(N)]
    total_w = 0; avg = [0.0]*N
    for d, gt_dist in neighbors:
        w = 1.0 / (d + 0.1)
        for i in range(N): avg[i] += w * gt_dist[i]
        total_w += w
    return [v/total_w for v in avg]


def run_model(ig, gt, cm, odm, sdm, ao, sett_knn_data=None):
    """Run model with optional KNN for settlements."""
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_fn(ig, yx[0], yx[1], odm, sdm)
            for v in vals: rc[ct].append(v)
    R = compute_shift(rc, cm)
    shifted = {ct: apply_shift(d, R) for ct, d in cm.items()}
    bl = {}
    for ct in set(list(shifted.keys()) + list(rc.keys())):
        h = shifted.get(ct, [1.0/N]*N)
        ol = rc.get(ct, [])
        nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surprise(ol, h)
        nh = N_HIST_SURP if (s > SURP_THRESH and nr >= 5) else N_HIST
        rf = [0.0]*N
        for v in ol: rf[v] += 1.0/nr
        t = nr + nh
        bl[ct] = [(nr*rf[i] + nh*h[i])/t for i in range(N)]

    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS]*N; d[pc] = 1-5*FS
            else:
                # KNN for settlement cells if available
                if code == 1 and sett_knn_data is not None:
                    feat = (code, odm[y][x], sdm[y][x], cn(ig, y, x, 4), cn(ig, y, x, 1), cn(ig, y, x, 10))
                    prior = knn_sett_predict(feat, sett_knn_data, K=100)
                    prior = apply_shift(prior, R)
                else:
                    ct = ctx_fn(ig, y, x, odm, sdm)
                    prior = bl.get(ct, shifted.get(ct, [1.0/N]*N))[:]

                if (y, x) in ao:
                    vals = ao[(y, x)]; no = len(vals); oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/no
                    d = [(1-ALPHA)*prior[i] + ALPHA*oh[i] for i in range(N)]
                else: d = prior[:]
                d = ts(d, TEMP)
                d = [max(v, FD) for v in d]
            t = sum(d); row.append([v/t for v in d])
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

    print("Precomputing distance maps...")
    odm_c = {}; sdm_c = {}
    for f in files:
        ig = all_data[f].get("initial_grid")
        if ig:
            odm_c[f] = bfs_dist(ig, 10)
            sdm_c[f] = bfs_dist(ig, 1)
    print(f"  Done: {len(odm_c)} maps.")

    round_ids = sorted(rounds.keys())

    # Weight functions
    def w_equal(ridx, n): return 1.0
    def w_recent2(ridx, n): return 2.0 if ridx >= n - 5 else 1.0
    def w_recent3(ridx, n): return 3.0 if ridx >= n - 5 else 1.0
    def w_decay(ridx, n): return 0.95 ** (n - 1 - ridx)
    def w_half(ridx, n): return 1.0 if ridx >= n // 2 else 0.0

    strategies = [
        ("PROD",     w_equal,   False),
        ("RECENT_2", w_recent2, False),
        ("RECENT_3", w_recent3, False),
        ("DECAY",    w_decay,   False),
        ("HALF",     w_half,    False),
        ("KNN_SETT", w_equal,   True),
    ]

    SEEDS = [42, 123]

    print(f"\nRound Weighting + KNN Settlement Test: {len(files)} files, {len(rounds)} rounds, {len(SEEDS)} seeds")
    print("=" * 100)
    hdr = f"  {'Round':<10}"
    for s, _, _ in strategies: hdr += f" {s:>10}"
    print(hdr)
    print(f"  {'-'*10}" + f" {'-'*10}" * len(strategies))

    totals = {s: [] for s, _, _ in strategies}

    for trid in round_ids:
        # Build training data for KNN_SETT
        sett_train = []
        for f in files:
            if f.split("_seed")[0] == trid: continue
            d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(f); sdm = sdm_c.get(f)
            for y in range(40):
                for x in range(40):
                    if ig[y][x] == 1:  # Settlement only
                        feat = (1, odm[y][x], sdm[y][x], cn(ig, y, x, 4), cn(ig, y, x, 1), cn(ig, y, x, 10))
                        sett_train.append((feat, gt[y][x]))

        # Build CMs with different weights
        cms = {}
        for label, wfn, use_knn in strategies:
            cms[label] = build_weighted_cm(files, all_data, odm_c, sdm_c, trid, round_ids, wfn)

        round_scores = {s: [] for s, _, _ in strategies}
        for fname in rounds[trid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(fname); sdm = sdm_c.get(fname)

            for seed in SEEDS:
                random.seed(seed + all_data[fname].get("seed_index", 0))
                ao = get_obs(ig, gt)
                for label, wfn, use_knn in strategies:
                    sc = run_model(ig, gt, cms[label], odm, sdm, ao,
                                   sett_knn_data=sett_train if use_knn else None)
                    round_scores[label].append(sc)

        avgs = {s: sum(round_scores[s])/max(1,len(round_scores[s])) for s, _, _ in strategies}
        for s, _, _ in strategies: totals[s].extend(round_scores[s])
        best = max(avgs, key=avgs.get)
        short = trid[:8]
        line = f"  {short:<10}"
        for s, _, _ in strategies:
            m = "*" if s == best else " "
            line += f" {avgs[s]:9.1f}{m}"
        print(line)

    print(f"\n{'='*100}")
    prod_avg = sum(totals["PROD"]) / len(totals["PROD"])
    print(f"\n  {'Strategy':<12} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Std':>7} {'Wins':>5}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")
    for s, _, _ in strategies:
        sc = totals[s]
        avg = sum(sc)/len(sc)
        mn = min(sc)
        std = (sum((v-avg)**2 for v in sc)/len(sc))**0.5
        wins = sum(1 for a, b in zip(totals["PROD"], sc) if b > a)
        delta = avg - prod_avg
        marker = " <<<" if delta > 0.3 else ""
        print(f"  {s:<12} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {std:7.1f} {wins:>5}{marker}")

    print(f"\n  Settlement training size: {len(sett_train)} cells")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
