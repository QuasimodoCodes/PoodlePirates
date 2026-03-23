"""
obs_integration_test.py — Test observation integration improvements.

Key findings:
- 76% of KL loss comes from tail classes
- Uniform TEMP changes hurt Plains (67% of cells)
- ALPHA_1OBS=0.03, ALPHA_MULTI=0.10 are defined but NOT deployed

Strategies:
  PROD:        ALPHA=0.05 uniform
  STEPPED:     ALPHA=0.03 for 1 obs, 0.10 for 2+ obs
  HIGHER_A:    ALPHA=0.10 uniform (trust observations more)
  SPATIAL_2:   Blend nearby same-terrain obs for unobserved cells (alpha=0.015)
  SPATIAL_3:   Spatial + stepped alpha
  BIGALPHA:    ALPHA=0.20 for observed cells (aggressive)
  BAYESIAN:    Proper Bayesian update with Dirichlet prior

python -m scripts.obs_integration_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.001; TEMP = 1.10
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


def build_blend(cm, rc):
    """Build blended prior from historical + round observations."""
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
    return bl, shifted


def run_model(ig, gt, cm, odm, sdm, ao, strategy="PROD"):
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_fn(ig, yx[0], yx[1], odm, sdm)
            for v in vals: rc[ct].append(v)

    bl, shifted = build_blend(cm, rc)

    # Spatial: build neighbor obs map
    obs_neighbor = {}
    if strategy in ("SPATIAL_2", "SPATIAL_3"):
        observed_set = set(ao.keys())
        for y in range(40):
            for x in range(40):
                if (y, x) in observed_set: continue
                code = ig[y][x]
                if code in STATIC_CODES: continue
                nearby = []
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < 40 and 0 <= nx < 40 and (ny, nx) in ao:
                            if ig[ny][nx] == code:
                                nearby.extend(ao[(ny, nx)])
                if nearby:
                    obs_neighbor[(y, x)] = nearby

    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS]*N; d[pc] = 1-5*FS
            else:
                ct = ctx_fn(ig, y, x, odm, sdm)
                prior = bl.get(ct, shifted.get(ct, [1.0/N]*N))[:]

                if (y, x) in ao:
                    vals = ao[(y, x)]; nobs = len(vals)
                    oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/nobs

                    if strategy == "PROD":
                        a = 0.05
                    elif strategy == "STEPPED":
                        a = 0.03 if nobs == 1 else 0.10
                    elif strategy == "HIGHER_A":
                        a = 0.10
                    elif strategy == "BIGALPHA":
                        a = 0.20
                    elif strategy == "SPATIAL_2":
                        a = 0.05
                    elif strategy == "SPATIAL_3":
                        a = 0.03 if nobs == 1 else 0.10
                    elif strategy == "BAYESIAN":
                        # Dirichlet update: prior counts + observation counts
                        # Effective sample size of prior = 30 (tunable)
                        prior_n = 30
                        counts = [prior[i] * prior_n for i in range(N)]
                        for v in vals:
                            counts[v] += 1.0
                        total_c = sum(counts)
                        d = [c / total_c for c in counts]
                        d = ts(d, TEMP)
                        d = [max(v, FD) for v in d]
                        t = sum(d); row.append([v/t for v in d])
                        continue
                    else:
                        a = 0.05

                    d = [(1-a)*prior[i] + a*oh[i] for i in range(N)]

                elif strategy in ("SPATIAL_2", "SPATIAL_3") and (y, x) in obs_neighbor:
                    nobs = obs_neighbor[(y, x)]
                    nn = len(nobs); oh = [0.0]*N
                    for v in nobs: oh[v] += 1.0/nn
                    sa = 0.015
                    d = [(1-sa)*prior[i] + sa*oh[i] for i in range(N)]
                else:
                    d = prior[:]

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
    strategies = ["PROD", "STEPPED", "HIGHER_A", "BIGALPHA", "SPATIAL_2", "SPATIAL_3", "BAYESIAN"]

    SEEDS = [42, 123, 456]

    print(f"\nObs Integration Test: {len(files)} files, {len(rounds)} rounds, {len(SEEDS)} seeds")
    print(f"PROD:      ALPHA=0.05 uniform")
    print(f"STEPPED:   ALPHA=0.03 (1 obs), 0.10 (2+ obs)")
    print(f"HIGHER_A:  ALPHA=0.10")
    print(f"BIGALPHA:  ALPHA=0.20")
    print(f"SPATIAL_2: Spatial neighbor blend (alpha=0.015) + ALPHA=0.05")
    print(f"SPATIAL_3: Spatial + stepped alpha")
    print(f"BAYESIAN:  Dirichlet update (prior_n=30)")
    print("=" * 110)
    hdr = f"  {'Round':<10}"
    for s in strategies: hdr += f" {s:>10}"
    print(hdr)
    print(f"  {'-'*10}" + f" {'-'*10}" * len(strategies))

    totals = {s: [] for s in strategies}

    for trid in round_ids:
        ca = defaultdict(list)
        for f in files:
            if f.split("_seed")[0] == trid: continue
            d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(f); sdm = sdm_c.get(f)
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in STATIC_CODES:
                        ca[ctx_fn(ig, y, x, odm, sdm)].append(gt[y][x])
        cm = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in ca.items()}

        round_scores = {s: [] for s in strategies}
        for fname in rounds[trid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(fname); sdm = sdm_c.get(fname)

            for seed in SEEDS:
                random.seed(seed + all_data[fname].get("seed_index", 0))
                ao = get_obs(ig, gt)
                for strat in strategies:
                    sc = run_model(ig, gt, cm, odm, sdm, ao, strategy=strat)
                    round_scores[strat].append(sc)

        avgs = {s: sum(round_scores[s])/max(1, len(round_scores[s])) for s in strategies}
        for s in strategies: totals[s].extend(round_scores[s])
        best = max(avgs, key=avgs.get)
        short = trid[:8]
        line = f"  {short:<10}"
        for s in strategies:
            m = "*" if s == best else " "
            line += f" {avgs[s]:9.1f}{m}"
        print(line)

    print(f"\n{'='*110}")
    prod_avg = sum(totals["PROD"]) / len(totals["PROD"])
    print(f"\n  {'Strategy':<12} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Std':>7} {'Wins':>5}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")
    for s in strategies:
        scores = totals[s]
        avg = sum(scores)/len(scores)
        mn = min(scores)
        std = (sum((v - avg)**2 for v in scores)/len(scores))**0.5
        wins = sum(1 for a, b in zip(totals["PROD"], scores) if b > a)
        delta = avg - prod_avg
        marker = " <<<" if delta > 0.3 else ""
        print(f"  {s:<12} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {std:7.1f} {wins:>5}{marker}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
