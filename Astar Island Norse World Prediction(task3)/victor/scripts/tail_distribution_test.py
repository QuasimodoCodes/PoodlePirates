"""
tail_distribution_test.py — Test improved tail distribution modeling.

76% of our KL loss comes from tail (non-dominant) classes. We're getting the
dominant class right but not allocating enough to minority outcomes.

Strategies tested:
  PROD:      Current production (TEMP=1.10 uniform)
  PER_TEMP:  Per-terrain temperature (higher for uncertain terrains)
  ENT_CAL:   Entropy calibration (match historical entropy per bucket)
  VAR_FLOOR: Variance-aware floor (higher floor for high-variance buckets)
  SPATIAL:   Spatial smoothing (propagate obs info to neighbors)
  COMBINED:  Best of the above combined

python -m scripts.tail_distribution_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; ALPHA = 0.05
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


def run_model(ig, gt, cm, odm, sdm, ao, strategy="PROD", cm_entropy=None, cm_variance=None):
    """Run prediction model with given strategy."""
    FD = 0.001
    TEMP = 1.10

    # Per-terrain temperature settings
    TERRAIN_TEMPS = {
        1: 1.40,   # Settlement: avg GT entropy 1.03, very uncertain
        2: 1.40,   # Port: avg GT entropy 1.15, very uncertain
        4: 1.15,   # Forest: avg GT entropy 0.58, moderate
        11: 1.08,  # Plains: avg GT entropy 0.49, fairly certain
        0: 1.20,   # Empty: moderate
        3: 1.20,   # Ruin: moderate
    }

    # Strategy-specific floors
    VAR_FLOORS = {}  # filled from cm_variance if strategy uses it

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

    # For SPATIAL strategy: build neighbor observation map
    obs_neighbor = {}
    if strategy in ("SPATIAL", "COMBINED"):
        # For each unobserved cell, aggregate observations from nearby observed cells
        observed_set = set(ao.keys())
        for y in range(40):
            for x in range(40):
                if (y, x) in observed_set: continue
                code = ig[y][x]
                if code in STATIC_CODES: continue
                # Collect observations from cells within radius 3 with same terrain code
                nearby_obs = []
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < 40 and 0 <= nx < 40 and (ny, nx) in ao:
                            if ig[ny][nx] == code:
                                # Weight by inverse distance
                                d = abs(dy) + abs(dx)
                                if d == 0: continue
                                nearby_obs.extend(ao[(ny, nx)])
                if nearby_obs:
                    obs_neighbor[(y, x)] = nearby_obs

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
                    vals = ao[(y, x)]; no = len(vals); oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/no
                    d = [(1-ALPHA)*prior[i] + ALPHA*oh[i] for i in range(N)]
                elif strategy in ("SPATIAL", "COMBINED") and (y, x) in obs_neighbor:
                    # Soft spatial blend for unobserved cells
                    nobs = obs_neighbor[(y, x)]
                    no = len(nobs); oh = [0.0]*N
                    for v in nobs: oh[v] += 1.0/no
                    # Much weaker blend than direct observation
                    spatial_alpha = 0.02
                    d = [(1-spatial_alpha)*prior[i] + spatial_alpha*oh[i] for i in range(N)]
                else:
                    d = prior[:]

                # Temperature scaling
                if strategy == "PER_TEMP":
                    t = TERRAIN_TEMPS.get(code, 1.10)
                    d = ts(d, t)
                elif strategy == "COMBINED":
                    t = TERRAIN_TEMPS.get(code, 1.10)
                    d = ts(d, t)
                else:
                    d = ts(d, TEMP)

                # Entropy calibration: adjust distribution to match historical entropy
                if strategy in ("ENT_CAL", "COMBINED") and cm_entropy:
                    target_ent = cm_entropy.get(ct)
                    if target_ent is not None and target_ent > 0.1:
                        # Binary search for temperature that matches target entropy
                        cur_ent = _entropy(d)
                        if cur_ent < target_ent * 0.8:
                            # Need to soften more
                            lo, hi = 1.0, 3.0
                            base = bl.get(ct, shifted.get(ct, [1.0/N]*N))[:]
                            for _ in range(10):
                                mid = (lo + hi) / 2
                                trial = ts(base, mid)
                                if _entropy(trial) < target_ent:
                                    lo = mid
                                else:
                                    hi = mid
                            d = ts(base, (lo + hi) / 2)

                # Variance-aware floor
                if strategy in ("VAR_FLOOR", "COMBINED") and cm_variance:
                    var = cm_variance.get(ct, 0)
                    # Higher variance -> higher floor (more hedging)
                    if var > 0.02:
                        floor = 0.005
                    elif var > 0.01:
                        floor = 0.003
                    else:
                        floor = FD
                    d = [max(v, floor) for v in d]
                else:
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

    strategies = ["PROD", "PER_TEMP", "ENT_CAL", "VAR_FLOOR", "SPATIAL", "COMBINED"]

    # Run 3 seeds for robustness
    SEEDS = [42, 123, 456]

    print(f"\nTail Distribution Test: {len(files)} files, {len(rounds)} rounds, {len(SEEDS)} seeds")
    print(f"PROD:      TEMP=1.10 uniform, FD=0.001")
    print(f"PER_TEMP:  Settlement/Port=1.40, Forest=1.15, Plains=1.08")
    print(f"ENT_CAL:   Calibrate distribution entropy to historical average")
    print(f"VAR_FLOOR: Higher floor for high-variance context buckets")
    print(f"SPATIAL:   Propagate observations to nearby unobserved cells")
    print(f"COMBINED:  PER_TEMP + ENT_CAL + VAR_FLOOR + SPATIAL")
    print("=" * 100)

    totals = {s: [] for s in strategies}

    for trid in round_ids:
        # Build training data (LOO)
        ca = defaultdict(list)
        ca_entropy = defaultdict(list)  # per-bucket entropy tracking
        ca_variance = defaultdict(list)  # per-bucket variance tracking
        for f in files:
            if f.split("_seed")[0] == trid: continue
            d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(f); sdm = sdm_c.get(f)
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in STATIC_CODES:
                        ct = ctx_fn(ig, y, x, odm, sdm)
                        ca[ct].append(gt[y][x])
                        ca_entropy[ct].append(_entropy(gt[y][x]))

        cm = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in ca.items()}

        # Compute per-bucket average entropy and variance
        cm_entropy = {ct: sum(ents)/len(ents) for ct, ents in ca_entropy.items()}
        cm_variance = {}
        for ct, samples in ca.items():
            avg = cm[ct]
            # Average per-class variance
            var = sum(sum((s[i] - avg[i])**2 for s in samples)/len(samples) for i in range(N)) / N
            cm_variance[ct] = var

        round_scores = {s: [] for s in strategies}

        for fname in rounds[trid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(fname); sdm = sdm_c.get(fname)

            for seed in SEEDS:
                rng_state = random.getstate()
                random.seed(seed + all_data[fname].get("seed_index", 0))

                ao = get_obs(ig, gt)

                for strat in strategies:
                    random.setstate(random.getstate())  # same obs for all
                    sc = run_model(ig, gt, cm, odm, sdm, ao, strategy=strat,
                                   cm_entropy=cm_entropy, cm_variance=cm_variance)
                    round_scores[strat].append(sc)

                random.setstate(rng_state)

        avgs = {s: sum(round_scores[s])/len(round_scores[s]) for s in strategies}
        for s in strategies:
            totals[s].extend(round_scores[s])

        short = trid[:8]
        best = max(avgs, key=avgs.get)
        prod_avg = avgs["PROD"]
        line = f"  {short:<10}"
        for s in strategies:
            delta = avgs[s] - prod_avg
            marker = "*" if s == best else " "
            line += f" {avgs[s]:7.1f}{marker}"
        print(line)

    # Summary
    print(f"\n{'='*100}")
    prod_avg = sum(totals["PROD"]) / len(totals["PROD"])
    print(f"\n  {'Strategy':<12} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Max':>7} {'Std':>7} {'Wins':>5}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")
    for s in strategies:
        scores = totals[s]
        avg = sum(scores)/len(scores)
        mn = min(scores)
        mx = max(scores)
        std = (sum((v - avg)**2 for v in scores)/len(scores))**0.5
        wins = sum(1 for a, b in zip(totals["PROD"], scores) if b > a)
        delta = avg - prod_avg
        marker = " <<<" if delta > 0.3 else ""
        print(f"  {s:<12} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {mx:7.1f} {std:7.1f} {wins:>5}{marker}")

    # Per-terrain KL breakdown for PROD vs best alternative
    best_strat = max(strategies[1:], key=lambda s: sum(totals[s])/len(totals[s]))
    print(f"\n  Best alternative: {best_strat}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
