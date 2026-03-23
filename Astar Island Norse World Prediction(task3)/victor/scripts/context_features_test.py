"""
context_features_test.py — Test richer context features to split high-loss buckets.

Current context: (terrain_code, sett_bin[3], ocean_bin[2]) = 19 buckets
Problem: (Plains, sett_lo, ocean) = 20% of loss, avgKL=0.109

Test adding:
  - Forest neighbor count bins
  - Ocean neighbor count (graded, not just 0/1)
  - Settlement neighbor count (finer bins)
  - Distance to nearest coast cell
  - Combined features

python -m scripts.context_features_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.001; ALPHA = 0.05; TEMP = 1.10
N_HIST = 50; N_HIST_SURP = 5; SURP_THRESH = 0.30
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
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dy==0 and dx==0: continue
            ny, nx = y+dy, x+dx
            if 0<=ny<40 and 0<=nx<40 and ig[ny][nx]==tc: c+=1
    return c
def sel_sett(ig, n=5):
    ss={(y,x) for y in range(40) for x in range(40) if ig[y][x]==1}
    if not ss: return SPREAD_ANCHORS[:n]
    cov,sel=set(),[]
    for _ in range(n):
        b,bc=None,-1
        for a in ALL_A:
            c=len((AC[a]&ss)-cov)
            if c>bc: bc,b=c,a
        if not b or bc<=0: break
        sel.append(b); cov|=(AC[b]&ss)
    return sel
def get_obs(ig, gt):
    obs1={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    ao={}
    for a in sel_sett(ig,5):
        for yx in AC[a]: ao[yx]=[obs1[yx]]
    for a in SPREAD_ANCHORS:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx]=[obs2[yx]]
    return ao
def ts(d, t):
    if t==1.0: s=sum(d); return [v/s for v in d]
    ld=[math.log(max(p,1e-12))/t for p in d]; mx=max(ld)
    ed=[math.exp(v-mx) for v in ld]; s=sum(ed)
    return [v/s for v in ed]
def surp(ol, h):
    nr=len(ol)
    if nr<3: return 0.0
    rf=[0.0]*N
    for v in ol: rf[v]+=1.0/nr
    kf=sum(rf[i]*math.log(max(rf[i],1e-12)/max(h[i],1e-12)) for i in range(N) if rf[i]>1e-12)
    kr=sum(h[i]*math.log(max(h[i],1e-12)/max(rf[i],1e-12)) for i in range(N) if h[i]>1e-12)
    return (kf+kr)/2
def compute_global_shift_sqrt(rc, cm_hist):
    round_total=[0.0]*N; n_obs=0
    for ct,ol in rc.items():
        for cls in ol: round_total[cls]+=1; n_obs+=1
    if n_obs==0: return [1.0]*N
    round_freq=[c/n_obs for c in round_total]
    hist_total=[0.0]*N; nb=0
    for ct,dist in cm_hist.items():
        for i in range(N): hist_total[i]+=dist[i]
        nb+=1
    if nb==0: return [1.0]*N
    hist_freq=[h/nb for h in hist_total]
    return [math.sqrt(round_freq[i]/hist_freq[i]) if hist_freq[i]>1e-8 else 1.0 for i in range(N)]
def apply_shift(dist, R):
    shifted=[max(dist[i]*R[i],1e-12) for i in range(N)]
    s=sum(shifted); return [v/s for v in shifted]


# ── Context functions ───────────────────────────────────────────

def ctx_current(ig, y, x):
    """Current: (code, sett_bin[3], ocean_bin[2]) = ~19 buckets"""
    code = ig[y][x]
    sn = cn(ig, y, x, 1)
    on = cn(ig, y, x, 10)
    sb = "sh" if sn >= 3 else ("sl" if sn >= 1 else "sn")
    ob = "oc" if on >= 1 else "in"
    return (code, sb, ob)

def ctx_ocean_graded(ig, y, x):
    """Split ocean into 3 bins: 0, 1-3, 4+"""
    code = ig[y][x]
    sn = cn(ig, y, x, 1)
    on = cn(ig, y, x, 10)
    sb = "sh" if sn >= 3 else ("sl" if sn >= 1 else "sn")
    ob = "o3" if on >= 4 else ("o2" if on >= 1 else "in")
    return (code, sb, ob)

def ctx_forest_added(ig, y, x):
    """Add forest neighbor bin: (code, sett, ocean, forest_bin)"""
    code = ig[y][x]
    sn = cn(ig, y, x, 1)
    on = cn(ig, y, x, 10)
    fn = cn(ig, y, x, 4)
    sb = "sh" if sn >= 3 else ("sl" if sn >= 1 else "sn")
    ob = "oc" if on >= 1 else "in"
    fb = "fh" if fn >= 6 else ("fl" if fn >= 2 else "fn")
    return (code, sb, ob, fb)

def ctx_ocean_forest(ig, y, x):
    """Graded ocean + forest bins"""
    code = ig[y][x]
    sn = cn(ig, y, x, 1)
    on = cn(ig, y, x, 10)
    fn = cn(ig, y, x, 4)
    sb = "sh" if sn >= 3 else ("sl" if sn >= 1 else "sn")
    ob = "o3" if on >= 4 else ("o2" if on >= 1 else "in")
    fb = "fh" if fn >= 6 else "fl" if fn >= 2 else "fn"
    return (code, sb, ob, fb)

def ctx_fine_sett(ig, y, x):
    """Finer settlement bins: 0, 1, 2-3, 4+"""
    code = ig[y][x]
    sn = cn(ig, y, x, 1)
    on = cn(ig, y, x, 10)
    if sn >= 4: sb = "s4"
    elif sn >= 2: sb = "s2"
    elif sn >= 1: sb = "s1"
    else: sb = "s0"
    ob = "oc" if on >= 1 else "in"
    return (code, sb, ob)

def ctx_coast_dist(ig, y, x):
    """Add distance-to-coast feature"""
    code = ig[y][x]
    sn = cn(ig, y, x, 1)
    on = cn(ig, y, x, 10)
    sb = "sh" if sn >= 3 else ("sl" if sn >= 1 else "sn")
    ob = "oc" if on >= 1 else "in"
    # Distance to nearest ocean (r=1 already captured by ob)
    # Check r=2 ocean for "near coast" vs "deep inland"
    on2 = cn(ig, y, x, 10, r=3)
    if ob == "in":
        cb = "nc" if on2 >= 1 else "di"  # near coast vs deep inland
    else:
        cb = "co"  # coastal
    return (code, sb, cb)

def ctx_minimal(ig, y, x):
    """Minimal: just (code, ocean_bin) — fewer buckets, more data per bucket"""
    code = ig[y][x]
    on = cn(ig, y, x, 10)
    ob = "oc" if on >= 1 else "in"
    return (code, ob)

def ctx_sett_ocean_fine(ig, y, x):
    """Fine sett + graded ocean"""
    code = ig[y][x]
    sn = cn(ig, y, x, 1)
    on = cn(ig, y, x, 10)
    if sn >= 4: sb = "s4"
    elif sn >= 2: sb = "s2"
    elif sn >= 1: sb = "s1"
    else: sb = "s0"
    ob = "o3" if on >= 4 else ("o2" if on >= 1 else "in")
    return (code, sb, ob)


# ── Model runner ────────────────────────────────────────────────

def run_model(ig, gt, cm_hist, ctx_fn):
    ao = get_obs(ig, gt)
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_fn(ig, yx[0], yx[1])
            for v in vals: rc[ct].append(v)

    R = compute_global_shift_sqrt(rc, cm_hist)
    shifted = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}
    bl = {}
    for ct in set(list(shifted.keys()) + list(rc.keys())):
        h = shifted.get(ct, [1.0/N]*N); ol = rc.get(ct, []); nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surp(ol, h)
        nh = N_HIST_SURP if s > SURP_THRESH and nr >= 5 else N_HIST
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
                ct = ctx_fn(ig, y, x)
                prior = bl.get(ct, shifted.get(ct, [1.0/N]*N))[:]
                if (y, x) in ao:
                    vals = ao[(y,x)]; oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/len(vals)
                    d = [(1-ALPHA)*prior[i] + ALPHA*oh[i] for i in range(N)]
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

    round_ids = sorted(rounds.keys())
    print(f"Context features test: {len(files)} files, {len(rounds)} rounds")
    print("=" * 120)

    strategies = [
        ("A.current",       ctx_current),
        ("B.ocean_grad",    ctx_ocean_graded),
        ("C.+forest",       ctx_forest_added),
        ("D.oc+forest",     ctx_ocean_forest),
        ("E.fine_sett",     ctx_fine_sett),
        ("F.coast_dist",    ctx_coast_dist),
        ("G.minimal",       ctx_minimal),
        ("H.sett_oc_fine",  ctx_sett_ocean_fine),
    ]

    labels = [s[0] for s in strategies]
    print(f"\n  {'Round':<10}", end="")
    for l in labels: print(f" {l:>14}", end="")
    print()
    print(f"  {'-'*10}", end="")
    for _ in labels: print(f" {'-'*14}", end="")
    print()

    totals = {l: [] for l in labels}
    bucket_counts = {l: [] for l in labels}

    for test_rid in round_ids:
        # Build CM per strategy
        cms = {}
        for label, ctx_fn in strategies:
            cond_acc = defaultdict(list)
            for f in files:
                if f.split("_seed")[0] == test_rid: continue
                d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
                if not gt or not ig: continue
                for y in range(40):
                    for x in range(40):
                        c = ig[y][x]
                        if c not in STATIC_CODES:
                            cond_acc[ctx_fn(ig, y, x)].append(gt[y][x])
            cm = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in cond_acc.items()}
            cms[label] = cm
            bucket_counts[label].append(len(cm))

        round_scores = {l: [] for l in labels}
        for fname in rounds[test_rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            for label, ctx_fn in strategies:
                sc = run_model(ig, gt, cms[label], ctx_fn)
                round_scores[label].append(sc)

        short = test_rid[:8]
        print(f"  {short:<10}", end="")
        for l in labels:
            avg = sum(round_scores[l]) / len(round_scores[l])
            totals[l].extend(round_scores[l])
            print(f" {avg:14.1f}", end="")
        print()

    print()
    print("=" * 120)
    baseline_avg = sum(totals[labels[0]]) / len(totals[labels[0]])

    print(f"\n  {'Config':<18} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Std':>7} {'Buckets':>8}")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for label, ctx_fn in strategies:
        scores = totals[label]
        avg = sum(scores) / len(scores)
        mn = min(scores)
        std = (sum((s - avg)**2 for s in scores) / len(scores)) ** 0.5
        delta = avg - baseline_avg
        avg_buckets = sum(bucket_counts[label]) / len(bucket_counts[label])
        marker = " <<<" if delta > 0.3 else (" *" if delta > 0.1 else "")
        print(f"  {label:<18} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {std:7.1f} {avg_buckets:>7.0f}{marker}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
