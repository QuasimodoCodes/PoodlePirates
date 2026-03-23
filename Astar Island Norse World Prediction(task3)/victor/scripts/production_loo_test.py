"""
production_loo_test.py — Full LOO test of production model with all improvements.

Tests OLD (pre-changes) vs NEW (dist_ocean + adaptive alpha + plains targeting)
across all historical rounds.

python -m scripts.production_loo_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.001; TEMP = 1.10
N_HIST = 50; N_HIST_SURP = 5; SURP_THRESH = 0.30
HARD_THRESH = 5  # n_surprised >= this -> stepped alpha
random.seed(42)

ALL_A = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
AC = {a: set(_covered_cells(*a)) for a in ALL_A}

CLASS_NAMES = ["Empty", "Settl", "Port", "Ruin", "Forest", "Mtn"]


# ── BFS ocean distance ──

def ocean_dist_map(ig):
    H, W = 40, 40
    dist = [[99]*W for _ in range(H)]
    q = []
    for y in range(H):
        for x in range(W):
            if ig[y][x] == 10:
                dist[y][x] = 0; q.append((y, x))
    head = 0
    while head < len(q):
        cy, cx = q[head]; head += 1
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy+dy, cx+dx
            if 0<=ny<H and 0<=nx<W and dist[ny][nx] > dist[cy][cx]+1:
                dist[ny][nx] = dist[cy][cx]+1; q.append((ny, nx))
    return dist

def bin_od(d):
    if d <= 1: return "od0"
    if d <= 4: return "od1"
    if d <= 10: return "od2"
    return "od3"


# ── Helpers ──

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


# ── Context functions ──

def ctx_old(ig, y, x, odm=None):
    """OLD: binary ocean flag (current production before changes)."""
    code=ig[y][x]; sn=cn(ig,y,x,1); on=cn(ig,y,x,10)
    sb="sh" if sn>=3 else ("sl" if sn>=1 else "sn")
    ob="oc" if on>=1 else "in"
    return (code, sb, ob)

def ctx_new(ig, y, x, odm=None):
    """NEW: 4-bin dist_ocean for Plains, binary for others."""
    code=ig[y][x]; sn=cn(ig,y,x,1)
    sb="sh" if sn>=3 else ("sl" if sn>=1 else "sn")
    if code == 11 and odm is not None:
        ob = bin_od(odm[y][x])
    else:
        on=cn(ig,y,x,10)
        ob="oc" if on>=1 else "in"
    return (code, sb, ob)


# ── Query targeting ──

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

def sel_priority(ig, n=5):
    targets=set()
    for y in range(40):
        for x in range(40):
            code=ig[y][x]
            if code==1: targets.add((y,x))
            elif code==11:
                if cn(ig,y,x,10)>=1: targets.add((y,x))
    if not targets: return SPREAD_ANCHORS[:n]
    cov,sel=set(),[]
    for _ in range(n):
        b,bc=None,-1
        for a in ALL_A:
            c=len((AC[a]&targets)-cov)
            if c>bc: bc,b=c,a
        if not b or bc<=0: break
        sel.append(b); cov|=(AC[b]&targets)
    return sel

def get_obs(ig, gt, phase1_fn):
    obs1={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    ao={}
    for a in phase1_fn(ig, 5):
        for yx in AC[a]: ao[yx]=[obs1[yx]]
    for a in SPREAD_ANCHORS:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx]=[obs2[yx]]
    return ao


# ── Model ──

def run_model(ig, gt, cm_hist, ctx_fn, phase1_fn, alpha_mode, odm=None):
    ao = get_obs(ig, gt, phase1_fn)
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_fn(ig, yx[0], yx[1], odm)
            for v in vals: rc[ct].append(v)

    R = compute_global_shift_sqrt(rc, cm_hist)
    shifted = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}

    # Count surprised for adaptive alpha
    n_surprised = 0
    bl = {}
    for ct in set(list(shifted.keys()) + list(rc.keys())):
        h = shifted.get(ct, [1.0/N]*N); ol = rc.get(ct, []); nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surp(ol, h)
        if s > SURP_THRESH and nr >= 5:
            n_surprised += 1
            nh = N_HIST_SURP
        else:
            nh = N_HIST
        rf = [0.0]*N
        for v in ol: rf[v] += 1.0/nr
        t = nr + nh
        bl[ct] = [(nr*rf[i] + nh*h[i])/t for i in range(N)]

    hard = n_surprised >= HARD_THRESH

    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS]*N; d[pc] = 1-5*FS
            else:
                ct = ctx_fn(ig, y, x, odm)
                prior = bl.get(ct, shifted.get(ct, [1.0/N]*N))[:]
                if (y, x) in ao:
                    vals = ao[(y,x)]
                    n_obs = len(vals)
                    oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/n_obs
                    if alpha_mode == "adaptive" and hard:
                        a = 0.10 if n_obs >= 2 else 0.03
                    else:
                        a = 0.05
                    d = [(1-a)*prior[i] + a*oh[i] for i in range(N)]
                else:
                    d = prior[:]
                d = ts(d, TEMP)
                d = [max(v, FD) for v in d]
            t = sum(d); row.append([v/t for v in d])
        tensor.append(row)
    return score_t(tensor, gt), n_surprised


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    round_ids = sorted(rounds.keys())
    print(f"Production LOO test: {len(files)} files, {len(rounds)} rounds")
    print(f"OLD: 19 buckets, alpha=0.05, settlement targeting")
    print(f"NEW: dist_ocean (Plains), adaptive alpha, plains targeting")
    print("=" * 110)

    # Precompute ocean dist maps
    print("Precomputing ocean distance maps...")
    odm_cache = {}
    for f in files:
        ig = all_data[f].get("initial_grid")
        if ig: odm_cache[f] = ocean_dist_map(ig)
    print(f"  Done: {len(odm_cache)} maps.")

    strategies = [
        # label,           ctx_fn,   phase1_fn,     alpha_mode
        ("OLD",            ctx_old,  sel_sett,      "fixed"),
        ("NEW.dist_ocean", ctx_new,  sel_sett,      "fixed"),       # just dist_ocean
        ("NEW.full",       ctx_new,  sel_priority,  "adaptive"),    # dist_ocean + plains targeting + adaptive alpha
    ]
    labels = [s[0] for s in strategies]

    print(f"\n  {'Round':<10}", end="")
    for l in labels: print(f" {l:>14}", end="")
    print(f" {'NSurp':>6} {'Hard?':>6}")
    print(f"  {'-'*10}", end="")
    for _ in labels: print(f" {'-'*14}", end="")
    print(f" {'-'*6} {'-'*6}")

    totals = {l: [] for l in labels}
    n_hard = 0

    for test_rid in round_ids:
        # Build CM per strategy (OLD uses old ctx, NEW uses new ctx with odm)
        cms = {}
        for label, ctx_fn, _, _ in strategies:
            cond_acc = defaultdict(list)
            for f in files:
                if f.split("_seed")[0] == test_rid: continue
                d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
                if not gt or not ig: continue
                odm = odm_cache.get(f)
                for y in range(40):
                    for x in range(40):
                        c = ig[y][x]
                        if c not in STATIC_CODES:
                            cond_acc[ctx_fn(ig, y, x, odm)].append(gt[y][x])
            cms[label] = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in cond_acc.items()}

        round_scores = {l: [] for l in labels}
        round_nsurp = 0

        for fname in rounds[test_rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_cache.get(fname)

            rng_state = random.getstate()
            for label, ctx_fn, phase1_fn, alpha_mode in strategies:
                random.setstate(rng_state)
                sc, ns = run_model(ig, gt, cms[label], ctx_fn, phase1_fn, alpha_mode, odm)
                round_scores[label].append(sc)
                if label == "NEW.full":
                    round_nsurp = max(round_nsurp, ns)

        hard = round_nsurp >= HARD_THRESH
        if hard: n_hard += 1
        short = test_rid[:8]
        print(f"  {short:<10}", end="")
        for l in labels:
            avg = sum(round_scores[l]) / len(round_scores[l])
            totals[l].extend(round_scores[l])
            print(f" {avg:14.1f}", end="")
        print(f" {round_nsurp:>6} {'HARD' if hard else '':>6}")

    print(f"\n{'='*110}")
    baseline_avg = sum(totals["OLD"]) / len(totals["OLD"])

    print(f"\n  {'Config':<18} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Max':>7} {'Std':>7}")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for label, _, _, _ in strategies:
        scores = totals[label]
        avg = sum(scores) / len(scores)
        mn, mx = min(scores), max(scores)
        std = (sum((s - avg)**2 for s in scores) / len(scores)) ** 0.5
        delta = avg - baseline_avg
        marker = " <<<" if delta > 0.5 else (" *" if delta > 0.1 else "")
        print(f"  {label:<18} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {mx:7.1f} {std:7.1f}{marker}")

    print(f"\n  Hard rounds detected: {n_hard}/{len(round_ids)}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
