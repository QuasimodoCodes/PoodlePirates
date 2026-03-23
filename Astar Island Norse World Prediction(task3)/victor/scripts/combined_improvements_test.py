"""
combined_improvements_test.py — Test all 4 improvement directions.

1. Smarter alpha: scale observation weight with obs count
2. Round-specific confidence: adaptive N_HIST from shift magnitude
3. Better query targeting: different query allocation strategies
4. Combinations of the above

python -m scripts.combined_improvements_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.001; TEMP = 1.10
N_HIST = 50; N_HIST_SURP = 5; SURP_THRESH = 0.30
random.seed(42)

ALL_A = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
AC = {a: set(_covered_cells(*a)) for a in ALL_A}

# Dense spread: 5 original + 4 interior positions
DENSE_ANCHORS = [
    (1, 1), (24, 1), (1, 24), (25, 25), (12, 12),
    (7, 7), (18, 7), (7, 18), (18, 18),
]

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
def ctx(ig, y, x):
    code=ig[y][x]; sn=cn(ig,y,x,1); on=cn(ig,y,x,10)
    sb="sett_hi" if sn>=3 else ("sett_lo" if sn>=1 else "sett_no")
    ob="ocean" if on>=1 else "inland"
    return (code, sb, ob)
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
def sel_plains_ocean(ig, n=5):
    """Target plains cells near ocean + settlements."""
    targets = set()
    for y in range(40):
        for x in range(40):
            code = ig[y][x]
            if code == 1:  # Settlement
                targets.add((y, x))
            elif code == 11:  # Plains
                if cn(ig, y, x, 10) >= 1:  # Near ocean
                    targets.add((y, x))
    if not targets:
        return SPREAD_ANCHORS[:n]
    cov, sel = set(), []
    for _ in range(n):
        b, bc = None, -1
        for a in ALL_A:
            c = len((AC[a] & targets) - cov)
            if c > bc: bc, b = c, a
        if not b or bc <= 0: break
        sel.append(b); cov |= (AC[b] & targets)
    return sel
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
def shift_mag(R):
    return max(abs(r - 1.0) for r in R)


# ── Query strategies ─────────────────────────────────────────

def get_obs_current(ig, gt):
    """5 settlement + 5 spread (current production)"""
    obs1={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    ao={}
    for a in sel_sett(ig, 5):
        for yx in AC[a]: ao[yx]=[obs1[yx]]
    for a in SPREAD_ANCHORS:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx]=[obs2[yx]]
    return ao

def get_obs_more_spread(ig, gt):
    """3 settlement + 6 dense spread (more coverage)"""
    obs1={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    ao={}
    for a in sel_sett(ig, 3):
        for yx in AC[a]: ao[yx]=[obs1[yx]]
    for a in DENSE_ANCHORS[:6]:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx]=[obs2[yx]]
    return ao

def get_obs_plains_target(ig, gt):
    """5 plains+ocean targets + 5 spread"""
    obs1={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    ao={}
    for a in sel_plains_ocean(ig, 5):
        for yx in AC[a]: ao[yx]=[obs1[yx]]
    for a in SPREAD_ANCHORS:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx]=[obs2[yx]]
    return ao

def get_obs_dense_only(ig, gt):
    """9 dense spread positions, no settlement targeting"""
    obs1={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2={(y,x):so(gt[y][x]) for y in range(40) for x in range(40)}
    ao={}
    for a in DENSE_ANCHORS[:5]:
        for yx in AC[a]: ao[yx]=[obs1[yx]]
    for a in DENSE_ANCHORS[5:]:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx]=[obs2[yx]]
    return ao


# ── Core model ───────────────────────────────────────────────

def run_model(ig, gt, cm_hist, alpha_fn, nhist_mode, query_fn):
    ao = query_fn(ig, gt)
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx(ig, yx[0], yx[1])
            for v in vals: rc[ct].append(v)

    R = compute_global_shift_sqrt(rc, cm_hist)
    sm = shift_mag(R)
    shifted = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}

    # N_HIST selection
    if nhist_mode == "adaptive":
        if sm > 0.3:   nh_base, nh_surp = 25, 3
        elif sm > 0.15: nh_base, nh_surp = 40, 5
        else:           nh_base, nh_surp = 50, 5
    elif nhist_mode == "low30":
        nh_base, nh_surp = 30, 3
    elif nhist_mode == "low20":
        nh_base, nh_surp = 20, 3
    else:
        nh_base, nh_surp = N_HIST, N_HIST_SURP

    bl = {}
    for ct in set(list(shifted.keys()) + list(rc.keys())):
        h = shifted.get(ct, [1.0/N]*N); ol = rc.get(ct, []); nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surp(ol, h)
        nh = nh_surp if s > SURP_THRESH and nr >= 5 else nh_base
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
                ct = ctx(ig, y, x)
                prior = bl.get(ct, shifted.get(ct, [1.0/N]*N))[:]
                if (y, x) in ao:
                    vals = ao[(y,x)]
                    n_obs = len(vals)
                    oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/n_obs
                    a = alpha_fn(n_obs)
                    d = [(1-a)*prior[i] + a*oh[i] for i in range(N)]
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
    print(f"Combined improvements test: {len(files)} files, {len(rounds)} rounds")
    print("=" * 140)

    # ── Alpha functions ──
    def a_fixed(n):   return 0.05
    def a_sqrt(n):    return min(0.05 * math.sqrt(n), 0.20)
    def a_stepped(n): return 0.03 if n == 1 else 0.10
    def a_gentle(n):  return 0.03 if n == 1 else 0.07
    def a_count(n):   return n / (n + 20.0)

    strategies = [
        # label              alpha_fn   nhist_mode  query_fn
        ("A.baseline",       a_fixed,   "fixed",    get_obs_current),
        # --- Smarter alpha ---
        ("B.a_sqrt",         a_sqrt,    "fixed",    get_obs_current),
        ("C.a_stepped",      a_stepped, "fixed",    get_obs_current),
        ("D.a_gentle",       a_gentle,  "fixed",    get_obs_current),
        # --- Round-specific confidence ---
        ("E.adapt_nh",       a_fixed,   "adaptive", get_obs_current),
        ("F.low_nh30",       a_fixed,   "low30",    get_obs_current),
        # --- Query targeting ---
        ("G.more_sprd",      a_fixed,   "fixed",    get_obs_more_spread),
        ("H.plains_tgt",     a_fixed,   "fixed",    get_obs_plains_target),
        ("I.dense_only",     a_fixed,   "fixed",    get_obs_dense_only),
        # --- Best combinations ---
        ("J.gentle+adpt",    a_gentle,  "adaptive", get_obs_current),
        ("K.step+plains",    a_stepped, "fixed",    get_obs_plains_target),
        ("L.all3",           a_gentle,  "adaptive", get_obs_plains_target),
    ]

    labels = [s[0] for s in strategies]

    # Print header in two rows (6 strategies each) for readability
    half = 6
    for group_start in range(0, len(labels), half):
        group = labels[group_start:group_start + half]
        print(f"\n  {'Round':<10}", end="")
        for l in group: print(f" {l:>14}", end="")
        print()
        print(f"  {'-'*10}", end="")
        for _ in group: print(f" {'-'*14}", end="")
        print()

        totals_g = {l: [] for l in group}

        for test_rid in round_ids:
            # Build conditional matrix (same for all strategies)
            cond_acc = defaultdict(list)
            for f in files:
                if f.split("_seed")[0] == test_rid: continue
                d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
                if not gt or not ig: continue
                for y in range(40):
                    for x in range(40):
                        c = ig[y][x]
                        if c not in STATIC_CODES:
                            cond_acc[ctx(ig, y, x)].append(gt[y][x])
            cm_hist = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in cond_acc.items()}

            # Save random state for fair comparison
            rng_state = random.getstate()

            round_scores = {l: [] for l in group}
            for fname in rounds[test_rid]:
                d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
                if not gt or not ig: continue
                for label, alpha_fn, nhist_mode, query_fn in strategies[group_start:group_start + half]:
                    random.setstate(rng_state)  # fair comparison
                    sc = run_model(ig, gt, cm_hist, alpha_fn, nhist_mode, query_fn)
                    round_scores[label].append(sc)

            short = test_rid[:8]
            print(f"  {short:<10}", end="")
            for l in group:
                avg = sum(round_scores[l]) / len(round_scores[l])
                totals_g[l].extend(round_scores[l])
                print(f" {avg:14.1f}", end="")
            print()

        # Store totals for final summary
        if group_start == 0:
            totals = dict(totals_g)
        else:
            totals.update(totals_g)

    # ── Final summary ──
    print()
    print("=" * 140)
    baseline_avg = sum(totals[labels[0]]) / len(totals[labels[0]])

    print(f"\n  {'Config':<18} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Max':>7} {'Std':>7}  Direction")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*20}")
    directions = [
        "baseline", "alpha", "alpha", "alpha",
        "confidence", "confidence",
        "query", "query", "query",
        "combo", "combo", "combo",
    ]
    for (label, _, _, _), direction in zip(strategies, directions):
        scores = totals[label]
        avg = sum(scores) / len(scores)
        mn, mx = min(scores), max(scores)
        std = (sum((s - avg)**2 for s in scores) / len(scores)) ** 0.5
        delta = avg - baseline_avg
        marker = " <<<" if delta > 0.3 else (" *" if delta > 0.1 else "")
        print(f"  {label:<18} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {mx:7.1f} {std:7.1f}  {direction}{marker}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
