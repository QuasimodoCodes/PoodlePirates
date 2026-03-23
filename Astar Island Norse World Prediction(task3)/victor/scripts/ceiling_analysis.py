"""
ceiling_analysis.py — What's our theoretical score ceiling?

Compute oracle scores to understand where the gap is:
  PROD:       Current production model
  ORACLE_CM:  Perfect conditional matrix (from test round's own data, LOO by seed)
  ORACLE_OBS: If we could observe every dynamic cell (1 observation each)
  ORACLE_BOTH: Perfect CM + all cells observed
  BUCKET_2X:  Double the context buckets (finer-grained spatial bins)

python -m scripts.ceiling_analysis
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
def bin_od_fine(d):
    if d <= 1: return "od0"
    if d <= 2: return "od1a"
    if d <= 4: return "od1b"
    if d <= 7: return "od2a"
    if d <= 10: return "od2b"
    if d <= 15: return "od3a"
    return "od3b"
def bin_sd_fine(d):
    if d <= 1: return "sd_adj"
    if d <= 2: return "sd_close"
    if d <= 4: return "sd_near"
    if d <= 7: return "sd_mid"
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

def ctx_fn_fine(ig, y, x, odm, sdm):
    """Finer-grained context: 7 ocean bins x 6 sett bins for Plains/Forest."""
    code = ig[y][x]; sn = cn(ig, y, x, 1)
    sb = "sh" if sn >= 3 else ("sl" if sn >= 1 else "sn")
    if code in (1, 4, 11) and odm and sdm:
        return (code, bin_sd_fine(sdm[y][x]), bin_od_fine(odm[y][x]))
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


def build_tensor(ig, gt, cm, odm, sdm, ao, ctx_func=None):
    """Build prediction tensor with given CM, obs, and context function."""
    if ctx_func is None:
        ctx_func = ctx_fn

    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_func(ig, yx[0], yx[1], odm, sdm)
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
                ct = ctx_func(ig, y, x, odm, sdm)
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
    return tensor


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

    print(f"\nCeiling Analysis: {len(files)} files, {len(rounds)} rounds")
    print("=" * 100)
    print(f"  {'Round':<10} {'PROD':>8} {'ORACLE':>8} {'ALL_OBS':>8} {'ORC+OBS':>8} {'FINE_BK':>8} {'NO_SHIFT':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    all_scores = defaultdict(list)

    random.seed(42)

    for trid in round_ids:
        # Standard CM (LOO, excluding test round)
        ca_std = defaultdict(list)
        ca_fine = defaultdict(list)
        for f in files:
            if f.split("_seed")[0] == trid: continue
            d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(f); sdm = sdm_c.get(f)
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in STATIC_CODES:
                        ca_std[ctx_fn(ig, y, x, odm, sdm)].append(gt[y][x])
                        ca_fine[ctx_fn_fine(ig, y, x, odm, sdm)].append(gt[y][x])
        cm_std = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in ca_std.items()}
        cm_fine = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in ca_fine.items()}

        # Oracle CM: built from OTHER seeds of SAME round (perfect round-specific info)
        round_files = rounds[trid]

        round_scores = defaultdict(list)

        for fname in round_files:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(fname); sdm = sdm_c.get(fname)

            # Build oracle CM from other seeds of same round
            ca_oracle = defaultdict(list)
            for other in round_files:
                if other == fname: continue
                od = all_data[other]; ogt = od.get("ground_truth"); oig = od.get("initial_grid")
                if not ogt or not oig: continue
                oodm = odm_c.get(other); osdm = sdm_c.get(other)
                for y in range(40):
                    for x in range(40):
                        c = oig[y][x]
                        if c not in STATIC_CODES:
                            ca_oracle[ctx_fn(oig, y, x, oodm, osdm)].append(ogt[y][x])
            cm_oracle = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in ca_oracle.items()}

            rng = random.getstate()

            # PROD: standard CM + standard obs
            random.setstate(rng)
            ao = get_obs(ig, gt)
            tensor = build_tensor(ig, gt, cm_std, odm, sdm, ao)
            round_scores["PROD"].append(score_t(tensor, gt))

            # ORACLE_CM: oracle CM + standard obs
            random.setstate(rng)
            ao = get_obs(ig, gt)
            tensor = build_tensor(ig, gt, cm_oracle, odm, sdm, ao)
            round_scores["ORACLE"].append(score_t(tensor, gt))

            # ALL_OBS: standard CM + observe every cell
            random.setstate(rng)
            ao_all = {}
            for y in range(40):
                for x in range(40):
                    if ig[y][x] not in STATIC_CODES:
                        ao_all[(y, x)] = [so(gt[y][x])]
            tensor = build_tensor(ig, gt, cm_std, odm, sdm, ao_all)
            round_scores["ALL_OBS"].append(score_t(tensor, gt))

            # ORACLE + ALL_OBS
            random.setstate(rng)
            ao_all2 = {}
            for y in range(40):
                for x in range(40):
                    if ig[y][x] not in STATIC_CODES:
                        ao_all2[(y, x)] = [so(gt[y][x])]
            tensor = build_tensor(ig, gt, cm_oracle, odm, sdm, ao_all2)
            round_scores["ORC+OBS"].append(score_t(tensor, gt))

            # FINE_BK: finer bucketing + standard obs
            random.setstate(rng)
            ao = get_obs(ig, gt)
            tensor = build_tensor(ig, gt, cm_fine, odm, sdm, ao, ctx_func=ctx_fn_fine)
            round_scores["FINE_BK"].append(score_t(tensor, gt))

            # NO_SHIFT: standard CM, no global shift, standard obs
            random.setstate(rng)
            ao = get_obs(ig, gt)
            # Build tensor without shift
            rc = defaultdict(list)
            for yx, vals in ao.items():
                c = ig[yx[0]][yx[1]]
                if c not in STATIC_CODES:
                    ct = ctx_fn(ig, yx[0], yx[1], odm, sdm)
                    for v in vals: rc[ct].append(v)
            # No shift - use cm_std directly
            bl = {}
            for ct in set(list(cm_std.keys()) + list(rc.keys())):
                h = cm_std.get(ct, [1.0/N]*N)
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
                        pc = CODE_TO_CLASS[code]; dd = [FS]*N; dd[pc] = 1-5*FS
                    else:
                        ct = ctx_fn(ig, y, x, odm, sdm)
                        prior = bl.get(ct, cm_std.get(ct, [1.0/N]*N))[:]
                        if (y, x) in ao:
                            vals = ao[(y, x)]; no = len(vals); oh = [0.0]*N
                            for v in vals: oh[v] += 1.0/no
                            dd = [(1-ALPHA)*prior[i] + ALPHA*oh[i] for i in range(N)]
                        else: dd = prior[:]
                        dd = ts(dd, TEMP)
                        dd = [max(v, FD) for v in dd]
                    t = sum(dd); row.append([v/t for v in dd])
                tensor.append(row)
            round_scores["NO_SHIFT"].append(score_t(tensor, gt))

        short = trid[:8]
        avgs = {}
        for k in ["PROD", "ORACLE", "ALL_OBS", "ORC+OBS", "FINE_BK", "NO_SHIFT"]:
            sc = round_scores[k]
            avgs[k] = sum(sc)/len(sc) if sc else 0
            all_scores[k].extend(sc)

        best = max(avgs, key=avgs.get)
        line = f"  {short:<10}"
        for k in ["PROD", "ORACLE", "ALL_OBS", "ORC+OBS", "FINE_BK", "NO_SHIFT"]:
            m = "*" if k == best else " "
            line += f" {avgs[k]:7.1f}{m}"
        print(line)

    # Summary
    print(f"\n{'='*100}")
    prod_avg = sum(all_scores["PROD"]) / len(all_scores["PROD"])
    print(f"\n  {'Strategy':<12} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Max':>7} {'Std':>7}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for k in ["PROD", "ORACLE", "ALL_OBS", "ORC+OBS", "FINE_BK", "NO_SHIFT"]:
        sc = all_scores[k]
        avg = sum(sc)/len(sc)
        mn = min(sc)
        mx = max(sc)
        std = (sum((v-avg)**2 for v in sc)/len(sc))**0.5
        delta = avg - prod_avg
        print(f"  {k:<12} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {mx:7.1f} {std:7.1f}")

    print(f"\n  INTERPRETATION:")
    oracle_gap = sum(all_scores["ORACLE"])/len(all_scores["ORACLE"]) - prod_avg
    obs_gap = sum(all_scores["ALL_OBS"])/len(all_scores["ALL_OBS"]) - prod_avg
    both_gap = sum(all_scores["ORC+OBS"])/len(all_scores["ORC+OBS"]) - prod_avg
    print(f"  Oracle CM (perfect round priors) adds:   {oracle_gap:+.2f} pts")
    print(f"  Observing ALL cells adds:                {obs_gap:+.2f} pts")
    print(f"  Both combined adds:                      {both_gap:+.2f} pts")
    print(f"  Finer bucketing (42 bins -> ~80):         {sum(all_scores['FINE_BK'])/len(all_scores['FINE_BK'])-prod_avg:+.2f} pts")
    print(f"  Removing global shift:                   {sum(all_scores['NO_SHIFT'])/len(all_scores['NO_SHIFT'])-prod_avg:+.2f} pts")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
