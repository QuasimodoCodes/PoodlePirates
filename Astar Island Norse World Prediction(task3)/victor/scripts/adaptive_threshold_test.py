"""
adaptive_threshold_test.py — Find the best signal + threshold to switch
between OLD (fixed alpha) and NEW (stepped alpha + plains targeting)
on a per-round basis.

Tests signals: shift_magnitude, n_surprised, min(R), combined
For each threshold, selects best strategy per round and reports overall score.

python -m scripts.adaptive_threshold_test
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

CLASS_NAMES = ["Empty", "Settl", "Port", "Ruin", "Forest", "Mtn"]

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
def sel_priority(ig, n=5):
    targets = set()
    for y in range(40):
        for x in range(40):
            code = ig[y][x]
            if code == 1: targets.add((y, x))
            elif code == 11:
                if cn(ig, y, x, 10) >= 1: targets.add((y, x))
    if not targets: return SPREAD_ANCHORS[:n]
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


def run_model_with_signals(ig, gt, cm_hist, alpha_fn, phase1_fn):
    """Run model and also return difficulty signals."""
    ao = get_obs(ig, gt, phase1_fn)
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx(ig, yx[0], yx[1])
            for v in vals: rc[ct].append(v)

    R = compute_global_shift_sqrt(rc, cm_hist)
    shifted = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}

    # Count surprised buckets
    n_surprised = 0
    avg_surprise = 0.0
    n_buckets_with_obs = 0
    bl = {}
    for ct in set(list(shifted.keys()) + list(rc.keys())):
        h = shifted.get(ct, [1.0/N]*N); ol = rc.get(ct, []); nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surp(ol, h)
        n_buckets_with_obs += 1
        avg_surprise += s
        if s > SURP_THRESH and nr >= 5:
            n_surprised += 1
            nh = N_HIST_SURP
        else:
            nh = N_HIST
        rf = [0.0]*N
        for v in ol: rf[v] += 1.0/nr
        t = nr + nh
        bl[ct] = [(nr*rf[i] + nh*h[i])/t for i in range(N)]

    avg_surprise = avg_surprise / max(n_buckets_with_obs, 1)

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

    sc = score_t(tensor, gt)
    signals = {
        "shift_mag": max(abs(r - 1.0) for r in R),
        "n_surprised": n_surprised,
        "min_R": min(R),
        "avg_surprise": avg_surprise,
        "shift_R": R,
    }
    return sc, signals


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    round_ids = sorted(rounds.keys())
    print(f"Adaptive threshold test: {len(files)} files, {len(rounds)} rounds")
    print("=" * 110)

    def alpha_old(n): return 0.05
    def alpha_new(n): return 0.03 if n == 1 else 0.10

    # Collect per-round data
    round_data = []  # [(round_id, avg_old, avg_new, signals)]

    for test_rid in round_ids:
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

        old_scores, new_scores = [], []
        all_signals = []

        for fname in rounds[test_rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue

            rng_state = random.getstate()

            random.setstate(rng_state)
            sc_old, sig_old = run_model_with_signals(ig, gt, cm_hist, alpha_old, sel_sett)
            old_scores.append(sc_old)

            random.setstate(rng_state)
            sc_new, sig_new = run_model_with_signals(ig, gt, cm_hist, alpha_new, sel_priority)
            new_scores.append(sc_new)
            all_signals.append(sig_new)

        avg_old = sum(old_scores) / len(old_scores)
        avg_new = sum(new_scores) / len(new_scores)

        # Average signals across seeds
        avg_sig = {
            "shift_mag": sum(s["shift_mag"] for s in all_signals) / len(all_signals),
            "n_surprised": sum(s["n_surprised"] for s in all_signals) / len(all_signals),
            "min_R": sum(s["min_R"] for s in all_signals) / len(all_signals),
            "avg_surprise": sum(s["avg_surprise"] for s in all_signals) / len(all_signals),
        }

        round_data.append((test_rid, avg_old, avg_new, old_scores, new_scores, avg_sig))

    # ── Print per-round summary ──
    print(f"\n  {'Round':<10} {'OLD':>7} {'NEW':>7} {'Delta':>7} {'ShiftMag':>9} {'NSurp':>6} {'MinR':>6} {'AvgSurp':>8} {'Better':>7}")
    print(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*6} {'-'*6} {'-'*8} {'-'*7}")
    for rid, avg_old, avg_new, _, _, sig in round_data:
        delta = avg_new - avg_old
        better = "NEW" if delta > 0.05 else ("OLD" if delta < -0.05 else "TIE")
        print(f"  {rid[:8]:<10} {avg_old:7.2f} {avg_new:7.2f} {delta:+7.2f} {sig['shift_mag']:9.3f} {sig['n_surprised']:6.1f} {sig['min_R']:6.3f} {sig['avg_surprise']:8.3f} {better:>7}")

    # ── Sweep thresholds ──
    print(f"\n{'='*110}")
    print(f"  THRESHOLD SWEEP")
    print(f"{'='*110}")

    # Test different signals and thresholds
    # Signal: if signal > threshold -> use NEW, else use OLD
    signal_configs = [
        ("shift_mag",   [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80]),
        ("n_surprised",  [0, 1, 2, 3, 4, 5]),
        ("avg_surprise", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        ("min_R",        []),  # use < instead of > for min_R
    ]
    # Also test: min_R < threshold -> NEW (detect disappearing classes)
    min_R_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    all_old_avg = sum(d[1] for d in round_data) / len(round_data)
    all_new_avg = sum(d[2] for d in round_data) / len(round_data)

    print(f"\n  Always OLD: {all_old_avg:.2f}")
    print(f"  Always NEW: {all_new_avg:.2f}")

    # Oracle (always pick better per-round)
    oracle_total = sum(max(d[1], d[2]) for d in round_data) / len(round_data)
    print(f"  Oracle:     {oracle_total:.2f}")

    best_config = None
    best_score = all_old_avg

    print(f"\n  {'Signal':<15} {'Thresh':>7} {'Avg':>7} {'Delta':>7} {'N_use_NEW':>10} {'Rounds using NEW'}")
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*10} {'-'*40}")

    for signal_name, thresholds in signal_configs:
        for thresh in thresholds:
            total = 0.0
            n_new = 0
            new_rounds = []
            for rid, avg_old, avg_new, _, _, sig in round_data:
                if sig[signal_name] > thresh:
                    total += avg_new
                    n_new += 1
                    new_rounds.append(rid[:8])
                else:
                    total += avg_old
            avg = total / len(round_data)
            delta = avg - all_old_avg
            marker = " <<<" if delta > 0.15 else (" *" if delta > 0.05 else "")
            if avg > best_score:
                best_score = avg
                best_config = (signal_name, ">", thresh)
            print(f"  {signal_name:<15} {thresh:>7.2f} {avg:7.2f} {delta:+7.2f} {n_new:>10}  {', '.join(new_rounds)}{marker}")

    # min_R: use NEW when min_R < threshold (detecting disappearing classes)
    for thresh in min_R_thresholds:
        total = 0.0
        n_new = 0
        new_rounds = []
        for rid, avg_old, avg_new, _, _, sig in round_data:
            if sig["min_R"] < thresh:
                total += avg_new
                n_new += 1
                new_rounds.append(rid[:8])
            else:
                total += avg_old
        avg = total / len(round_data)
        delta = avg - all_old_avg
        marker = " <<<" if delta > 0.15 else (" *" if delta > 0.05 else "")
        if avg > best_score:
            best_score = avg
            best_config = ("min_R", "<", thresh)
        print(f"  {'min_R<':<15} {thresh:>7.2f} {avg:7.2f} {delta:+7.2f} {n_new:>10}  {', '.join(new_rounds)}{marker}")

    # ── Combined signals ──
    print(f"\n  Combined signals:")
    # shift_mag > X AND n_surprised >= Y
    for sm_t in [0.30, 0.35, 0.40]:
        for ns_t in [1, 2, 3]:
            total = 0.0
            n_new = 0
            new_rounds = []
            for rid, avg_old, avg_new, _, _, sig in round_data:
                if sig["shift_mag"] > sm_t and sig["n_surprised"] >= ns_t:
                    total += avg_new
                    n_new += 1
                    new_rounds.append(rid[:8])
                else:
                    total += avg_old
            avg = total / len(round_data)
            delta = avg - all_old_avg
            marker = " <<<" if delta > 0.15 else (" *" if delta > 0.05 else "")
            if avg > best_score:
                best_score = avg
                best_config = (f"sm>{sm_t}+ns>={ns_t}", "combined", 0)
            print(f"  sm>{sm_t:.2f}+ns>={ns_t}    {avg:7.2f} {delta:+7.2f} {n_new:>10}  {', '.join(new_rounds)}{marker}")

    # shift_mag > X AND avg_surprise > Y
    for sm_t in [0.30, 0.35, 0.40]:
        for as_t in [0.10, 0.15, 0.20]:
            total = 0.0
            n_new = 0
            new_rounds = []
            for rid, avg_old, avg_new, _, _, sig in round_data:
                if sig["shift_mag"] > sm_t and sig["avg_surprise"] > as_t:
                    total += avg_new
                    n_new += 1
                    new_rounds.append(rid[:8])
                else:
                    total += avg_old
            avg = total / len(round_data)
            delta = avg - all_old_avg
            marker = " <<<" if delta > 0.15 else (" *" if delta > 0.05 else "")
            if avg > best_score:
                best_score = avg
                best_config = (f"sm>{sm_t}+as>{as_t}", "combined", 0)
            print(f"  sm>{sm_t:.2f}+as>{as_t:.2f}   {avg:7.2f} {delta:+7.2f} {n_new:>10}  {', '.join(new_rounds)}{marker}")

    print(f"\n  Best config: {best_config} -> {best_score:.2f} ({best_score - all_old_avg:+.2f} vs always OLD)")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
