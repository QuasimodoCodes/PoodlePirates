"""
full_logged_test.py — LOO test of OLD vs NEW pipeline with detailed logging.

OLD: alpha=0.05 fixed, settlement-only targeting
NEW: alpha=0.03/0.10 stepped, plains+ocean targeting (K.step+plains)

Logs per-round:
  - Shift magnitude and direction
  - Observation coverage stats
  - Per-bucket loss breakdown
  - Per-seed scores
  - Where NEW helps vs hurts

python -m scripts.full_logged_test
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
CODE_NAMES = {0: "Empty", 1: "Settl", 2: "Port", 3: "Ruin",
              4: "Forest", 5: "Mtn", 10: "Ocean", 11: "Plains"}


def kl(p, q):
    return sum(pi * math.log(pi / max(qi, 1e-12)) for pi, qi in zip(p, q) if pi > 1e-12)

def weighted_kl_cell(pred, gt):
    """Return (entropy, kl) for a single cell."""
    e = _entropy(gt)
    k = kl(gt, pred)
    return e, k

def score_from_wkl(wkl, te):
    return max(0, min(100, 100 * math.exp(-3 * (wkl / te)))) if te > 1e-12 else 100

def score_t(pred, gt):
    wkl = te = 0.0
    for y in range(40):
        for x in range(40):
            e = _entropy(gt[y][x]); wkl += e * kl(gt[y][x], pred[y][x]); te += e
    return score_from_wkl(wkl, te)

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
    """Settlement-only targeting (OLD)."""
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
    """Settlement + plains-near-ocean targeting (NEW)."""
    targets = set()
    for y in range(40):
        for x in range(40):
            code = ig[y][x]
            if code == 1:
                targets.add((y, x))
            elif code == 11:
                if cn(ig, y, x, 10) >= 1:
                    targets.add((y, x))
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
    """Generate observations with given phase1 targeting function."""
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


def run_model(ig, gt, cm_hist, alpha_fn, phase1_fn, detailed=False):
    """Run model and optionally return detailed diagnostics."""
    ao = get_obs(ig, gt, phase1_fn)
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx(ig, yx[0], yx[1])
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

    # Build tensor and collect diagnostics
    tensor = []
    diag = {"n_obs_1": 0, "n_obs_2": 0, "n_unobs": 0,
            "wkl_obs": 0.0, "te_obs": 0.0, "wkl_unobs": 0.0, "te_unobs": 0.0,
            "bucket_wkl": defaultdict(float), "bucket_te": defaultdict(float)}

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
                    if n_obs >= 2:
                        diag["n_obs_2"] += 1
                    else:
                        diag["n_obs_1"] += 1
                else:
                    d = prior[:]
                    diag["n_unobs"] += 1
                d = ts(d, TEMP)
                d = [max(v, FD) for v in d]
            t = sum(d); d = [v/t for v in d]
            row.append(d)

            # Diagnostics
            if detailed and code not in STATIC_CODES:
                e, k = weighted_kl_cell(d, gt[y][x])
                ct_key = ctx(ig, y, x)
                diag["bucket_wkl"][ct_key] += e * k
                diag["bucket_te"][ct_key] += e
                if (y, x) in ao:
                    diag["wkl_obs"] += e * k
                    diag["te_obs"] += e
                else:
                    diag["wkl_unobs"] += e * k
                    diag["te_unobs"] += e
        tensor.append(row)

    sc = score_t(tensor, gt)
    diag["score"] = sc
    diag["shift_R"] = R
    diag["n_obs_total"] = len(ao)
    return sc, diag


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    round_ids = sorted(rounds.keys())
    print(f"Full logged test: {len(files)} files, {len(rounds)} rounds")
    print(f"OLD: alpha=0.05 fixed, settlement targeting")
    print(f"NEW: alpha=0.03/0.10 stepped, plains+ocean targeting")
    print("=" * 100)

    def alpha_old(n): return 0.05
    def alpha_new(n): return 0.03 if n == 1 else 0.10

    all_old = []
    all_new = []

    for test_rid in round_ids:
        # Build conditional matrix (LOO)
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

        short = test_rid[:8]
        print(f"\n{'─'*100}")
        print(f"  ROUND {short}  ({len(rounds[test_rid])} seeds)")
        print(f"{'─'*100}")

        old_scores = []
        new_scores = []
        old_diags = []
        new_diags = []

        for fname in rounds[test_rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            seed = fname.split("seed")[1].split("_")[0]

            rng_state = random.getstate()

            random.setstate(rng_state)
            sc_old, diag_old = run_model(ig, gt, cm_hist, alpha_old, sel_sett, detailed=True)
            old_scores.append(sc_old)
            old_diags.append(diag_old)

            random.setstate(rng_state)
            sc_new, diag_new = run_model(ig, gt, cm_hist, alpha_new, sel_priority, detailed=True)
            new_scores.append(sc_new)
            new_diags.append(diag_new)

            delta = sc_new - sc_old
            marker = "+++" if delta > 0.5 else ("++" if delta > 0.2 else ("+" if delta > 0 else ("-" if delta > -0.2 else "--")))
            print(f"  seed{seed}: OLD={sc_old:6.2f}  NEW={sc_new:6.2f}  delta={delta:+.2f} {marker}"
                  f"  |  obs: {diag_new['n_obs_1']}x1 + {diag_new['n_obs_2']}x2 = {diag_new['n_obs_total']} cells")

        avg_old = sum(old_scores) / len(old_scores)
        avg_new = sum(new_scores) / len(new_scores)
        delta_avg = avg_new - avg_old
        all_old.extend(old_scores)
        all_new.extend(new_scores)

        # Shift info from first seed
        R = new_diags[0]["shift_R"]
        shift_str = "  ".join(f"{CLASS_NAMES[i]}:{R[i]:.3f}" for i in range(N))
        print(f"  Shift: {shift_str}")
        print(f"  Shift magnitude: {max(abs(r-1) for r in R):.3f}")

        # Observation breakdown from first seed
        d0 = new_diags[0]
        obs_score = score_from_wkl(d0["wkl_obs"], d0["te_obs"]) if d0["te_obs"] > 0 else 100
        unobs_score = score_from_wkl(d0["wkl_unobs"], d0["te_unobs"]) if d0["te_unobs"] > 0 else 100
        print(f"  Obs cells score: {obs_score:.1f}  |  Unobs cells score: {unobs_score:.1f}")

        # Top-loss buckets from first seed (NEW)
        bucket_loss = {}
        for ct in d0["bucket_wkl"]:
            if d0["bucket_te"][ct] > 0.01:
                bucket_loss[ct] = d0["bucket_wkl"][ct]
        top_buckets = sorted(bucket_loss.items(), key=lambda x: -x[1])[:5]
        if top_buckets:
            total_loss = sum(bucket_loss.values())
            print(f"  Top-loss buckets (NEW):")
            for ct, loss in top_buckets:
                code_name = CODE_NAMES.get(ct[0], f"c{ct[0]}")
                pct = 100 * loss / total_loss if total_loss > 0 else 0
                print(f"    ({code_name}, {ct[1]}, {ct[2]}): {pct:.1f}% of loss")

        print(f"  ROUND AVG: OLD={avg_old:.2f}  NEW={avg_new:.2f}  delta={delta_avg:+.2f}")

    # ── Final summary ──
    print(f"\n{'='*100}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*100}")
    avg_o = sum(all_old) / len(all_old)
    avg_n = sum(all_new) / len(all_new)
    std_o = (sum((s - avg_o)**2 for s in all_old) / len(all_old)) ** 0.5
    std_n = (sum((s - avg_n)**2 for s in all_new) / len(all_new)) ** 0.5
    print(f"  OLD:  avg={avg_o:.2f}  std={std_o:.1f}  min={min(all_old):.1f}  max={max(all_old):.1f}")
    print(f"  NEW:  avg={avg_n:.2f}  std={std_n:.1f}  min={min(all_new):.1f}  max={max(all_new):.1f}")
    print(f"  Delta: {avg_n - avg_o:+.2f}")

    # Per-round deltas
    print(f"\n  Per-round deltas (sorted by improvement):")
    round_deltas = []
    idx = 0
    for test_rid in round_ids:
        n_seeds = len(rounds[test_rid])
        old_avg = sum(all_old[idx:idx+n_seeds]) / n_seeds
        new_avg = sum(all_new[idx:idx+n_seeds]) / n_seeds
        round_deltas.append((test_rid[:8], new_avg - old_avg, old_avg, new_avg))
        idx += n_seeds

    round_deltas.sort(key=lambda x: x[1])
    for rid, delta, old, new in round_deltas:
        bar = "+" * max(0, int(delta * 10)) if delta > 0 else "-" * max(0, int(-delta * 10))
        print(f"    {rid}: {delta:+.2f}  (OLD={old:.1f} -> NEW={new:.1f})  {bar}")

    # Win/loss count
    wins = sum(1 for _, d, _, _ in round_deltas if d > 0.05)
    losses = sum(1 for _, d, _, _ in round_deltas if d < -0.05)
    ties = len(round_deltas) - wins - losses
    print(f"\n  Wins: {wins}  Ties: {ties}  Losses: {losses}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
