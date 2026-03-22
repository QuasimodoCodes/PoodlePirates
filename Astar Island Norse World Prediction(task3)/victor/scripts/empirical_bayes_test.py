"""
empirical_bayes_test.py — Test Empirical Bayes posterior update vs current alpha blending.

Theory (Gemini):
  The optimal prediction when observing class k is:
    q_i = E[p_i * p_k] / E[p_k]
  where expectations are over the historical ground truth distributions.
  This automatically scales trust based on how informative each observation class is.

We test at bucket level (context buckets) since per-cell history is too sparse.

Strategies:
  A. Baseline: current alpha=0.05 linear blend
  B. Empirical Bayes: q_i = second_moment[ctx][i][k] / prior[ctx][k]
  C. EB + global shift (combine Rec 2 + Rec 3)
  D. EB with fallback to alpha when bucket has <20 history samples
  E. EB + sqrt global shift
  F. Alpha=0.15 (higher trust in obs, for comparison)

python -m scripts.empirical_bayes_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.005; TEMP = 1.10
TF = {1: .008, 2: .008, 3: .006, 4: .003, 11: .004, 0: .005}
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


def build_second_moment(history_data, files, exclude_rid, ctx_fn):
    """Build E[p_i * p_k] and E[p_k] per context bucket from history.

    second_moment[ctx][i][k] = average of (gt_dist[i] * gt_dist[k]) over all cells in bucket
    first_moment[ctx][i] = average of gt_dist[i] = the prior

    Returns (first_moment, second_moment, bucket_counts)
    """
    # Accumulate
    sm_acc = defaultdict(lambda: [[0.0]*N for _ in range(N)])  # [ctx][i][k]
    fm_acc = defaultdict(lambda: [0.0]*N)  # [ctx][i]
    counts = defaultdict(int)

    for f in files:
        if f.split("_seed")[0] == exclude_rid:
            continue
        d = history_data[f]
        gt, ig = d.get("ground_truth"), d.get("initial_grid")
        if not gt or not ig:
            continue
        for y in range(40):
            for x in range(40):
                c = ig[y][x]
                if c in STATIC_CODES:
                    continue
                ct = ctx_fn(ig, y, x)
                dist = gt[y][x]
                counts[ct] += 1
                for i in range(N):
                    fm_acc[ct][i] += dist[i]
                    for k in range(N):
                        sm_acc[ct][i][k] += dist[i] * dist[k]

    # Normalize
    first_moment = {}
    second_moment = {}
    for ct in counts:
        n = counts[ct]
        first_moment[ct] = [fm_acc[ct][i] / n for i in range(N)]
        second_moment[ct] = [[sm_acc[ct][i][k] / n for k in range(N)] for i in range(N)]

    return first_moment, second_moment, dict(counts)


def eb_update(first_moment, second_moment, obs_class):
    """Empirical Bayes posterior: q_i = E[p_i * p_k] / E[p_k] where k=obs_class."""
    pk = first_moment[obs_class]  # E[p_k]
    if pk < 1e-12:
        return first_moment[:]  # fallback to prior
    q = [second_moment[i][obs_class] / pk for i in range(N)]
    # Ensure valid distribution
    s = sum(q)
    if s < 1e-12:
        return first_moment[:]
    return [v / s for v in q]


def eb_update_multi(first_moment, second_moment, obs_classes):
    """EB update for multiple observations: average the per-observation posteriors."""
    if not obs_classes:
        return first_moment[:]
    posteriors = []
    for k in obs_classes:
        posteriors.append(eb_update(first_moment, second_moment, k))
    # Average
    avg = [sum(p[i] for p in posteriors) / len(posteriors) for i in range(N)]
    s = sum(avg)
    return [v / s for v in avg]


def compute_global_shift(ao, ig, cm_hist):
    round_counts = [0.0] * N
    total_obs = 0
    for yx, vals in ao.items():
        code = ig[yx[0]][yx[1]]
        if code in STATIC_CODES:
            continue
        for v in vals:
            round_counts[v] += 1
            total_obs += 1
    if total_obs == 0:
        return [1.0] * N
    round_freq = [c / total_obs for c in round_counts]
    hist_total = [0.0] * N
    hist_count = 0
    for ct, dist in cm_hist.items():
        for i in range(N):
            hist_total[i] += dist[i]
        hist_count += 1
    if hist_count == 0:
        return [1.0] * N
    hist_freq = [h / hist_count for h in hist_total]
    R = [1.0] * N
    for i in range(N):
        if hist_freq[i] > 1e-8:
            R[i] = round_freq[i] / hist_freq[i]
    return R


def apply_shift(dist, R, power=1.0):
    shifted = [max(dist[i] * (R[i] ** power), 1e-12) for i in range(N)]
    s = sum(shifted)
    return [v / s for v in shifted]


def blend_buckets(ao, ig, cm, n_hist=50, n_hist_surp=5):
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx(ig, yx[0], yx[1])
            for v in vals: rc[ct].append(v)
    bl = {}
    for ct in set(list(cm.keys()) + list(rc.keys())):
        h = cm.get(ct, [1/N]*N); ol = rc.get(ct, []); nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surp(ol, h)
        nh = n_hist_surp if s > .3 and nr >= 5 else n_hist
        rf = [0.0]*N
        for v in ol: rf[v] += 1.0/nr
        t = nr + nh
        bl[ct] = [(nr*rf[i] + nh*h[i])/t for i in range(N)]
    return bl


def build_tensor_alpha(ig, ao, bl, cm, alpha):
    """Current approach: linear alpha blend for observed cells."""
    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS]*N; d[pc] = 1-5*FS
            else:
                ct = ctx(ig, y, x)
                prior = bl.get(ct, cm.get(ct, [1/N]*N))[:]
                if (y, x) in ao:
                    vals = ao[(y,x)]; oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/len(vals)
                    d = [(1-alpha)*prior[i] + alpha*oh[i] for i in range(N)]
                else:
                    d = prior[:]
                d = ts(d, TEMP)
                fl = TF.get(code, FD); d = [max(v, fl) for v in d]
            t = sum(d); row.append([v/t for v in d])
        tensor.append(row)
    return tensor


def build_tensor_eb(ig, ao, bl, cm, fm_ctx, sm_ctx, bucket_counts, min_samples=20, alpha_fallback=0.05):
    """Empirical Bayes: use EB update for observed cells in well-sampled buckets."""
    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS]*N; d[pc] = 1-5*FS
            else:
                ct = ctx(ig, y, x)
                bucket_prior = bl.get(ct, cm.get(ct, [1/N]*N))[:]

                if (y, x) in ao and ct in sm_ctx and bucket_counts.get(ct, 0) >= min_samples:
                    # Use Empirical Bayes update
                    d = eb_update_multi(fm_ctx[ct], sm_ctx[ct], ao[(y, x)])
                elif (y, x) in ao:
                    # Fallback to alpha blend for small buckets
                    vals = ao[(y,x)]; oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/len(vals)
                    d = [(1-alpha_fallback)*bucket_prior[i] + alpha_fallback*oh[i] for i in range(N)]
                else:
                    d = bucket_prior[:]
                d = ts(d, TEMP)
                fl = TF.get(code, FD); d = [max(v, fl) for v in d]
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

    round_ids = sorted(rounds.keys())
    n_rounds = len(rounds)
    print(f"Empirical Bayes test: {len(files)} files, {n_rounds} rounds")
    print("=" * 95)

    strategies = [
        "A.baseline",
        "B.EB",
        "C.EB+shift",
        "D.EB+fb20",
        "E.EB+sqrt",
        "F.alpha15",
    ]

    print(f"\n  {'Round':<12}", end="")
    for s in strategies: print(f" {s:>12}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in strategies: print(f" {'-'*12}", end="")
    print()

    totals = {s: [] for s in strategies}

    for test_rid in round_ids:
        # Build standard conditional matrix (first moment only)
        cond_acc = defaultdict(list)
        for f in files:
            if f.split("_seed")[0] == test_rid: continue
            d = all_data[f]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in STATIC_CODES:
                        cond_acc[ctx(ig, y, x)].append(gt[y][x])
        cm_hist = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in cond_acc.items()}

        # Build second moment matrix for EB
        fm_ctx, sm_ctx, bucket_counts = build_second_moment(all_data, files, test_rid, ctx)

        round_scores = {s: [] for s in strategies}

        for fname in rounds[test_rid]:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue

            ao = get_obs(ig, gt)
            bl = blend_buckets(ao, ig, cm_hist)

            # A. Baseline (alpha=0.05)
            tensor = build_tensor_alpha(ig, ao, bl, cm_hist, 0.05)
            round_scores["A.baseline"].append(score_t(tensor, gt))

            # B. Empirical Bayes (no fallback threshold)
            tensor = build_tensor_eb(ig, ao, bl, cm_hist, fm_ctx, sm_ctx, bucket_counts, min_samples=0)
            round_scores["B.EB"].append(score_t(tensor, gt))

            # C. EB + global shift
            R = compute_global_shift(ao, ig, cm_hist)
            shifted_cm = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}
            shifted_fm = {ct: apply_shift(dist, R) for ct, dist in fm_ctx.items()}
            # Shift second moment too: scale sm[i][k] by R[i]*R[k] (approximately)
            shifted_sm = {}
            for ct in sm_ctx:
                shifted_sm[ct] = [[sm_ctx[ct][i][k] * R[i] * R[k] for k in range(N)] for i in range(N)]
                # Renormalize rows
                for i in range(N):
                    rs = sum(shifted_sm[ct][i])
                    if rs > 1e-12:
                        shifted_sm[ct][i] = [v / rs * shifted_fm[ct][i] for v in shifted_sm[ct][i]]
            bl_shifted = blend_buckets(ao, ig, shifted_cm)
            tensor = build_tensor_eb(ig, ao, bl_shifted, shifted_cm, shifted_fm, shifted_sm, bucket_counts, min_samples=0)
            round_scores["C.EB+shift"].append(score_t(tensor, gt))

            # D. EB with fallback for buckets < 20 samples
            tensor = build_tensor_eb(ig, ao, bl, cm_hist, fm_ctx, sm_ctx, bucket_counts, min_samples=20)
            round_scores["D.EB+fb20"].append(score_t(tensor, gt))

            # E. EB + sqrt global shift
            R_sqrt = compute_global_shift(ao, ig, cm_hist)
            shifted_cm_sq = {ct: apply_shift(dist, R_sqrt, power=0.5) for ct, dist in cm_hist.items()}
            bl_sq = blend_buckets(ao, ig, shifted_cm_sq)
            tensor = build_tensor_eb(ig, ao, bl_sq, shifted_cm_sq, fm_ctx, sm_ctx, bucket_counts, min_samples=0)
            round_scores["E.EB+sqrt"].append(score_t(tensor, gt))

            # F. Alpha=0.15 (for comparison)
            tensor = build_tensor_alpha(ig, ao, bl, cm_hist, 0.15)
            round_scores["F.alpha15"].append(score_t(tensor, gt))

        short = test_rid[:8]
        print(f"  {short:<12}", end="")
        for s in strategies:
            avg = sum(round_scores[s]) / len(round_scores[s])
            totals[s].extend(round_scores[s])
            print(f" {avg:12.1f}", end="")
        print()

    # Summary
    print()
    print("=" * 95)
    baseline_avg = sum(totals["A.baseline"]) / len(totals["A.baseline"])

    print(f"\n  {'Strategy':<16} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Std':>7}")
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for s in strategies:
        scores = totals[s]
        avg = sum(scores) / len(scores)
        mn = min(scores)
        std = (sum((sc - avg) ** 2 for sc in scores) / len(scores)) ** 0.5
        delta = avg - baseline_avg
        marker = " <<<" if delta > 0.3 else (" *" if delta > 0.1 else "")
        print(f"  {s:<16} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {std:7.1f}{marker}")

    # Bucket stats
    print(f"\n  Bucket count stats (from last round):")
    for ct in sorted(bucket_counts.keys(), key=lambda c: -bucket_counts[c]):
        print(f"    {str(ct):<40} n={bucket_counts[ct]:>5}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
