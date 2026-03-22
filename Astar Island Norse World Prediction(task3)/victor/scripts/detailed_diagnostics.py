"""
detailed_diagnostics.py — Deep diagnostic analysis of prediction errors.

Logs per-bucket, per-terrain, per-class, observed vs unobserved breakdowns
to understand exactly where we lose points and what to tune.

python -m scripts.detailed_diagnostics
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.001; ALPHA = 0.05; TEMP = 1.10
TF = {1: .001, 2: .001, 3: .001, 4: .001, 11: .001, 0: .001}
N_HIST = 50; N_HIST_SURP = 5; SURP_THRESH = 0.30
random.seed(42)

ALL_A = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
AC = {a: set(_covered_cells(*a)) for a in ALL_A}

CLASS_NAMES = ["Empty", "Settl", "Port", "Ruin", "Forest", "Mtn"]
CODE_NAMES = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest", 11: "Plains"}


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

def surp(ol, h):
    nr = len(ol)
    if nr < 3: return 0.0
    rf = [0.0] * N
    for v in ol: rf[v] += 1.0 / nr
    kf = sum(rf[i] * math.log(max(rf[i], 1e-12) / max(h[i], 1e-12)) for i in range(N) if rf[i] > 1e-12)
    kr = sum(h[i] * math.log(max(h[i], 1e-12) / max(rf[i], 1e-12)) for i in range(N) if h[i] > 1e-12)
    return (kf + kr) / 2

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

def compute_global_shift_sqrt(rc, cm_hist):
    round_total = [0.0] * N; n_obs = 0
    for ct, ol in rc.items():
        for cls in ol: round_total[cls] += 1; n_obs += 1
    if n_obs == 0: return [1.0] * N
    round_freq = [c / n_obs for c in round_total]
    hist_total = [0.0] * N; nb = 0
    for ct, dist in cm_hist.items():
        for i in range(N): hist_total[i] += dist[i]
        nb += 1
    if nb == 0: return [1.0] * N
    hist_freq = [h / nb for h in hist_total]
    return [math.sqrt(round_freq[i] / hist_freq[i]) if hist_freq[i] > 1e-8 else 1.0 for i in range(N)]

def apply_shift(dist, R):
    shifted = [max(dist[i] * R[i], 1e-12) for i in range(N)]
    s = sum(shifted)
    return [v / s for v in shifted]


def run_with_diagnostics(ig, gt, cm_hist):
    """Run full NEW pipeline and collect detailed diagnostics."""
    ao = get_obs(ig, gt)

    # Collect per-bucket obs
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx(ig, yx[0], yx[1])
            for v in vals: rc[ct].append(v)

    # Global shift + calibrate
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

    # Build tensor + collect diagnostics
    diag = {
        "per_bucket": defaultdict(lambda: {"kl": [], "wkl": [], "n": 0, "obs": 0, "unobs": 0}),
        "per_code": defaultdict(lambda: {"kl": [], "wkl": [], "n": 0}),
        "observed": {"kl": [], "wkl": []},
        "unobserved": {"kl": [], "wkl": []},
        "per_class_error": defaultdict(lambda: {"over": [], "under": [], "n": 0}),
        "confidence_bins": defaultdict(lambda: {"kl": [], "n": 0, "gt_avg": [], "pred_avg": []}),
        "floor_impact": {"cells_floored": 0, "total_dynamic": 0, "floor_kl_cost": 0.0},
        "temp_impact": {"pre_temp_kl": [], "post_temp_kl": []},
    }

    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS]*N; d[pc] = 1-5*FS
                row.append(d)
                continue

            ct = ctx(ig, y, x)
            prior = bl.get(ct, shifted.get(ct, [1.0/N]*N))[:]
            is_obs = (y, x) in ao

            if is_obs:
                vals = ao[(y,x)]; oh = [0.0]*N
                for v in vals: oh[v] += 1.0/len(vals)
                d = [(1-ALPHA)*prior[i] + ALPHA*oh[i] for i in range(N)]
            else:
                d = prior[:]

            # Pre-temperature KL
            pre_temp = d[:]
            pre_s = sum(pre_temp)
            pre_temp = [v/pre_s for v in pre_temp]
            pre_kl = kl(gt[y][x], pre_temp)

            # Apply temperature
            d = ts(d, TEMP)

            post_temp = d[:]
            post_s = sum(post_temp)
            post_temp = [v/post_s for v in post_temp]
            post_kl = kl(gt[y][x], post_temp)

            diag["temp_impact"]["pre_temp_kl"].append(pre_kl)
            diag["temp_impact"]["post_temp_kl"].append(post_kl)

            # Check floor impact
            diag["floor_impact"]["total_dynamic"] += 1
            any_floored = False
            fl = TF.get(code, FD)
            for i in range(N):
                if d[i] < fl:
                    any_floored = True
                    d[i] = fl
            if any_floored:
                diag["floor_impact"]["cells_floored"] += 1

            t = sum(d); d = [v/t for v in d]

            # Compute cell-level metrics
            e = _entropy(gt[y][x])
            cell_kl = kl(gt[y][x], d)
            cell_wkl = e * cell_kl

            # Per-bucket
            bd = diag["per_bucket"][ct]
            bd["kl"].append(cell_kl)
            bd["wkl"].append(cell_wkl)
            bd["n"] += 1
            if is_obs: bd["obs"] += 1
            else: bd["unobs"] += 1

            # Per-code
            cd = diag["per_code"][code]
            cd["kl"].append(cell_kl)
            cd["wkl"].append(cell_wkl)
            cd["n"] += 1

            # Observed vs unobserved
            tag = "observed" if is_obs else "unobserved"
            diag[tag]["kl"].append(cell_kl)
            diag[tag]["wkl"].append(cell_wkl)

            # Per-class prediction error
            for i in range(N):
                err = d[i] - gt[y][x][i]
                bucket = diag["per_class_error"][i]
                bucket["n"] += 1
                if err > 0:
                    bucket["over"].append(err)
                else:
                    bucket["under"].append(abs(err))

            # Confidence bins: group by max(prediction)
            max_pred = max(d)
            if max_pred > 0.9: bin_label = "0.9+"
            elif max_pred > 0.8: bin_label = "0.8-0.9"
            elif max_pred > 0.6: bin_label = "0.6-0.8"
            elif max_pred > 0.4: bin_label = "0.4-0.6"
            else: bin_label = "<0.4"
            cb = diag["confidence_bins"][bin_label]
            cb["kl"].append(cell_kl)
            cb["n"] += 1
            cb["gt_avg"].append(max(gt[y][x]))
            cb["pred_avg"].append(max_pred)

            row.append(d)
        tensor.append(row)

    sc = score_t(tensor, gt)
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
    print(f"Detailed diagnostics: {len(files)} files, {len(rounds)} rounds")
    print("=" * 90)

    # Accumulators
    all_scores = []
    acc_bucket = defaultdict(lambda: {"kl": [], "wkl": [], "n": 0, "obs": 0, "unobs": 0})
    acc_code = defaultdict(lambda: {"kl": [], "wkl": [], "n": 0})
    acc_obs = {"kl": [], "wkl": []}
    acc_unobs = {"kl": [], "wkl": []}
    acc_class_err = defaultdict(lambda: {"over": [], "under": [], "n": 0})
    acc_conf = defaultdict(lambda: {"kl": [], "n": 0, "gt_avg": [], "pred_avg": []})
    acc_floor = {"cells_floored": 0, "total_dynamic": 0}
    acc_temp = {"pre": [], "post": []}

    # Per-round tracking
    round_scores = {}
    round_shift = {}

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

        seed_scores = []
        for fname in rounds[test_rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue

            sc, diag = run_with_diagnostics(ig, gt, cm_hist)
            seed_scores.append(sc)
            all_scores.append(sc)

            # Merge diagnostics
            for ct, bd in diag["per_bucket"].items():
                a = acc_bucket[ct]
                a["kl"].extend(bd["kl"]); a["wkl"].extend(bd["wkl"])
                a["n"] += bd["n"]; a["obs"] += bd["obs"]; a["unobs"] += bd["unobs"]
            for code, cd in diag["per_code"].items():
                a = acc_code[code]
                a["kl"].extend(cd["kl"]); a["wkl"].extend(cd["wkl"]); a["n"] += cd["n"]
            acc_obs["kl"].extend(diag["observed"]["kl"])
            acc_obs["wkl"].extend(diag["observed"]["wkl"])
            acc_unobs["kl"].extend(diag["unobserved"]["kl"])
            acc_unobs["wkl"].extend(diag["unobserved"]["wkl"])
            for i, ce in diag["per_class_error"].items():
                a = acc_class_err[i]
                a["over"].extend(ce["over"]); a["under"].extend(ce["under"]); a["n"] += ce["n"]
            for bl, cb in diag["confidence_bins"].items():
                a = acc_conf[bl]
                a["kl"].extend(cb["kl"]); a["n"] += cb["n"]
                a["gt_avg"].extend(cb["gt_avg"]); a["pred_avg"].extend(cb["pred_avg"])
            acc_floor["cells_floored"] += diag["floor_impact"]["cells_floored"]
            acc_floor["total_dynamic"] += diag["floor_impact"]["total_dynamic"]
            acc_temp["pre"].extend(diag["temp_impact"]["pre_temp_kl"])
            acc_temp["post"].extend(diag["temp_impact"]["post_temp_kl"])

        avg = sum(seed_scores) / len(seed_scores)
        round_scores[test_rid] = avg

    # ═══════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════

    overall_avg = sum(all_scores) / len(all_scores)
    print(f"\n  Overall avg score: {overall_avg:.2f} ({len(all_scores)} seeds)")

    # 1. Per-round scores
    print(f"\n{'='*90}")
    print("  1. PER-ROUND SCORES")
    print(f"{'='*90}")
    for rid in round_ids:
        sc = round_scores[rid]
        bar = "#" * int(sc)
        print(f"  {rid[:8]} {sc:6.1f}  {bar}")

    # 2. Per-terrain code breakdown
    print(f"\n{'='*90}")
    print("  2. PER-TERRAIN CODE LOSS BREAKDOWN")
    print(f"{'='*90}")
    total_wkl = sum(sum(a["wkl"]) for a in acc_code.values())
    print(f"  {'Code':<12} {'Cells':>8} {'AvgKL':>8} {'TotalWKL':>10} {'%Loss':>7} {'AvgEnt':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*8}")
    for code in sorted(acc_code.keys()):
        a = acc_code[code]
        avg_kl = sum(a["kl"]) / len(a["kl"])
        twkl = sum(a["wkl"])
        pct = 100 * twkl / total_wkl if total_wkl > 0 else 0
        avg_ent = sum(a["wkl"]) / sum(a["kl"]) if sum(a["kl"]) > 0 else 0
        name = CODE_NAMES.get(code, f"code{code}")
        print(f"  {name:<12} {a['n']:>8} {avg_kl:>8.4f} {twkl:>10.1f} {pct:>6.1f}% {avg_ent:>8.3f}")

    # 3. Observed vs unobserved
    print(f"\n{'='*90}")
    print("  3. OBSERVED vs UNOBSERVED CELLS")
    print(f"{'='*90}")
    obs_avg = sum(acc_obs["kl"]) / len(acc_obs["kl"]) if acc_obs["kl"] else 0
    unobs_avg = sum(acc_unobs["kl"]) / len(acc_unobs["kl"]) if acc_unobs["kl"] else 0
    obs_wkl = sum(acc_obs["wkl"])
    unobs_wkl = sum(acc_unobs["wkl"])
    print(f"  Observed:   {len(acc_obs['kl']):>8} cells  avgKL={obs_avg:.4f}  totalWKL={obs_wkl:.1f} ({100*obs_wkl/total_wkl:.1f}%)")
    print(f"  Unobserved: {len(acc_unobs['kl']):>8} cells  avgKL={unobs_avg:.4f}  totalWKL={unobs_wkl:.1f} ({100*unobs_wkl/total_wkl:.1f}%)")
    print(f"  Gap: unobs - obs = {unobs_avg - obs_avg:+.4f} avgKL")

    # 4. Per-bucket breakdown (top 12)
    print(f"\n{'='*90}")
    print("  4. TOP 12 LOSS BUCKETS (by total weighted KL)")
    print(f"{'='*90}")
    bucket_sorted = sorted(acc_bucket.keys(), key=lambda c: -sum(acc_bucket[c]["wkl"]))
    print(f"  {'Bucket':<38} {'Cells':>6} {'Obs%':>5} {'AvgKL':>7} {'TotWKL':>8} {'%Loss':>6}")
    print(f"  {'-'*38} {'-'*6} {'-'*5} {'-'*7} {'-'*8} {'-'*6}")
    for ct in bucket_sorted[:12]:
        a = acc_bucket[ct]
        avg_kl = sum(a["kl"]) / len(a["kl"])
        twkl = sum(a["wkl"])
        pct = 100 * twkl / total_wkl
        obs_pct = 100 * a["obs"] / a["n"] if a["n"] > 0 else 0
        print(f"  {str(ct):<38} {a['n']:>6} {obs_pct:>4.0f}% {avg_kl:>7.4f} {twkl:>8.1f} {pct:>5.1f}%")

    # 5. Confidence calibration
    print(f"\n{'='*90}")
    print("  5. CONFIDENCE CALIBRATION (pred confidence vs actual)")
    print(f"{'='*90}")
    print(f"  {'Bin':<10} {'Cells':>8} {'AvgKL':>8} {'PredConf':>10} {'GTConf':>10} {'Gap':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for bl in sorted(acc_conf.keys()):
        a = acc_conf[bl]
        avg_kl = sum(a["kl"]) / len(a["kl"])
        avg_pred = sum(a["pred_avg"]) / len(a["pred_avg"])
        avg_gt = sum(a["gt_avg"]) / len(a["gt_avg"])
        print(f"  {bl:<10} {a['n']:>8} {avg_kl:>8.4f} {avg_pred:>10.3f} {avg_gt:>10.3f} {avg_gt-avg_pred:>+8.3f}")

    # 6. Per-class prediction error direction
    print(f"\n{'='*90}")
    print("  6. PER-CLASS PREDICTION ERROR (over vs under prediction)")
    print(f"{'='*90}")
    print(f"  {'Class':<8} {'Cells':>8} {'AvgOver':>9} {'AvgUnder':>9} {'Bias':>8} {'Direction':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*10}")
    for i in range(N):
        a = acc_class_err[i]
        avg_over = sum(a["over"]) / len(a["over"]) if a["over"] else 0
        avg_under = sum(a["under"]) / len(a["under"]) if a["under"] else 0
        n_over = len(a["over"])
        n_under = len(a["under"])
        total_over = sum(a["over"])
        total_under = sum(a["under"])
        bias = total_over - total_under
        direction = "OVER" if bias > 0 else "UNDER"
        print(f"  {CLASS_NAMES[i]:<8} {a['n']:>8} {avg_over:>9.4f} {avg_under:>9.4f} {bias/a['n']:>+8.4f} {direction:>10}")

    # 7. Temperature impact
    print(f"\n{'='*90}")
    print("  7. TEMPERATURE IMPACT (pre vs post temperature scaling)")
    print(f"{'='*90}")
    pre_avg = sum(acc_temp["pre"]) / len(acc_temp["pre"])
    post_avg = sum(acc_temp["post"]) / len(acc_temp["post"])
    print(f"  Pre-temperature  avg KL: {pre_avg:.5f}")
    print(f"  Post-temperature avg KL: {post_avg:.5f}")
    print(f"  Temperature effect:      {post_avg - pre_avg:+.5f} ({'helps' if post_avg < pre_avg else 'hurts'})")

    # 8. Floor impact
    print(f"\n{'='*90}")
    print("  8. FLOOR IMPACT")
    print(f"{'='*90}")
    pct_floored = 100 * acc_floor["cells_floored"] / acc_floor["total_dynamic"] if acc_floor["total_dynamic"] > 0 else 0
    print(f"  Cells where any class was floored: {acc_floor['cells_floored']} / {acc_floor['total_dynamic']} ({pct_floored:.1f}%)")

    # 9. What would it take to reach 90?
    print(f"\n{'='*90}")
    print("  9. PATH TO 90 POINTS")
    print(f"{'='*90}")
    target = 90
    target_wkl = -math.log(target / 100) / 3
    current_wkl_per_cell = total_wkl / sum(a["n"] for a in acc_code.values())
    total_entropy = total_wkl / current_wkl_per_cell if current_wkl_per_cell > 0 else 1
    # Actually recompute: score = 100*exp(-3*wkl/total_ent)
    # We need sum(e*kl)/sum(e) < target_wkl
    print(f"  Current avg weighted KL: {current_wkl_per_cell:.5f}")
    print(f"  Target for {target} pts:     {target_wkl:.5f}")
    print(f"  Reduction needed:        {current_wkl_per_cell / target_wkl:.1f}x")
    print(f"")

    # Decompose: where could we save?
    print(f"  Loss decomposition by category:")
    obs_share = obs_wkl / total_wkl * 100
    unobs_share = unobs_wkl / total_wkl * 100
    print(f"    Observed cells:   {obs_share:.1f}% of loss ({len(acc_obs['kl'])} cells)")
    print(f"    Unobserved cells: {unobs_share:.1f}% of loss ({len(acc_unobs['kl'])} cells)")
    print(f"")

    # Which buckets have the most room for improvement?
    print(f"  Biggest loss buckets vs theoretical minimum:")
    for ct in bucket_sorted[:6]:
        a = acc_bucket[ct]
        avg_kl = sum(a["kl"]) / len(a["kl"])
        twkl = sum(a["wkl"])
        # Theoretical min: if prediction = GT mean, KL comes from GT variance only
        # We can't do better than the cross-round variance
        print(f"    {str(ct):<38} totalWKL={twkl:>7.1f}  avgKL={avg_kl:.4f}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
