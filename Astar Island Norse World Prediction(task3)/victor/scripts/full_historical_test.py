"""
full_historical_test.py — Full LOO test comparing old pipeline vs new pipeline.

Simulates the entire prediction flow for each historical round:
  OLD: uncalibrated conditional matrix (the bug) + no global shift
  NEW: calibrated conditional matrix (bug fix) + sqrt global shift

This is the definitive test to measure total improvement.

python -m scripts.full_historical_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.005; ALPHA = 0.05; TEMP = 1.10
TF = {1: .008, 2: .008, 3: .006, 4: .003, 11: .004, 0: .005}
N_HIST = 50; N_HIST_SURP = 5; SURP_THRESH = 0.30
random.seed(42)

ALL_A = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
AC = {a: set(_covered_cells(*a)) for a in ALL_A}


# ── Shared helpers ──────────────────────────────────────────────────────

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


# ── Calibration logic ───────────────────────────────────────────────────

def compute_global_shift_sqrt(round_counts_ctx, cm_hist):
    """Sqrt-damped global shift R[i] = sqrt(round_freq[i] / hist_freq[i])."""
    round_total = [0.0] * N; n_obs = 0
    for ct, obs_list in round_counts_ctx.items():
        for cls in obs_list:
            round_total[cls] += 1; n_obs += 1
    if n_obs == 0:
        return [1.0] * N
    round_freq = [c / n_obs for c in round_total]

    hist_total = [0.0] * N; n_buckets = 0
    for ct, dist in cm_hist.items():
        for i in range(N): hist_total[i] += dist[i]
        n_buckets += 1
    if n_buckets == 0:
        return [1.0] * N
    hist_freq = [h / n_buckets for h in hist_total]

    R = [1.0] * N
    for i in range(N):
        if hist_freq[i] > 1e-8:
            R[i] = math.sqrt(round_freq[i] / hist_freq[i])
    return R


def apply_shift(dist, R):
    shifted = [max(dist[i] * R[i], 1e-12) for i in range(N)]
    s = sum(shifted)
    return [v / s for v in shifted]


def calibrate_cm(ao, ig, cm_hist, use_global_shift=False):
    """Calibrate conditional matrix from round observations.

    Returns calibrated CM dict.
    """
    # Collect per-bucket observations
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx(ig, yx[0], yx[1])
            for v in vals: rc[ct].append(v)

    # Optionally apply global shift to historical priors
    if use_global_shift:
        R = compute_global_shift_sqrt(rc, cm_hist)
        shifted_hist = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}
    else:
        shifted_hist = cm_hist

    # Blend per-bucket
    bl = {}
    for ct in set(list(shifted_hist.keys()) + list(rc.keys())):
        h = shifted_hist.get(ct, [1/N]*N)
        ol = rc.get(ct, []); nr = len(ol)
        if nr == 0:
            bl[ct] = h[:]; continue
        s = surp(ol, h)
        nh = N_HIST_SURP if s > SURP_THRESH and nr >= 5 else N_HIST
        rf = [0.0]*N
        for v in ol: rf[v] += 1.0/nr
        t = nr + nh
        bl[ct] = [(nr*rf[i] + nh*h[i])/t for i in range(N)]
    return bl


def build_tensor(ig, ao, cm_for_unobs, cm_for_obs=None):
    """Build prediction tensor.

    cm_for_unobs: conditional matrix used for unobserved cells
    cm_for_obs: conditional matrix used as prior for observed cells
                (if None, same as cm_for_unobs)
    """
    if cm_for_obs is None:
        cm_for_obs = cm_for_unobs

    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS]*N; d[pc] = 1-5*FS
            else:
                ct = ctx(ig, y, x)
                if (y, x) in ao:
                    prior = cm_for_obs.get(ct, [1/N]*N)[:]
                    vals = ao[(y,x)]; oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/len(vals)
                    d = [(1-ALPHA)*prior[i] + ALPHA*oh[i] for i in range(N)]
                else:
                    d = cm_for_unobs.get(ct, [1/N]*N)[:]
                d = ts(d, TEMP)
                fl = TF.get(code, FD); d = [max(v, fl) for v in d]
            t = sum(d); row.append([v/t for v in d])
        tensor.append(row)
    return tensor


# ── Main ────────────────────────────────────────────────────────────────

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
    print(f"Full historical test: {len(files)} files, {n_rounds} rounds")
    print("=" * 80)

    labels = [
        "OLD(bug+noshift)",
        "BUGFIX_ONLY",
        "SHIFT_ONLY",
        "NEW(fix+shift)",
    ]

    print(f"\n  {'Round':<10}", end="")
    for l in labels: print(f" {l:>18}", end="")
    print("   Note")
    print(f"  {'-'*10}", end="")
    for _ in labels: print(f" {'-'*18}", end="")
    print(f" {'-'*10}")

    totals = {l: [] for l in labels}

    for test_rid in round_ids:
        # Build conditional matrix from all OTHER rounds (historical)
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

        round_scores = {l: [] for l in labels}

        for fname in rounds[test_rid]:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue

            ao = get_obs(ig, gt)

            # OLD: uncalibrated CM for predictions (the bug), no global shift
            bl_old = calibrate_cm(ao, ig, cm_hist, use_global_shift=False)
            # Bug: predictions use cm_hist (uncalibrated) not bl_old
            tensor = build_tensor(ig, ao, cm_for_unobs=cm_hist, cm_for_obs=cm_hist)
            round_scores["OLD(bug+noshift)"].append(score_t(tensor, gt))

            # BUGFIX: calibrated CM used for predictions, no global shift
            tensor = build_tensor(ig, ao, cm_for_unobs=bl_old, cm_for_obs=bl_old)
            round_scores["BUGFIX_ONLY"].append(score_t(tensor, gt))

            # SHIFT: global shift on hist, but still use uncalibrated for predictions (bug)
            bl_shifted = calibrate_cm(ao, ig, cm_hist, use_global_shift=True)
            # Bug still present: use cm_hist for predictions
            cm_shifted_raw = {ct: apply_shift(dist, compute_global_shift_sqrt(
                defaultdict(list, {ctx(ig, yx[0], yx[1]): [v for v in vals]
                                   for yx, vals in ao.items()
                                   if ig[yx[0]][yx[1]] not in STATIC_CODES}),
                cm_hist)) for ct, dist in cm_hist.items()}
            tensor = build_tensor(ig, ao, cm_for_unobs=cm_shifted_raw, cm_for_obs=cm_shifted_raw)
            round_scores["SHIFT_ONLY"].append(score_t(tensor, gt))

            # NEW: calibrated CM with global shift, used for predictions
            tensor = build_tensor(ig, ao, cm_for_unobs=bl_shifted, cm_for_obs=bl_shifted)
            round_scores["NEW(fix+shift)"].append(score_t(tensor, gt))

        # Detect extreme round
        is_extreme = False
        for fname in rounds[test_rid]:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            sett_count = dyn_count = 0
            for y in range(40):
                for x in range(40):
                    if ig[y][x] not in STATIC_CODES:
                        dyn_count += 1
                        if gt[y][x][1] > 0.3: sett_count += 1
            if dyn_count > 0 and sett_count / dyn_count > 0.15:
                is_extreme = True; break

        tag = "EXTREME" if is_extreme else ""
        short = test_rid[:8]
        print(f"  {short:<10}", end="")
        for l in labels:
            avg = sum(round_scores[l]) / len(round_scores[l])
            totals[l].extend(round_scores[l])
            print(f" {avg:18.1f}", end="")
        print(f"   {tag}")

    # Summary
    print()
    print("=" * 80)
    old_avg = sum(totals["OLD(bug+noshift)"]) / len(totals["OLD(bug+noshift)"])
    new_avg = sum(totals["NEW(fix+shift)"]) / len(totals["NEW(fix+shift)"])

    print(f"\n  {'Pipeline':<20} {'Avg':>7} {'vs OLD':>7} {'Min':>7} {'Max':>7} {'Std':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for l in labels:
        scores = totals[l]
        avg = sum(scores) / len(scores)
        mn, mx = min(scores), max(scores)
        std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5
        delta = avg - old_avg
        marker = " <<<" if l == "NEW(fix+shift)" else ""
        print(f"  {l:<20} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {mx:7.1f} {std:7.1f}{marker}")

    print(f"\n  Total improvement: OLD {old_avg:.2f} -> NEW {new_avg:.2f} ({new_avg - old_avg:+.2f} pts)")

    # Simulated leaderboard impact (best single round × weight)
    print(f"\n  Simulated best-round scores:")
    for l in ["OLD(bug+noshift)", "NEW(fix+shift)"]:
        round_avgs = []
        idx = 0
        for rid in round_ids:
            n_seeds = len(rounds[rid])
            round_avg = sum(totals[l][idx:idx+n_seeds]) / n_seeds
            round_avgs.append((rid[:8], round_avg))
            idx += n_seeds
        best = max(round_avgs, key=lambda x: x[1])
        print(f"    {l}: best = {best[0]} @ {best[1]:.1f}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
