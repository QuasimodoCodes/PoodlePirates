"""
global_shift_test.py — Test global class-frequency shift multiplier for extreme rounds.

Idea (from Gemini review):
  When a round is "extreme" (e.g., Settlements 2x more common), even buckets
  with few observations should reflect that global shift. Currently, small buckets
  are regularized toward the historical average (N_HIST=50 dominates).

  Fix: Compute global class frequency ratio R(c) = round_freq(c) / hist_freq(c)
  for each class c from ALL round observations. Multiply historical priors by R(c)
  before blending with bucket-level observations.

Also tests the bug fix: passing calibrated conditional matrix to predictions.

Strategies:
  A. Baseline (current: calibrate bucket-level only, no global shift)
  B. Global shift: multiply hist prior by R(c) before bucket blending
  C. Global shift + lower N_HIST (25 instead of 50) — more trust in shifted prior
  D. Global shift + adaptive N_HIST (surprise-based, current logic)
  E. Sqrt-damped shift: R(c)^0.5 — gentler shift for noisy estimates
  F. Bug fix only: use calibrated CM (simulates the main.py fix)

python -m scripts.global_shift_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.005; ALPHA = 0.05; TEMP = 1.10
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
    """Simulate 2-phase observations: 5 settlement + 5 spread tiles."""
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


def compute_global_shift(ao, ig, hist_cm):
    """Compute global class-frequency ratio from round observations vs history.

    Returns R[i] = round_freq[i] / hist_freq[i] for each class i.
    Only uses dynamic cells.
    """
    # Round frequency: count observed classes across all dynamic observed cells
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

    # Historical frequency: average across all buckets weighted by bucket size
    hist_total = [0.0] * N
    hist_count = 0
    for ct, dist in hist_cm.items():
        for i in range(N):
            hist_total[i] += dist[i]
        hist_count += 1

    if hist_count == 0:
        return [1.0] * N

    hist_freq = [h / hist_count for h in hist_total]

    # Ratio R[i] = round_freq[i] / hist_freq[i]
    R = [1.0] * N
    for i in range(N):
        if hist_freq[i] > 1e-8:
            R[i] = round_freq[i] / hist_freq[i]
        else:
            R[i] = 1.0

    return R


def apply_shift(dist, R, power=1.0):
    """Multiply distribution by R^power and renormalize."""
    shifted = [max(dist[i] * (R[i] ** power), 1e-12) for i in range(N)]
    s = sum(shifted)
    return [v / s for v in shifted]


def blend_buckets(ao, ig, cm, n_hist=50, n_hist_surp=5):
    """Standard bucket-level blending (current calibration logic)."""
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


def build_tensor(ig, ao, bl, cm):
    """Build prediction tensor from blended buckets."""
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
                    d = [(1-ALPHA)*prior[i] + ALPHA*oh[i] for i in range(N)]
                else:
                    d = prior[:]
                d = ts(d, TEMP)
                fl = TF.get(code, FD); d = [max(v, fl) for v in d]
            t = sum(d); row.append([v/t for v in d])
        tensor.append(row)
    return tensor


def run_strategy(ig, gt, cm_hist, strategy):
    """Run a single strategy and return score.

    Strategies:
      'baseline':    current logic (no global shift)
      'shift':       global shift on hist prior before bucket blending
      'shift_lo':    global shift + N_HIST=25
      'shift_adapt': global shift + adaptive N_HIST (surprise)
      'shift_sqrt':  sqrt-damped shift (R^0.5)
      'bugfix':      simulate bug fix - calibrate CM then use it
    """
    ao = get_obs(ig, gt)

    if strategy == 'baseline':
        bl = blend_buckets(ao, ig, cm_hist)
        return score_t(build_tensor(ig, ao, bl, cm_hist), gt)

    elif strategy == 'shift':
        R = compute_global_shift(ao, ig, cm_hist)
        shifted_cm = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}
        bl = blend_buckets(ao, ig, shifted_cm)
        return score_t(build_tensor(ig, ao, bl, shifted_cm), gt)

    elif strategy == 'shift_lo':
        R = compute_global_shift(ao, ig, cm_hist)
        shifted_cm = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}
        bl = blend_buckets(ao, ig, shifted_cm, n_hist=25)
        return score_t(build_tensor(ig, ao, bl, shifted_cm), gt)

    elif strategy == 'shift_adapt':
        R = compute_global_shift(ao, ig, cm_hist)
        shifted_cm = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}
        bl = blend_buckets(ao, ig, shifted_cm, n_hist=50, n_hist_surp=5)
        return score_t(build_tensor(ig, ao, bl, shifted_cm), gt)

    elif strategy == 'shift_sqrt':
        R = compute_global_shift(ao, ig, cm_hist)
        shifted_cm = {ct: apply_shift(dist, R, power=0.5) for ct, dist in cm_hist.items()}
        bl = blend_buckets(ao, ig, shifted_cm)
        return score_t(build_tensor(ig, ao, bl, shifted_cm), gt)

    elif strategy == 'bugfix':
        # Simulate: calibrate CM (blend buckets), then use calibrated CM for predictions
        bl = blend_buckets(ao, ig, cm_hist)
        # bl IS the calibrated CM — use it directly (this is what the bug fix does)
        return score_t(build_tensor(ig, ao, bl, bl), gt)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


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
    print(f"Global shift test: {len(files)} files, {n_rounds} rounds")
    print("=" * 90)

    strategies = [
        ("A.baseline",    "baseline"),
        ("B.shift",       "shift"),
        ("C.shift+lo25",  "shift_lo"),
        ("D.shift+adapt", "shift_adapt"),
        ("E.shift_sqrt",  "shift_sqrt"),
        ("F.bugfix_only", "bugfix"),
    ]

    print(f"\n  {'Round':<12}", end="")
    for name, _ in strategies: print(f" {name:>14}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in strategies: print(f" {'-'*14}", end="")
    print()

    totals = {name: [] for name, _ in strategies}

    for test_rid in round_ids:
        # Build conditional matrix from all OTHER rounds
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

        round_scores = {name: [] for name, _ in strategies}

        for fname in rounds[test_rid]:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue

            for name, strat_key in strategies:
                sc = run_strategy(ig, gt, cm_hist, strat_key)
                round_scores[name].append(sc)

        # Detect if this was an extreme round
        # (check if any seed had high settlement surprise)
        is_extreme = False
        for fname in rounds[test_rid]:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue
            # Quick check: count settlement class in GT vs typical ~6%
            sett_count = 0; dyn_count = 0
            for y in range(40):
                for x in range(40):
                    if ig[y][x] not in STATIC_CODES:
                        dyn_count += 1
                        # class 1 = settlement
                        if gt[y][x][1] > 0.3:
                            sett_count += 1
            if dyn_count > 0 and sett_count / dyn_count > 0.15:
                is_extreme = True
                break

        tag = " *EXT*" if is_extreme else ""
        short = test_rid[:8]
        print(f"  {short:<12}", end="")
        for name, _ in strategies:
            avg = sum(round_scores[name]) / len(round_scores[name])
            totals[name].extend(round_scores[name])
            print(f" {avg:14.1f}", end="")
        print(tag)

    # Summary
    print()
    print("=" * 90)
    baseline_scores = totals[strategies[0][0]]
    baseline_avg = sum(baseline_scores) / len(baseline_scores)

    print(f"\n  {'Strategy':<20} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Std':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, _ in strategies:
        scores = totals[name]
        avg = sum(scores) / len(scores)
        mn = min(scores)
        std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5
        delta = avg - baseline_avg
        marker = " <<<" if delta > 0.3 else (" *" if delta > 0.1 else "")
        print(f"  {name:<20} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {std:7.1f}{marker}")

    # Per-round delta breakdown for shift vs baseline
    print(f"\n  Per-round delta (shift vs baseline):")
    for test_rid in round_ids:
        bl_scores = [s for i, s in enumerate(baseline_scores)
                     if i // 5 == round_ids.index(test_rid)]  # rough grouping

    print("\n  *EXT* = detected extreme round (>15% settlement class in GT)")
    print("  Done.")


if __name__ == "__main__":
    main()
