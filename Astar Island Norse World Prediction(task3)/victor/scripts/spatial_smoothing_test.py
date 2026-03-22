"""
spatial_smoothing_test.py — Test spatial smoothing of observation residuals.

Current problem: ~5k dynamic cells per seed get zero observation signal.
Fix: Propagate observation residuals to nearby unobserved cells via Gaussian kernel.

Strategies:
  A. Baseline (current: no spatial smoothing)
  B. Gaussian kernel sigma=2, lambda=0.3
  C. Gaussian kernel sigma=3, lambda=0.3
  D. Gaussian kernel sigma=4, lambda=0.3
  E. Gaussian kernel sigma=3, lambda=0.5
  F. Gaussian kernel sigma=3, lambda=0.1
  G. Only smooth unobserved cells, sigma=3, lambda=0.5
  H. Distance-weighted by same terrain only, sigma=3, lambda=0.3

python -m scripts.spatial_smoothing_test
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

def calibrate_cm(ao, ig, cm_hist):
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
    return bl


def build_base_tensor(ig, ao, bl):
    """Build tensor WITHOUT spatial smoothing (our current new pipeline)."""
    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS]*N; d[pc] = 1-5*FS
            else:
                ct = ctx(ig, y, x)
                prior = bl.get(ct, [1.0/N]*N)[:]
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


def spatial_smooth(tensor, ig, ao, sigma, lam, unobs_only=False, same_terrain=False):
    """Apply Gaussian kernel smoothing of observation residuals.

    For each cell, compute weighted average of residuals from nearby observed cells.
    residual = obs_empirical - prediction_at_obs_cell
    """
    # Precompute observed cell residuals
    obs_residuals = {}  # (y,x) -> residual vector
    for (y, x), vals in ao.items():
        code = ig[y][x]
        if code in STATIC_CODES:
            continue
        oh = [0.0] * N
        for v in vals: oh[v] += 1.0 / len(vals)
        pred = tensor[y][x]
        residual = [oh[i] - pred[i] for i in range(N)]
        obs_residuals[(y, x)] = residual

    # Precompute Gaussian kernel weights for efficiency
    r = int(3 * sigma)  # cutoff at 3 sigma
    kernel = {}
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            d2 = dy*dy + dx*dx
            kernel[(dy, dx)] = math.exp(-d2 / (2 * sigma * sigma))

    smoothed = [row[:] for row in tensor]  # deep copy rows only
    smoothed = [[cell[:] for cell in row] for row in tensor]  # full deep copy

    for y in range(40):
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                continue
            if unobs_only and (y, x) in ao:
                continue  # skip observed cells

            # Accumulate weighted residuals from nearby observed cells
            w_total = 0.0
            r_accum = [0.0] * N
            for (dy, dx), w in kernel.items():
                ny, nx = y + dy, x + dx
                if (ny, nx) not in obs_residuals:
                    continue
                if same_terrain and ig[ny][nx] != code:
                    continue
                w_total += w
                res = obs_residuals[(ny, nx)]
                for i in range(N):
                    r_accum[i] += w * res[i]

            if w_total < 1e-12:
                continue

            # Apply smoothed residual
            pred = tensor[y][x]
            new_pred = [pred[i] + lam * r_accum[i] / w_total for i in range(N)]
            # Ensure non-negative and renormalize
            new_pred = [max(v, 1e-12) for v in new_pred]
            # Apply floors
            fl = TF.get(code, FD)
            new_pred = [max(v, fl) for v in new_pred]
            s = sum(new_pred)
            smoothed[y][x] = [v / s for v in new_pred]

    return smoothed


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    round_ids = sorted(rounds.keys())
    print(f"Spatial smoothing test: {len(files)} files, {len(rounds)} rounds")
    print("=" * 100)

    strategies = [
        ("A.baseline",    None, None, False, False),
        ("B.s2_l0.3",     2.0,  0.3,  False, False),
        ("C.s3_l0.3",     3.0,  0.3,  False, False),
        ("D.s4_l0.3",     4.0,  0.3,  False, False),
        ("E.s3_l0.5",     3.0,  0.5,  False, False),
        ("F.s3_l0.1",     3.0,  0.1,  False, False),
        ("G.unobs_s3",    3.0,  0.5,  True,  False),
        ("H.same_s3",     3.0,  0.3,  False, True),
    ]

    labels = [s[0] for s in strategies]
    print(f"\n  {'Round':<10}", end="")
    for l in labels: print(f" {l:>11}", end="")
    print()
    print(f"  {'-'*10}", end="")
    for _ in labels: print(f" {'-'*11}", end="")
    print()

    totals = {l: [] for l in labels}

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

        round_scores = {l: [] for l in labels}

        for fname in rounds[test_rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue

            ao = get_obs(ig, gt)
            bl = calibrate_cm(ao, ig, cm_hist)
            base_tensor = build_base_tensor(ig, ao, bl)

            for label, sigma, lam, unobs_only, same_terrain in strategies:
                if sigma is None:
                    sc = score_t(base_tensor, gt)
                else:
                    smoothed = spatial_smooth(base_tensor, ig, ao, sigma, lam, unobs_only, same_terrain)
                    sc = score_t(smoothed, gt)
                round_scores[label].append(sc)

        short = test_rid[:8]
        print(f"  {short:<10}", end="")
        for l in labels:
            avg = sum(round_scores[l]) / len(round_scores[l])
            totals[l].extend(round_scores[l])
            print(f" {avg:11.1f}", end="")
        print()

    # Summary
    print()
    print("=" * 100)
    baseline_avg = sum(totals["A.baseline"]) / len(totals["A.baseline"])

    print(f"\n  {'Strategy':<16} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Std':>7}")
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for label, sigma, lam, unobs_only, same_terrain in strategies:
        scores = totals[label]
        avg = sum(scores) / len(scores)
        mn = min(scores)
        std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5
        delta = avg - baseline_avg
        desc = "baseline" if sigma is None else f"sigma={sigma} lam={lam}"
        if unobs_only: desc += " unobs"
        if same_terrain: desc += " same"
        marker = " <<<" if delta > 0.3 else (" *" if delta > 0.1 else "")
        print(f"  {label:<16} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {std:7.1f}{marker}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
