"""
dirichlet_test.py — Test Dirichlet-Multinomial with raw counts vs fixed N_HIST.

Current: N_HIST=50 for all buckets (same regularization regardless of bucket size).
Dirichlet: Use actual historical counts as prior. Big buckets (29k samples) get strong
prior, small buckets (12 samples) get weak prior that adapts easily.

Also test: data-driven floors from historical minimums.

python -m scripts.dirichlet_test
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


def build_history(all_data, files, exclude_rid, ctx_fn):
    """Build both normalized CM and raw count CM from history."""
    # Raw counts per bucket per class
    raw_counts = defaultdict(lambda: [0.0] * N)  # ctx -> [count_per_class]
    cell_dists = defaultdict(list)  # ctx -> [list of gt distributions]

    for f in files:
        if f.split("_seed")[0] == exclude_rid:
            continue
        d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
        if not gt or not ig: continue
        for y in range(40):
            for x in range(40):
                c = ig[y][x]
                if c in STATIC_CODES: continue
                ct = ctx_fn(ig, y, x)
                dist = gt[y][x]
                cell_dists[ct].append(dist)
                for i in range(N):
                    raw_counts[ct][i] += dist[i]

    # Normalized (mean) CM
    cm_norm = {}
    for ct, dists in cell_dists.items():
        n = len(dists)
        cm_norm[ct] = [sum(d[i] for d in dists) / n for i in range(N)]

    # Compute data-driven floors: 0.5 * min per-round average per terrain code per class
    per_round_avgs = defaultdict(lambda: defaultdict(list))  # code -> class -> [round_avgs]
    round_ids_seen = set()
    for f in files:
        rid = f.split("_seed")[0]
        if rid == exclude_rid: continue
        round_ids_seen.add(rid)

    for rid in round_ids_seen:
        rid_counts = defaultdict(lambda: [0.0] * N)
        rid_n = defaultdict(int)
        for f in files:
            if f.split("_seed")[0] != rid: continue
            d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c in STATIC_CODES: continue
                    for i in range(N):
                        rid_counts[c][i] += gt[y][x][i]
                    rid_n[c] += 1
        for c in rid_counts:
            if rid_n[c] > 0:
                for i in range(N):
                    per_round_avgs[c][i].append(rid_counts[c][i] / rid_n[c])

    data_floors = {}
    for c in per_round_avgs:
        data_floors[c] = [0.5 * min(avgs) if avgs else FD for i, avgs in sorted(per_round_avgs[c].items())]

    return cm_norm, dict(raw_counts), dict({ct: len(dists) for ct, dists in cell_dists.items()}), data_floors


def surp(ol, h):
    nr = len(ol)
    if nr < 3: return 0.0
    rf = [0.0] * N
    for v in ol: rf[v] += 1.0 / nr
    kf = sum(rf[i] * math.log(max(rf[i], 1e-12) / max(h[i], 1e-12)) for i in range(N) if rf[i] > 1e-12)
    kr = sum(h[i] * math.log(max(h[i], 1e-12) / max(rf[i], 1e-12)) for i in range(N) if h[i] > 1e-12)
    return (kf + kr) / 2


def run_strategy(ig, gt, cm_hist, raw_counts, bucket_sizes, data_floors, strategy):
    ao = get_obs(ig, gt)

    # Collect round observations per bucket
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx(ig, yx[0], yx[1])
            for v in vals: rc[ct].append(v)

    # Global shift
    R = compute_global_shift_sqrt(rc, cm_hist)
    shifted = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}

    if strategy == "A.baseline":
        # Current: fixed N_HIST=50, shifted CM
        bl = {}
        for ct in set(list(shifted.keys()) + list(rc.keys())):
            h = shifted.get(ct, [1.0/N]*N); ol = rc.get(ct, []); nr = len(ol)
            if nr == 0: bl[ct] = h[:]; continue
            s = surp(ol, h)
            nh = 5 if s > 0.3 and nr >= 5 else 50
            rf = [0.0]*N
            for v in ol: rf[v] += 1.0/nr
            t = nr + nh
            bl[ct] = [(nr*rf[i] + nh*h[i])/t for i in range(N)]
        floors = TF

    elif strategy.startswith("B.dirichlet") or strategy.startswith("C.dir") or strategy.startswith("D.dir"):
        # Dirichlet: use raw counts as prior (scaled)
        if strategy == "B.dirichlet":
            scale = 1.0  # raw counts as-is
        elif strategy == "C.dir_s0.1":
            scale = 0.1  # scale down counts (weaker prior)
        elif strategy == "D.dir_s0.01":
            scale = 0.01

        bl = {}
        for ct in set(list(shifted.keys()) + list(rc.keys())):
            # Dirichlet prior from scaled raw counts (with shift applied)
            if ct in raw_counts:
                # Apply global shift to raw counts
                raw = raw_counts[ct]
                total_raw = sum(raw)
                if total_raw > 0:
                    norm = [r / total_raw for r in raw]
                    shifted_norm = apply_shift(norm, R)
                    alpha_prior = [shifted_norm[i] * total_raw * scale for i in range(N)]
                else:
                    alpha_prior = [scale] * N
            else:
                alpha_prior = [scale] * N

            ol = rc.get(ct, [])
            if not ol:
                s = sum(alpha_prior)
                bl[ct] = [a / s for a in alpha_prior]
                continue

            # Add round observation counts
            alpha_post = alpha_prior[:]
            for v in ol:
                alpha_post[v] += 1.0

            s = sum(alpha_post)
            bl[ct] = [a / s for a in alpha_post]
        floors = TF

    elif strategy == "E.dir_floors":
        # Dirichlet + data-driven floors
        scale = 0.1
        bl = {}
        for ct in set(list(shifted.keys()) + list(rc.keys())):
            if ct in raw_counts:
                raw = raw_counts[ct]
                total_raw = sum(raw)
                if total_raw > 0:
                    norm = [r / total_raw for r in raw]
                    shifted_norm = apply_shift(norm, R)
                    alpha_prior = [shifted_norm[i] * total_raw * scale for i in range(N)]
                else:
                    alpha_prior = [scale] * N
            else:
                alpha_prior = [scale] * N
            ol = rc.get(ct, [])
            if not ol:
                s = sum(alpha_prior)
                bl[ct] = [a / s for a in alpha_prior]
                continue
            alpha_post = alpha_prior[:]
            for v in ol: alpha_post[v] += 1.0
            s = sum(alpha_post)
            bl[ct] = [a / s for a in alpha_post]
        floors = {}
        for code in [0, 1, 2, 3, 4, 11]:
            if code in data_floors:
                floors[code] = max(min(data_floors[code]), 0.001)
            else:
                floors[code] = FD

    elif strategy == "F.floors_only":
        # Just data-driven floors, no Dirichlet
        bl = {}
        for ct in set(list(shifted.keys()) + list(rc.keys())):
            h = shifted.get(ct, [1.0/N]*N); ol = rc.get(ct, []); nr = len(ol)
            if nr == 0: bl[ct] = h[:]; continue
            s = surp(ol, h)
            nh = 5 if s > 0.3 and nr >= 5 else 50
            rf = [0.0]*N
            for v in ol: rf[v] += 1.0/nr
            t = nr + nh
            bl[ct] = [(nr*rf[i] + nh*h[i])/t for i in range(N)]
        floors = {}
        for code in [0, 1, 2, 3, 4, 11]:
            if code in data_floors:
                floors[code] = max(min(data_floors[code]), 0.001)
            else:
                floors[code] = FD

    else:
        raise ValueError(strategy)

    # Build tensor
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
                    vals = ao[(y,x)]; oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/len(vals)
                    d = [(1-ALPHA)*prior[i] + ALPHA*oh[i] for i in range(N)]
                else:
                    d = prior[:]
                d = ts(d, TEMP)
                fl_val = floors.get(code, FD) if isinstance(floors, dict) else TF.get(code, FD)
                d = [max(v, fl_val) for v in d]
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
    print(f"Dirichlet test: {len(files)} files, {len(rounds)} rounds")
    print("=" * 100)

    strategies = [
        "A.baseline",
        "B.dirichlet",
        "C.dir_s0.1",
        "D.dir_s0.01",
        "E.dir_floors",
        "F.floors_only",
    ]

    print(f"\n  {'Round':<10}", end="")
    for s in strategies: print(f" {s:>12}", end="")
    print()
    print(f"  {'-'*10}", end="")
    for _ in strategies: print(f" {'-'*12}", end="")
    print()

    totals = {s: [] for s in strategies}

    for test_rid in round_ids:
        cm_hist, raw_counts, bucket_sizes, data_floors = build_history(
            all_data, files, test_rid, ctx)

        round_scores = {s: [] for s in strategies}

        for fname in rounds[test_rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue

            for strat in strategies:
                sc = run_strategy(ig, gt, cm_hist, raw_counts, bucket_sizes, data_floors, strat)
                round_scores[strat].append(sc)

        short = test_rid[:8]
        print(f"  {short:<10}", end="")
        for s in strategies:
            avg = sum(round_scores[s]) / len(round_scores[s])
            totals[s].extend(round_scores[s])
            print(f" {avg:12.1f}", end="")
        print()

    # Summary
    print()
    print("=" * 100)
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

    # Show bucket size distribution
    print(f"\n  Bucket sizes (last round):")
    for ct in sorted(bucket_sizes.keys(), key=lambda c: -bucket_sizes[c])[:8]:
        rc = raw_counts[ct]
        total = sum(rc)
        equiv_nhist = total  # effective N_HIST in Dirichlet
        print(f"    {str(ct):<40} cells={bucket_sizes[ct]:>6}  raw_total={total:>8.0f}  equiv_nhist={equiv_nhist:>8.0f}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
