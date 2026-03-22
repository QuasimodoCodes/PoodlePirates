"""
spatial_features_test.py — Test per-cell spatial features to improve Plains predictions.

Plains cells (code 11) account for 40-60% of prediction loss. The current 19-bucket
context model (code, sett_bin[3], ocean_bin[2]) treats all Plains cells within a bucket
identically. But within a bucket, cells vary a lot — mainly driven by distance to
nearest settlement. Cells 2 tiles away from a settlement behave very differently from
cells 10+ tiles away, even if both have 0 settlement neighbors in r=2.

This script tests whether adding per-cell spatial features (computed from the free
initial_grid) can reduce Plains loss by creating finer-grained buckets for Plains ONLY.
Other terrain types continue to use the standard 19-bucket context.

Spatial features tested:
  1. Distance to nearest settlement (Manhattan, binned: 0-2, 3-5, 6-10, 11+)
  2. Settlement count within radius 5
  3. Distance to nearest ocean (Manhattan, binned)
  4. Settlement density gradient (r=3 vs r=6)
  5. Combinations of the above

python -m scripts.spatial_features_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.001; ALPHA = 0.05; TEMP = 1.10
N_HIST = 50; N_HIST_SURP = 5; SURP_THRESH = 0.30
random.seed(42)

ALL_A = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
AC = {a: set(_covered_cells(*a)) for a in ALL_A}


# ── Utility functions (same as context_features_test) ──────────

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
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < 40 and 0 <= nx < 40 and ig[ny][nx] == tc: c += 1
    return c

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

def ts(d, t):
    if t == 1.0: s = sum(d); return [v / s for v in d]
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
    s = sum(shifted); return [v / s for v in shifted]


# ── Precompute spatial feature maps for an initial_grid ────────

def _find_positions(ig, code):
    """Return set of (y, x) positions where ig[y][x] == code."""
    return {(y, x) for y in range(40) for x in range(40) if ig[y][x] == code}

def _manhattan_dist(y1, x1, y2, x2):
    return abs(y1 - y2) + abs(x1 - x2)

def _min_manhattan_dist(y, x, positions):
    """Minimum Manhattan distance from (y,x) to any cell in positions set."""
    if not positions: return 99
    return min(_manhattan_dist(y, x, py, px) for py, px in positions)

def _count_in_radius(y, x, positions, r):
    """Count cells in positions that are within Manhattan radius r of (y,x)."""
    c = 0
    for py, px in positions:
        if _manhattan_dist(y, x, py, px) <= r:
            c += 1
    return c

def compute_spatial_features(ig):
    """
    Precompute spatial feature maps for all cells.
    Returns dict: (y,x) -> {
        'dist_sett': int,       # Manhattan distance to nearest settlement
        'dist_ocean': int,      # Manhattan distance to nearest ocean
        'sett_r5': int,         # Number of settlements within Manhattan r=5
        'sett_r3': int,         # Number of settlements within Manhattan r=3
        'sett_r6': int,         # Number of settlements within Manhattan r=6
    }
    """
    sett_pos = _find_positions(ig, 1)
    ocean_pos = _find_positions(ig, 10)

    features = {}
    for y in range(40):
        for x in range(40):
            features[(y, x)] = {
                'dist_sett': _min_manhattan_dist(y, x, sett_pos),
                'dist_ocean': _min_manhattan_dist(y, x, ocean_pos),
                'sett_r5': _count_in_radius(y, x, sett_pos, 5),
                'sett_r3': _count_in_radius(y, x, sett_pos, 3),
                'sett_r6': _count_in_radius(y, x, sett_pos, 6),
            }
    return features


# ── Bin helpers ────────────────────────────────────────────────

def bin_dist_sett(d):
    """Bin settlement distance: 0-2, 3-5, 6-10, 11+"""
    if d <= 2: return "sd0"
    if d <= 5: return "sd1"
    if d <= 10: return "sd2"
    return "sd3"

def bin_dist_sett_fine(d):
    """Finer settlement distance: 0-1, 2-3, 4-6, 7-10, 11+"""
    if d <= 1: return "sd0"
    if d <= 3: return "sd1"
    if d <= 6: return "sd2"
    if d <= 10: return "sd3"
    return "sd4"

def bin_dist_ocean(d):
    """Bin ocean distance: 0-1, 2-4, 5-10, 11+"""
    if d <= 1: return "od0"
    if d <= 4: return "od1"
    if d <= 10: return "od2"
    return "od3"

def bin_dist_ocean_coarse(d):
    """Coarser ocean distance: 0-2, 3-8, 9+"""
    if d <= 2: return "od0"
    if d <= 8: return "od1"
    return "od2"

def bin_sett_r5(c):
    """Bin settlement count within r=5: 0, 1-2, 3-5, 6+"""
    if c == 0: return "sr0"
    if c <= 2: return "sr1"
    if c <= 5: return "sr2"
    return "sr3"

def bin_density_gradient(r3, r6):
    """
    Settlement density gradient: compare inner (r=3) vs outer (r=6).
    High inner/outer ratio = settlement core, low = settlement edge/fringe.
    """
    if r6 == 0: return "dg0"  # no settlements nearby
    ratio = r3 / r6
    if ratio >= 0.6: return "dg3"   # dense core
    if ratio >= 0.3: return "dg2"   # inner zone
    if r3 > 0: return "dg1"         # edge: some nearby but mostly far
    return "dg0"                     # fringe: settlements only in 3-6 range


# ── Context functions ──────────────────────────────────────────
#
# All return a context key used for bucket-level averaging.
# For non-Plains cells, all strategies use the standard (code, sett_bin, ocean_bin).
# For Plains cells, strategies add spatial features.

def _base_ctx(ig, y, x):
    """Standard context for non-Plains cells."""
    code = ig[y][x]
    sn = cn(ig, y, x, 1)
    on = cn(ig, y, x, 10)
    sb = "sh" if sn >= 3 else ("sl" if sn >= 1 else "sn")
    ob = "oc" if on >= 1 else "in"
    return (code, sb, ob)


def make_ctx_fn(plains_fn):
    """
    Factory: returns a context function that uses _base_ctx for non-Plains cells
    and calls plains_fn(ig, y, x, features) for Plains cells.
    The returned function has an extra features_cache dict that must be populated
    before use.
    """
    def ctx_fn(ig, y, x, feat_map=None):
        code = ig[y][x]
        if code != 11:
            return _base_ctx(ig, y, x)
        return plains_fn(ig, y, x, feat_map)
    return ctx_fn


# Strategy A: Baseline (current 19-bucket, no spatial features)
def ctx_baseline(ig, y, x, feat_map=None):
    return _base_ctx(ig, y, x)

# Strategy B: Plains split by distance to nearest settlement
def ctx_dist_sett(ig, y, x, feat_map=None):
    f = feat_map[(y, x)]
    base = _base_ctx(ig, y, x)
    return base + (bin_dist_sett(f['dist_sett']),)

# Strategy C: Plains split by distance to nearest settlement (finer bins)
def ctx_dist_sett_fine(ig, y, x, feat_map=None):
    f = feat_map[(y, x)]
    base = _base_ctx(ig, y, x)
    return base + (bin_dist_sett_fine(f['dist_sett']),)

# Strategy D: Plains split by settlement count within r=5
def ctx_sett_r5(ig, y, x, feat_map=None):
    f = feat_map[(y, x)]
    base = _base_ctx(ig, y, x)
    return base + (bin_sett_r5(f['sett_r5']),)

# Strategy E: Plains split by distance to ocean
def ctx_dist_ocean(ig, y, x, feat_map=None):
    f = feat_map[(y, x)]
    base = _base_ctx(ig, y, x)
    return base + (bin_dist_ocean(f['dist_ocean']),)

# Strategy F: Plains split by settlement density gradient
def ctx_gradient(ig, y, x, feat_map=None):
    f = feat_map[(y, x)]
    base = _base_ctx(ig, y, x)
    return base + (bin_density_gradient(f['sett_r3'], f['sett_r6']),)

# Strategy G: Plains split by dist_sett + dist_ocean (two extra dims)
def ctx_sett_ocean_dist(ig, y, x, feat_map=None):
    f = feat_map[(y, x)]
    base = _base_ctx(ig, y, x)
    return base + (bin_dist_sett(f['dist_sett']), bin_dist_ocean_coarse(f['dist_ocean']))

# Strategy H: Plains split by dist_sett + gradient
def ctx_sett_gradient(ig, y, x, feat_map=None):
    f = feat_map[(y, x)]
    base = _base_ctx(ig, y, x)
    return base + (bin_dist_sett(f['dist_sett']), bin_density_gradient(f['sett_r3'], f['sett_r6']))

# Strategy I: Plains split by sett_r5 + dist_ocean (broader spatial context)
def ctx_r5_ocean(ig, y, x, feat_map=None):
    f = feat_map[(y, x)]
    base = _base_ctx(ig, y, x)
    return base + (bin_sett_r5(f['sett_r5']), bin_dist_ocean_coarse(f['dist_ocean']))

# Strategy J: Plains replace sett_bin with dist_sett (pure spatial, no neighbor count)
def ctx_pure_spatial(ig, y, x, feat_map=None):
    f = feat_map[(y, x)]
    code = ig[y][x]
    on = cn(ig, y, x, 10)
    ob = "oc" if on >= 1 else "in"
    return (code, bin_dist_sett(f['dist_sett']), ob)


# ── Model runner ───────────────────────────────────────────────

def run_model(ig, gt, cm_hist, ctx_fn, feat_map):
    """
    Run one seed: simulate observations, build posterior, score against GT.
    ctx_fn(ig, y, x, feat_map) returns the context key.
    """
    ao = get_obs(ig, gt)
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_fn(ig, yx[0], yx[1], feat_map)
            for v in vals: rc[ct].append(v)

    R = compute_global_shift_sqrt(rc, cm_hist)
    shifted = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}
    bl = {}
    for ct in set(list(shifted.keys()) + list(rc.keys())):
        h = shifted.get(ct, [1.0 / N] * N); ol = rc.get(ct, []); nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surp(ol, h)
        nh = N_HIST_SURP if s > SURP_THRESH and nr >= 5 else N_HIST
        rf = [0.0] * N
        for v in ol: rf[v] += 1.0 / nr
        t = nr + nh
        bl[ct] = [(nr * rf[i] + nh * h[i]) / t for i in range(N)]

    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS] * N; d[pc] = 1 - 5 * FS
            else:
                ct = ctx_fn(ig, y, x, feat_map)
                prior = bl.get(ct, shifted.get(ct, [1.0 / N] * N))[:]
                if (y, x) in ao:
                    vals = ao[(y, x)]; oh = [0.0] * N
                    for v in vals: oh[v] += 1.0 / len(vals)
                    d = [(1 - ALPHA) * prior[i] + ALPHA * oh[i] for i in range(N)]
                else:
                    d = prior[:]
                d = ts(d, TEMP)
                d = [max(v, FD) for v in d]
            t = sum(d); row.append([v / t for v in d])
        tensor.append(row)
    return score_t(tensor, gt)


# ── Plains-only score component ───────────────────────────────

def score_plains_only(pred, gt, ig):
    """Compute score contribution from Plains cells only (for diagnostic)."""
    wkl = te = 0.0
    for y in range(40):
        for x in range(40):
            if ig[y][x] == 11:
                e = _entropy(gt[y][x])
                wkl += e * kl(gt[y][x], pred[y][x])
                te += e
    return (wkl, te)


def run_model_with_plains_score(ig, gt, cm_hist, ctx_fn, feat_map):
    """Run model and return (overall_score, plains_wkl, plains_te)."""
    ao = get_obs(ig, gt)
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_fn(ig, yx[0], yx[1], feat_map)
            for v in vals: rc[ct].append(v)

    R = compute_global_shift_sqrt(rc, cm_hist)
    shifted = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}
    bl = {}
    for ct in set(list(shifted.keys()) + list(rc.keys())):
        h = shifted.get(ct, [1.0 / N] * N); ol = rc.get(ct, []); nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surp(ol, h)
        nh = N_HIST_SURP if s > SURP_THRESH and nr >= 5 else N_HIST
        rf = [0.0] * N
        for v in ol: rf[v] += 1.0 / nr
        t = nr + nh
        bl[ct] = [(nr * rf[i] + nh * h[i]) / t for i in range(N)]

    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS] * N; d[pc] = 1 - 5 * FS
            else:
                ct = ctx_fn(ig, y, x, feat_map)
                prior = bl.get(ct, shifted.get(ct, [1.0 / N] * N))[:]
                if (y, x) in ao:
                    vals = ao[(y, x)]; oh = [0.0] * N
                    for v in vals: oh[v] += 1.0 / len(vals)
                    d = [(1 - ALPHA) * prior[i] + ALPHA * oh[i] for i in range(N)]
                else:
                    d = prior[:]
                d = ts(d, TEMP)
                d = [max(v, FD) for v in d]
            t = sum(d); row.append([v / t for v in d])
        tensor.append(row)

    overall = score_t(tensor, gt)
    p_wkl, p_te = score_plains_only(tensor, gt, ig)
    return overall, p_wkl, p_te


# ── Main ───────────────────────────────────────────────────────

def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    round_ids = sorted(rounds.keys())
    print(f"Spatial features test: {len(files)} files, {len(rounds)} rounds")
    print("=" * 140)

    # Precompute spatial features for all grids
    print("Precomputing spatial features for all grids...")
    feat_cache = {}  # filename -> feature map
    for f in files:
        d = all_data[f]; ig = d.get("initial_grid")
        if ig: feat_cache[f] = compute_spatial_features(ig)
    print(f"  Done: {len(feat_cache)} feature maps computed.")

    strategies = [
        ("A.baseline",      ctx_baseline),
        ("B.dist_sett",     ctx_dist_sett),
        ("C.dist_sett_f",   ctx_dist_sett_fine),
        ("D.sett_r5",       ctx_sett_r5),
        ("E.dist_ocean",    ctx_dist_ocean),
        ("F.gradient",      ctx_gradient),
        ("G.sett+ocean",    ctx_sett_ocean_dist),
        ("H.sett+grad",     ctx_sett_gradient),
        ("I.r5+ocean",      ctx_r5_ocean),
        ("J.pure_spatial",  ctx_pure_spatial),
    ]

    labels = [s[0] for s in strategies]
    print(f"\n  {'Round':<10}", end="")
    for l in labels: print(f" {l:>14}", end="")
    print()
    print(f"  {'-' * 10}", end="")
    for _ in labels: print(f" {'-' * 14}", end="")
    print()

    totals = {l: [] for l in labels}
    plains_wkl_totals = {l: 0.0 for l in labels}
    plains_te_totals = {l: 0.0 for l in labels}
    bucket_counts = {l: [] for l in labels}

    for test_rid in round_ids:
        # Build conditional matrix per strategy (leave-one-round-out)
        cms = {}
        for label, ctx_fn in strategies:
            cond_acc = defaultdict(list)
            for f in files:
                if f.split("_seed")[0] == test_rid: continue
                d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
                if not gt or not ig: continue
                fm = feat_cache.get(f)
                for y in range(40):
                    for x in range(40):
                        c = ig[y][x]
                        if c not in STATIC_CODES:
                            ct = ctx_fn(ig, y, x, fm)
                            cond_acc[ct].append(gt[y][x])
            cm = {c: [sum(s[i] for s in ss) / len(ss) for i in range(N)] for c, ss in cond_acc.items()}
            cms[label] = cm
            bucket_counts[label].append(len(cm))

        round_scores = {l: [] for l in labels}
        for fname in rounds[test_rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            fm = feat_cache.get(fname)
            for label, ctx_fn in strategies:
                sc, p_wkl, p_te = run_model_with_plains_score(ig, gt, cms[label], ctx_fn, fm)
                round_scores[label].append(sc)
                plains_wkl_totals[label] += p_wkl
                plains_te_totals[label] += p_te

        short = test_rid[:8]
        print(f"  {short:<10}", end="")
        for l in labels:
            avg = sum(round_scores[l]) / len(round_scores[l])
            totals[l].extend(round_scores[l])
            print(f" {avg:14.1f}", end="")
        print()

    print()
    print("=" * 140)
    baseline_avg = sum(totals[labels[0]]) / len(totals[labels[0]])

    print(f"\n  {'Config':<18} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Std':>7} {'Bkts':>6} {'PlWKL':>8} {'PlScore':>8}")
    print(f"  {'-' * 18} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 8} {'-' * 8}")
    for label, ctx_fn in strategies:
        scores = totals[label]
        avg = sum(scores) / len(scores)
        mn = min(scores)
        std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5
        delta = avg - baseline_avg
        avg_buckets = sum(bucket_counts[label]) / len(bucket_counts[label])
        p_wkl = plains_wkl_totals[label]
        p_te = plains_te_totals[label]
        p_score = max(0, min(100, 100 * math.exp(-3 * (p_wkl / p_te)))) if p_te > 1e-12 else 100
        marker = " <<<" if delta > 0.3 else (" *" if delta > 0.1 else "")
        print(f"  {label:<18} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {std:7.1f} {avg_buckets:>5.0f} {p_wkl:>8.1f} {p_score:>7.1f}{marker}")

    # ── Diagnostic: bucket population analysis ──────────────────
    print(f"\n\n  DIAGNOSTIC: Plains bucket population (last round, first seed)")
    print(f"  {'-' * 90}")
    last_rid = round_ids[-1]
    last_fname = rounds[last_rid][0]
    d = all_data[last_fname]; ig = d.get("initial_grid")
    fm = feat_cache.get(last_fname)
    if ig and fm:
        for label, ctx_fn in strategies:
            # Count plains cells per bucket using all history
            plains_buckets = defaultdict(int)
            for y in range(40):
                for x in range(40):
                    if ig[y][x] == 11:
                        ct = ctx_fn(ig, y, x, fm)
                        plains_buckets[ct] += 1
            total_plains = sum(plains_buckets.values())
            n_buckets = len(plains_buckets)
            sizes = sorted(plains_buckets.values(), reverse=True)
            top5 = sizes[:5]
            bot5 = sizes[-5:] if len(sizes) >= 5 else sizes
            min_sz = min(sizes) if sizes else 0
            max_sz = max(sizes) if sizes else 0
            avg_sz = total_plains / n_buckets if n_buckets > 0 else 0
            print(f"  {label:<18} {n_buckets:>3} plains buckets, "
                  f"cells/bucket: min={min_sz} avg={avg_sz:.0f} max={max_sz}  "
                  f"top5={top5}  bot5={bot5}")

    # ── Diagnostic: spatial feature distributions ──────────────
    print(f"\n\n  DIAGNOSTIC: Spatial feature distributions for Plains cells")
    print(f"  {'-' * 90}")
    dist_sett_hist = defaultdict(int)
    dist_ocean_hist = defaultdict(int)
    sett_r5_hist = defaultdict(int)
    gradient_hist = defaultdict(int)
    plains_count = 0
    for f in files:
        d = all_data[f]; ig = d.get("initial_grid")
        fm = feat_cache.get(f)
        if not ig or not fm: continue
        for y in range(40):
            for x in range(40):
                if ig[y][x] == 11:
                    plains_count += 1
                    fv = fm[(y, x)]
                    dist_sett_hist[bin_dist_sett(fv['dist_sett'])] += 1
                    dist_ocean_hist[bin_dist_ocean(fv['dist_ocean'])] += 1
                    sett_r5_hist[bin_sett_r5(fv['sett_r5'])] += 1
                    gradient_hist[bin_density_gradient(fv['sett_r3'], fv['sett_r6'])] += 1
    print(f"  Total Plains cells across all files: {plains_count}")
    print(f"\n  dist_sett bins:")
    for b in sorted(dist_sett_hist.keys()):
        pct = 100 * dist_sett_hist[b] / plains_count
        print(f"    {b}: {dist_sett_hist[b]:>7} ({pct:5.1f}%)")
    print(f"\n  dist_ocean bins:")
    for b in sorted(dist_ocean_hist.keys()):
        pct = 100 * dist_ocean_hist[b] / plains_count
        print(f"    {b}: {dist_ocean_hist[b]:>7} ({pct:5.1f}%)")
    print(f"\n  sett_r5 bins:")
    for b in sorted(sett_r5_hist.keys()):
        pct = 100 * sett_r5_hist[b] / plains_count
        print(f"    {b}: {sett_r5_hist[b]:>7} ({pct:5.1f}%)")
    print(f"\n  density_gradient bins:")
    for b in sorted(gradient_hist.keys()):
        pct = 100 * gradient_hist[b] / plains_count
        print(f"    {b}: {gradient_hist[b]:>7} ({pct:5.1f}%)")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
