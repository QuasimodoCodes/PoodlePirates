"""
query_strategy_test2.py — Test smarter query placement for better calibration.

Current: 5 settlement tiles + 5 spread tiles per seed (25+25)
Problem: extreme rounds shift distributions but 50 queries don't give enough
         samples per context bucket to detect/calibrate the shift.

Strategies to test:
A) Current: 5 settlement + 5 spread (baseline)
B) All spread: 10 spread tiles per seed (max spatial diversity, more buckets sampled)
C) Context-diverse: greedily pick tiles that maximize distinct context buckets observed
D) 3 settlement + 7 spread (more calibration tiles)
E) All 10 tiles targeting max dynamic cells (no static waste)
F) 5 settlement + 5 context-diverse (instead of fixed spread)

python -m scripts.query_strategy_test2
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

# Additional spread anchors for strategy B (10 tiles)
SPREAD_10 = [
    (0, 0), (25, 0), (0, 25), (25, 25), (12, 12),
    (12, 0), (0, 12), (25, 12), (12, 25), (6, 6),
]

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
    while len(sel) < n:
        sel.append(SPREAD_ANCHORS[len(sel) % len(SPREAD_ANCHORS)])
    return sel[:n]


def sel_ctx_diverse(ig, n=5, exclude=None):
    """Greedily select tiles that maximize distinct context buckets observed."""
    if exclude is None:
        exclude = set()
    # Precompute context for each dynamic cell
    cell_ctx = {}
    for y in range(40):
        for x in range(40):
            if ig[y][x] not in STATIC_CODES:
                cell_ctx[(y, x)] = ctx(ig, y, x)

    covered_ctxs = set()
    for yx in exclude:
        if yx in cell_ctx:
            covered_ctxs.add(cell_ctx[yx])

    selected = []
    for _ in range(n):
        best, best_score = None, -1
        for a in ALL_A:
            new_ctxs = set()
            n_dynamic = 0
            for yx in AC[a]:
                if yx in cell_ctx:
                    n_dynamic += 1
                    c = cell_ctx[yx]
                    if c not in covered_ctxs:
                        new_ctxs.add(c)
            # Score: prioritize new context buckets, break ties by dynamic cell count
            score = len(new_ctxs) * 10000 + n_dynamic
            if score > best_score:
                best_score, best = score, a
        if best is None:
            break
        selected.append(best)
        for yx in AC[best]:
            if yx in cell_ctx:
                covered_ctxs.add(cell_ctx[yx])
    return selected[:n]


def sel_max_dynamic(ig, n=10):
    """Greedily select tiles covering most dynamic cells."""
    covered = set()
    selected = []
    for _ in range(n):
        best, best_count = None, -1
        for a in ALL_A:
            count = sum(1 for yx in AC[a] if ig[yx[0]][yx[1]] not in STATIC_CODES and yx not in covered)
            if count > best_count:
                best_count, best = count, a
        if best is None or best_count <= 0:
            break
        selected.append(best)
        covered |= AC[best]
    return selected[:n]


def build_prediction(ig, gt, cm, tile_anchors_list):
    """Build prediction using given tile anchors (list of lists, one per observation draw)."""
    # Each entry in tile_anchors_list is a list of anchors for one observation pass
    ao = {}
    for draw_idx, anchors in enumerate(tile_anchors_list):
        obs = {(y, x): so(gt[y][x]) for y in range(40) for x in range(40)}
        for a in anchors:
            for yx in AC[a]:
                if yx in ao:
                    ao[yx].append(obs[yx])
                else:
                    ao[yx] = [obs[yx]]

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
        nh = 5 if s > .3 and nr >= 5 else 50
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
    print(f"Query strategy test 2: {len(files)} files, {n_rounds} rounds")
    print("=" * 70)

    strategies = ["A.curr(5s+5sp)", "B.10spread", "C.5s+5ctx", "D.3s+7sp", "E.10dyn", "F.10ctx"]

    print(f"\n  {'Round':<12}", end="")
    for s in strategies: print(f" {s:>14}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in strategies: print(f" {'-'*14}", end="")
    print()

    totals = {s: [] for s in strategies}

    for test_rid in round_ids:
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
        cm = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in cond_acc.items()}

        round_scores = {s: [] for s in strategies}

        for fname in rounds[test_rid]:
            d = all_data[fname]; gt, ig = d.get("ground_truth"), d.get("initial_grid")
            if not gt or not ig: continue

            # Precompute tile selections per strategy
            sett5 = sel_sett(ig, 5)
            sett3 = sel_sett(ig, 3)
            ctx5_after_sett = sel_ctx_diverse(ig, 5, exclude=set().union(*(AC[a] for a in sett5)))
            ctx10 = sel_ctx_diverse(ig, 10)
            dyn10 = sel_max_dynamic(ig, 10)
            spread7 = SPREAD_10[:7]

            strat_tiles = {
                "A.curr(5s+5sp)": [sett5, SPREAD_ANCHORS],
                "B.10spread":     [SPREAD_10],
                "C.5s+5ctx":      [sett5, ctx5_after_sett],
                "D.3s+7sp":       [sett3, spread7],
                "E.10dyn":        [dyn10],
                "F.10ctx":        [ctx10],
            }

            for strat_name, tile_lists in strat_tiles.items():
                tensor = build_prediction(ig, gt, cm, tile_lists)
                sc = score_t(tensor, gt)
                round_scores[strat_name].append(sc)

        short = test_rid[:8]
        print(f"  {short:<12}", end="")
        for s in strategies:
            avg = sum(round_scores[s]) / len(round_scores[s])
            totals[s].extend(round_scores[s])
            print(f" {avg:14.1f}", end="")
        print()

    print()
    print("=" * 90)
    baseline = sum(totals[strategies[0]]) / len(totals[strategies[0]])
    print(f"\n  {'Strategy':<20} {'Avg':>7} {'Δ':>7} {'Min':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7}")
    for s in strategies:
        avg = sum(totals[s]) / len(totals[s])
        mn = min(totals[s])
        delta = avg - baseline
        print(f"  {s:<20} {avg:7.2f} {delta:+7.2f} {mn:7.1f}")


if __name__ == "__main__":
    main()
