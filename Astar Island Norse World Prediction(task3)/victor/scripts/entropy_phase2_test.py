"""
entropy_phase2_test.py — Test entropy-guided Phase 2 vs fixed spread.

SPREAD (current): Phase 2 uses fixed corners + center anchors.
ENTROPY: Phase 2 greedily picks tiles covering highest-entropy unobserved cells.

python -m scripts.entropy_phase2_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.001; TEMP = 1.10; ALPHA = 0.05
N_HIST = 50; N_HIST_SURP = 20; SURP_THRESH = 0.30
random.seed(42)

ALL_A = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
AC = {a: set(_covered_cells(*a)) for a in ALL_A}


def bfs_dist(ig, tc):
    H, W = 40, 40
    dist = [[99]*W for _ in range(H)]; q = []
    for y in range(H):
        for x in range(W):
            if ig[y][x] == tc: dist[y][x] = 0; q.append((y, x))
    head = 0
    while head < len(q):
        cy, cx = q[head]; head += 1
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy+dy, cx+dx
            if 0 <= ny < H and 0 <= nx < W and dist[ny][nx] > dist[cy][cx]+1:
                dist[ny][nx] = dist[cy][cx]+1; q.append((ny, nx))
    return dist

def bin_od(d):
    if d <= 1: return "od0"
    if d <= 4: return "od1"
    if d <= 10: return "od2"
    return "od3"
def bin_sd(d):
    if d <= 2: return "sd_close"
    if d <= 5: return "sd_mid"
    if d <= 10: return "sd_far"
    return "sd_void"
def kl_div(p, q):
    return sum(pi * math.log(pi / max(qi, 1e-12)) for pi, qi in zip(p, q) if pi > 1e-12)
def score_t(pred, gt):
    wkl = te = 0.0
    for y in range(40):
        for x in range(40):
            e = _entropy(gt[y][x]); wkl += e * kl_div(gt[y][x], pred[y][x]); te += e
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
            ny, nx = y+dy, x+dx
            if 0 <= ny < 40 and 0 <= nx < 40 and ig[ny][nx] == tc: c += 1
    return c
def ts(d, t):
    if t == 1.0: s = sum(d); return [v/s for v in d]
    ld = [math.log(max(p, 1e-12))/t for p in d]; mx = max(ld)
    ed = [math.exp(v - mx) for v in ld]; s = sum(ed)
    return [v/s for v in ed]
def surprise(ol, h):
    nr = len(ol)
    if nr < 3: return 0.0
    rf = [0.0]*N
    for v in ol: rf[v] += 1.0/nr
    kf = sum(rf[i]*math.log(max(rf[i],1e-12)/max(h[i],1e-12)) for i in range(N) if rf[i] > 1e-12)
    kr = sum(h[i]*math.log(max(h[i],1e-12)/max(rf[i],1e-12)) for i in range(N) if h[i] > 1e-12)
    return (kf + kr) / 2
def compute_shift(rc, cm):
    rt = [0.0]*N; no = 0
    for ct, ol in rc.items():
        for cls in ol: rt[cls] += 1; no += 1
    if no == 0: return [1.0]*N
    rf = [c/no for c in rt]
    ht = [0.0]*N; nb = 0
    for ct, d in cm.items():
        for i in range(N): ht[i] += d[i]
        nb += 1
    if nb == 0: return [1.0]*N
    hf = [h/nb for h in ht]
    return [math.sqrt(rf[i]/hf[i]) if hf[i] > 1e-8 else 1.0 for i in range(N)]
def apply_shift(d, R):
    s = [max(d[i]*R[i], 1e-12) for i in range(N)]
    t = sum(s); return [v/t for v in s]

def ctx_fn(ig, y, x, odm, sdm):
    code = ig[y][x]; sn = cn(ig, y, x, 1)
    sb = "sh" if sn >= 3 else ("sl" if sn >= 1 else "sn")
    if code in (4, 11) and odm and sdm:
        return (code, bin_sd(sdm[y][x]), bin_od(odm[y][x]))
    else:
        on = cn(ig, y, x, 10)
        return (code, sb, "oc" if on >= 1 else "in")

def sel_sett(ig, n=5):
    ss = {(y,x) for y in range(40) for x in range(40) if ig[y][x] == 1}
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


def select_entropy_tiles(ig, cm, odm, sdm, phase1_covered, n=5):
    """Greedily pick n tiles covering the most uncertain unobserved cells."""
    cell_entropy = {}
    for y in range(40):
        for x in range(40):
            if (y, x) in phase1_covered:
                continue
            code = ig[y][x]
            if code in STATIC_CODES:
                continue
            ct = ctx_fn(ig, y, x, odm, sdm)
            prior = cm.get(ct, [1.0/N]*N)
            cell_entropy[(y, x)] = _entropy(prior)

    covered = set()
    selected = []
    for _ in range(n):
        best_a = None
        best_score = -1
        for a in ALL_A:
            tile_cells = AC[a]
            score = sum(cell_entropy.get(yx, 0) for yx in tile_cells if yx not in covered)
            if score > best_score:
                best_score = score
                best_a = a
        if best_a is None or best_score <= 0:
            remaining = [a for a in SPREAD_ANCHORS if a not in selected]
            if remaining:
                selected.append(remaining[0])
            continue
        selected.append(best_a)
        covered |= AC[best_a]
    return selected


def build_predictions(ig, gt, cm, odm, sdm, ao):
    """Build prediction tensor given observations."""
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_fn(ig, yx[0], yx[1], odm, sdm)
            for v in vals: rc[ct].append(v)
    R = compute_shift(rc, cm)
    shifted = {ct: apply_shift(d, R) for ct, d in cm.items()}
    bl = {}
    for ct in set(list(shifted.keys()) + list(rc.keys())):
        h = shifted.get(ct, [1.0/N]*N)
        ol = rc.get(ct, [])
        nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surprise(ol, h)
        nh = N_HIST_SURP if (s > SURP_THRESH and nr >= 5) else N_HIST
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
                ct = ctx_fn(ig, y, x, odm, sdm)
                prior = bl.get(ct, shifted.get(ct, [1.0/N]*N))[:]
                if (y, x) in ao:
                    vals = ao[(y, x)]; no = len(vals); oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/no
                    d = [(1-ALPHA)*prior[i] + ALPHA*oh[i] for i in range(N)]
                else: d = prior[:]
                d = ts(d, TEMP)
                d = [max(v, FD) for v in d]
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

    print("Precomputing distance maps...")
    odm_c = {}; sdm_c = {}
    for f in files:
        ig = all_data[f].get("initial_grid")
        if ig:
            odm_c[f] = bfs_dist(ig, 10)
            sdm_c[f] = bfs_dist(ig, 1)
    print(f"  Done: {len(odm_c)} maps.")

    round_ids = sorted(rounds.keys())

    print(f"\nEntropy Phase 2 Test: {len(files)} files, {len(rounds)} rounds")
    print(f"SPREAD: fixed corners+center | ENTROPY: greedy highest-entropy tiles")
    print("=" * 70)
    print(f"  {'Round':<10} {'SPREAD':>8} {'ENTROPY':>8} {'Delta':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    spread_all = []
    entropy_all = []

    for trid in round_ids:
        ca = defaultdict(list)
        for f in files:
            if f.split("_seed")[0] == trid: continue
            d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(f); sdm = sdm_c.get(f)
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in STATIC_CODES:
                        ca[ctx_fn(ig, y, x, odm, sdm)].append(gt[y][x])
        cm = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in ca.items()}

        spread_scores = []
        entropy_scores = []

        for fname in rounds[trid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(fname); sdm = sdm_c.get(fname)

            rng = random.getstate()

            # SPREAD
            random.setstate(rng)
            obs1 = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
            obs2 = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
            ao_spread = {}
            for a in sel_sett(ig, 5):
                for yx in AC[a]: ao_spread[yx] = [obs1[yx]]
            for a in SPREAD_ANCHORS:
                for yx in AC[a]:
                    if yx in ao_spread: ao_spread[yx].append(obs2[yx])
                    else: ao_spread[yx] = [obs2[yx]]
            sc_spread = build_predictions(ig, gt, cm, odm, sdm, ao_spread)
            spread_scores.append(sc_spread)

            # ENTROPY
            random.setstate(rng)
            obs1e = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
            obs2e = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
            ao_entropy = {}
            phase1_tiles = sel_sett(ig, 5)
            phase1_covered = set()
            for a in phase1_tiles:
                for yx in AC[a]:
                    ao_entropy[yx] = [obs1e[yx]]
                    phase1_covered.add(yx)
            phase2_tiles = select_entropy_tiles(ig, cm, odm, sdm, phase1_covered, n=5)
            for a in phase2_tiles:
                for yx in AC[a]:
                    if yx in ao_entropy: ao_entropy[yx].append(obs2e[yx])
                    else: ao_entropy[yx] = [obs2e[yx]]
            sc_entropy = build_predictions(ig, gt, cm, odm, sdm, ao_entropy)
            entropy_scores.append(sc_entropy)

        s_avg = sum(spread_scores) / len(spread_scores)
        e_avg = sum(entropy_scores) / len(entropy_scores)
        delta = e_avg - s_avg
        spread_all.extend(spread_scores)
        entropy_all.extend(entropy_scores)

        note = ""
        if delta > 1: note = " <<<"
        elif delta < -1: note = " !!!"
        short = trid[:8]
        print(f"  {short:<10} {s_avg:8.1f} {e_avg:8.1f} {delta:+8.1f}{note}")

    print(f"\n{'='*70}")
    s_avg = sum(spread_all) / len(spread_all)
    e_avg = sum(entropy_all) / len(entropy_all)
    s_std = (sum((s - s_avg)**2 for s in spread_all) / len(spread_all)) ** 0.5
    e_std = (sum((s - e_avg)**2 for s in entropy_all) / len(entropy_all)) ** 0.5

    print(f"\n  {'Config':<10} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Max':>7} {'Std':>7}")
    print(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    print(f"  {'SPREAD':<10} {s_avg:7.2f} {0:+7.2f} {min(spread_all):7.1f} {max(spread_all):7.1f} {s_std:7.1f}")
    print(f"  {'ENTROPY':<10} {e_avg:7.2f} {e_avg-s_avg:+7.2f} {min(entropy_all):7.1f} {max(entropy_all):7.1f} {e_std:7.1f}")

    wins = sum(1 for s, e in zip(spread_all, entropy_all) if e > s)
    losses = sum(1 for s, e in zip(spread_all, entropy_all) if e < s)
    print(f"\n  Per-seed: {wins} wins, {losses} losses out of {len(spread_all)}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
