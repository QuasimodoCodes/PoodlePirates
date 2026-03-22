"""
r17_comparison.py — Compare OLD vs NEW pipeline on R17 ground truth.
python -m scripts.r17_comparison
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


def run_pipeline(ig, gt, cm_hist, use_bugfix, use_shift):
    ao = get_obs(ig, gt)

    # Collect per-bucket obs
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx(ig, yx[0], yx[1])
            for v in vals: rc[ct].append(v)

    # Global shift
    if use_shift:
        round_total = [0.0] * N; n_obs = 0
        for ct2, ol in rc.items():
            for cls in ol: round_total[cls] += 1; n_obs += 1
        if n_obs > 0:
            round_freq = [c / n_obs for c in round_total]
        else:
            round_freq = [1.0 / N] * N
        hist_total = [0.0] * N; nb = 0
        for ct2, dist in cm_hist.items():
            for i in range(N): hist_total[i] += dist[i]
            nb += 1
        hist_freq = [h / nb for h in hist_total]
        R = [math.sqrt(round_freq[i] / hist_freq[i]) if hist_freq[i] > 1e-8 else 1.0 for i in range(N)]
        base_cm = {}
        for ct2, dist in cm_hist.items():
            sh = [max(dist[i] * R[i], 1e-12) for i in range(N)]
            s = sum(sh); base_cm[ct2] = [v / s for v in sh]
    else:
        base_cm = cm_hist

    # Blend buckets
    bl = {}
    for ct2 in set(list(base_cm.keys()) + list(rc.keys())):
        h = base_cm.get(ct2, [1.0 / N] * N); ol = rc.get(ct2, []); nr = len(ol)
        if nr == 0: bl[ct2] = h[:]; continue
        s = surp(ol, h)
        nh = N_HIST_SURP if s > SURP_THRESH and nr >= 5 else N_HIST
        rf = [0.0] * N
        for v in ol: rf[v] += 1.0 / nr
        t = nr + nh
        bl[ct2] = [(nr * rf[i] + nh * h[i]) / t for i in range(N)]

    # Choose CM for predictions
    pred_cm = bl if use_bugfix else cm_hist

    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS] * N; d[pc] = 1 - 5 * FS
            else:
                ct = ctx(ig, y, x)
                prior = pred_cm.get(ct, base_cm.get(ct, [1.0 / N] * N))[:]
                if (y, x) in ao:
                    vals = ao[(y, x)]; oh = [0.0] * N
                    for v in vals: oh[v] += 1.0 / len(vals)
                    d = [(1 - ALPHA) * prior[i] + ALPHA * oh[i] for i in range(N)]
                else:
                    d = prior[:]
                d = ts(d, TEMP)
                fl = TF.get(code, FD); d = [max(v, fl) for v in d]
            t = sum(d); row.append([v / t for v in d])
        tensor.append(row)
    return score_t(tensor, gt)


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    test_rid = "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9"
    test_files = sorted(f for f in files if f.startswith(test_rid))
    print(f"R17 comparison ({test_rid[:8]}): {len(test_files)} seeds, {len(files)} total history")
    print("=" * 70)

    # Build CM from all OTHER rounds
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
    cm_hist = {c: [sum(s[i] for s in ss) / len(ss) for i in range(N)] for c, ss in cond_acc.items()}

    print(f"\n  {'Seed':<6} {'OLD(bug)':>10} {'NEW(fix+shift)':>16} {'Delta':>8} {'Actual':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*16} {'-'*8} {'-'*8}")

    old_scores, new_scores, actual_scores = [], [], []

    for fname in test_files:
        d = all_data[fname]
        gt = d.get("ground_truth"); ig = d.get("initial_grid")
        if not gt or not ig: continue
        seed_idx = int(fname.split("seed")[1].split("_")[0])
        actual = d.get("score", 0)
        actual_scores.append(actual)

        old_sc = run_pipeline(ig, gt, cm_hist, use_bugfix=False, use_shift=False)
        new_sc = run_pipeline(ig, gt, cm_hist, use_bugfix=True, use_shift=True)
        old_scores.append(old_sc)
        new_scores.append(new_sc)

        print(f"  {seed_idx:<6} {old_sc:10.1f} {new_sc:16.1f} {new_sc - old_sc:+8.1f} {actual:8.1f}")

    old_avg = sum(old_scores) / len(old_scores)
    new_avg = sum(new_scores) / len(new_scores)
    act_avg = sum(actual_scores) / len(actual_scores)
    print(f"\n  {'AVG':<6} {old_avg:10.1f} {new_avg:16.1f} {new_avg - old_avg:+8.1f} {act_avg:8.1f}")
    print(f"\n  Actual submitted R17 avg:  {act_avg:.1f}")
    print(f"  NEW pipeline simulated:    {new_avg:.1f} ({new_avg - act_avg:+.1f} vs actual)")


if __name__ == "__main__":
    main()
