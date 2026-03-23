"""
r22_analysis.py — Analyze Round 22 and compare NHS values.

python -m scripts.r22_analysis
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.001; ALPHA = 0.05; TEMP = 1.10

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
def sample_outcome(d):
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


def run_model(ig, gt, cm, odm, sdm, NHS, ST):
    random_state_before = random.getstate()
    obs1 = {(y,x): sample_outcome(gt[y][x]) for y in range(40) for x in range(40)}
    obs2 = {(y,x): sample_outcome(gt[y][x]) for y in range(40) for x in range(40)}
    ao = {}
    for a in sel_sett(ig, 5):
        for yx in AC[a]: ao[yx] = [obs1[yx]]
    for a in SPREAD_ANCHORS:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx] = [obs2[yx]]

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
        nh = NHS if (s > ST and nr >= 5) else 50
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
                else:
                    d = prior[:]
                d = ts(d, TEMP)
                d = [max(v, FD) for v in d]
            t = sum(d); row.append([v/t for v in d])
        tensor.append(row)
    return score_t(tensor, gt)


def main():
    hdir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(hdir) if f.endswith("_analysis.json"))
    r22_id = "a8be24e1-bd48-49bb-aa46-c5593da79f6f"
    train_files = [f for f in files if not f.startswith(r22_id)]
    test_files = sorted([f for f in files if f.startswith(r22_id)])

    all_data = {}
    for f in files:
        with open(os.path.join(hdir, f)) as fh:
            all_data[f] = json.load(fh)

    odm_c = {}; sdm_c = {}
    for f in files:
        ig = all_data[f].get("initial_grid")
        if ig:
            odm_c[f] = bfs_dist(ig, 10)
            sdm_c[f] = bfs_dist(ig, 1)

    # Build CM from training data
    ca = defaultdict(list)
    for f in train_files:
        d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
        if not gt or not ig: continue
        odm = odm_c.get(f); sdm = sdm_c.get(f)
        for y in range(40):
            for x in range(40):
                c = ig[y][x]
                if c not in STATIC_CODES:
                    ca[ctx_fn(ig, y, x, odm, sdm)].append(gt[y][x])
    cm = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in ca.items()}

    # Actual submitted scores
    actual = [all_data[f]["score"] for f in test_files]
    print(f"Round 22 ACTUAL:  avg={sum(actual)/len(actual):.1f}  seeds={[round(s,1) for s in actual]}")
    print()

    # Test different NHS values
    configs = [
        ("NHS=5  (submitted)", 5, 0.30),
        ("NHS=10",            10, 0.30),
        ("NHS=15",            15, 0.30),
        ("NHS=20 (deployed)", 20, 0.30),
        ("NHS=30",            30, 0.30),
        ("NHS=50 (=N_HIST)",  50, 0.30),
    ]

    print(f"  {'Config':<22} {'Avg':>6}  Per-seed scores")
    print(f"  {'-'*22} {'-'*6}  {'-'*40}")

    for label, NHS, ST in configs:
        seed_scores = []
        for fname in test_files:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_c.get(fname); sdm = sdm_c.get(fname)
            random.seed(42 + all_data[fname]["seed_index"])
            sc = run_model(ig, gt, cm, odm, sdm, NHS, ST)
            seed_scores.append(sc)
        avg = sum(seed_scores) / len(seed_scores)
        print(f"  {label:<22} {avg:6.1f}  {[round(s,1) for s in seed_scores]}")

    print("\n  Note: simulated scores differ from actual because observations are randomly sampled.")
    print("  The relative ordering (NHS effect) is what matters.")


if __name__ == "__main__":
    main()
