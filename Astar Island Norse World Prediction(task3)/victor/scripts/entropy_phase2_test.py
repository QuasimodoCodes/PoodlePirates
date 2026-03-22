"""
entropy_phase2_test.py — LOO test comparing Phase 2 strategies.

Compares:
  SPREAD  (current production): Phase 1 = settlement targeting, Phase 2 = fixed spread anchors
  ENTROPY: Phase 1 = settlement targeting, Phase 2 = entropy-guided greedy tile selection

Entropy-guided Phase 2 greedily picks 5 tiles that maximize total entropy of
unobserved cells (cells NOT already covered by Phase 1).  Entropy is computed
from the model's intermediate predictions (bucket averages after global shift).

All other parameters are identical: N_HIST=50, TEMP=1.10, ctx_new (dist_ocean
for Plains), adaptive alpha, etc.

python -m scripts.entropy_phase2_test
"""

import os, sys, json, math, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import SPREAD_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N = 6; FS = 1e-5; FD = 0.001; TEMP = 1.10
N_HIST = 50; N_HIST_SURP = 5; SURP_THRESH = 0.30
HARD_THRESH = 5
random.seed(42)

ALL_A = [(ax, ay) for ay in range(40 - TILE_H + 1) for ax in range(40 - TILE_W + 1)]
AC = {a: set(_covered_cells(*a)) for a in ALL_A}

CLASS_NAMES = ["Empty", "Settl", "Port", "Ruin", "Forest", "Mtn"]


# ── BFS ocean distance ──

def ocean_dist_map(ig):
    H, W = 40, 40
    dist = [[99]*W for _ in range(H)]
    q = []
    for y in range(H):
        for x in range(W):
            if ig[y][x] == 10:
                dist[y][x] = 0; q.append((y, x))
    head = 0
    while head < len(q):
        cy, cx = q[head]; head += 1
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy+dy, cx+dx
            if 0<=ny<H and 0<=nx<W and dist[ny][nx] > dist[cy][cx]+1:
                dist[ny][nx] = dist[cy][cx]+1; q.append((ny, nx))
    return dist

def bin_od(d):
    if d <= 1: return "od0"
    if d <= 4: return "od1"
    if d <= 10: return "od2"
    return "od3"


# ── Helpers ──

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
            if dy==0 and dx==0: continue
            ny, nx = y+dy, x+dx
            if 0<=ny<40 and 0<=nx<40 and ig[ny][nx]==tc: c+=1
    return c

def ts(d, t):
    if t==1.0: s=sum(d); return [v/s for v in d]
    ld=[math.log(max(p,1e-12))/t for p in d]; mx=max(ld)
    ed=[math.exp(v-mx) for v in ld]; s=sum(ed)
    return [v/s for v in ed]

def surp(ol, h):
    nr=len(ol)
    if nr<3: return 0.0
    rf=[0.0]*N
    for v in ol: rf[v]+=1.0/nr
    kf=sum(rf[i]*math.log(max(rf[i],1e-12)/max(h[i],1e-12)) for i in range(N) if rf[i]>1e-12)
    kr=sum(h[i]*math.log(max(h[i],1e-12)/max(rf[i],1e-12)) for i in range(N) if h[i]>1e-12)
    return (kf+kr)/2

def compute_global_shift_sqrt(rc, cm_hist):
    round_total=[0.0]*N; n_obs=0
    for ct,ol in rc.items():
        for cls in ol: round_total[cls]+=1; n_obs+=1
    if n_obs==0: return [1.0]*N
    round_freq=[c/n_obs for c in round_total]
    hist_total=[0.0]*N; nb=0
    for ct,dist in cm_hist.items():
        for i in range(N): hist_total[i]+=dist[i]
        nb+=1
    if nb==0: return [1.0]*N
    hist_freq=[h/nb for h in hist_total]
    return [math.sqrt(round_freq[i]/hist_freq[i]) if hist_freq[i]>1e-8 else 1.0 for i in range(N)]

def apply_shift(dist, R):
    shifted=[max(dist[i]*R[i],1e-12) for i in range(N)]
    s=sum(shifted); return [v/s for v in shifted]


# ── Context function (NEW: dist_ocean for Plains) ──

def ctx_new(ig, y, x, odm=None):
    """NEW: 4-bin dist_ocean for Plains, binary for others."""
    code=ig[y][x]; sn=cn(ig,y,x,1)
    sb="sh" if sn>=3 else ("sl" if sn>=1 else "sn")
    if code == 11 and odm is not None:
        ob = bin_od(odm[y][x])
    else:
        on=cn(ig,y,x,10)
        ob="oc" if on>=1 else "in"
    return (code, sb, ob)


# ── Query targeting ──

def sel_sett(ig, n=5):
    """Phase 1: greedily pick tiles covering the most settlement cells."""
    ss={(y,x) for y in range(40) for x in range(40) if ig[y][x]==1}
    if not ss: return SPREAD_ANCHORS[:n]
    cov,sel=set(),[]
    for _ in range(n):
        b,bc=None,-1
        for a in ALL_A:
            c=len((AC[a]&ss)-cov)
            if c>bc: bc,b=c,a
        if not b or bc<=0: break
        sel.append(b); cov|=(AC[b]&ss)
    return sel


# ── Entropy-guided Phase 2 tile selection ──

def select_entropy_tiles(ig, phase1_anchors, shifted_cm, odm, n=5):
    """
    Greedily select n tiles for Phase 2 that cover the highest total entropy
    among cells NOT already observed in Phase 1.

    Uses shifted bucket averages (model intermediate predictions) as the
    entropy source.

    Args:
        ig:              40x40 initial grid
        phase1_anchors:  list of (ax, ay) tiles selected for Phase 1
        shifted_cm:      dict mapping context_tuple -> shifted 6-class distribution
        odm:             ocean distance map for this grid
        n:               number of tiles to select

    Returns:
        list of (ax, ay) anchor tuples for Phase 2
    """
    # Determine which cells are already covered by Phase 1 tiles
    phase1_covered = set()
    for a in phase1_anchors:
        phase1_covered |= AC[a]

    # Build per-cell entropy from model's shifted predictions (unobserved cells only)
    cell_entropy = {}
    for y in range(40):
        for x in range(40):
            if (y, x) in phase1_covered:
                continue  # already observed — skip
            code = ig[y][x]
            if code in STATIC_CODES:
                continue  # static cells have ~0 entropy, not worth targeting
            ct = ctx_new(ig, y, x, odm)
            dist = shifted_cm.get(ct, [1.0/N]*N)
            ent = _entropy(dist)
            if ent > 1e-9:
                cell_entropy[(y, x)] = ent

    # Greedy tile selection: pick tile maximizing total entropy of remaining cells
    selected = []
    remaining = dict(cell_entropy)

    for _ in range(n):
        best_anchor = None
        best_score = -1.0

        for a in ALL_A:
            score = sum(remaining.get(cell, 0.0) for cell in AC[a])
            if score > best_score:
                best_score = score
                best_anchor = a

        if best_anchor is None or best_score <= 1e-9:
            break

        selected.append(best_anchor)
        # Remove covered cells so next tile targets a different area
        for cell in AC[best_anchor]:
            remaining.pop(cell, None)

    # Fill with spread anchors if we got fewer than n tiles
    while len(selected) < n and len(selected) < len(SPREAD_ANCHORS):
        a = SPREAD_ANCHORS[len(selected)]
        if a not in selected:
            selected.append(a)

    return selected[:n]


# ── Observation generation ──

def get_obs_spread(ig, gt, phase1_anchors):
    """SPREAD strategy: Phase 1 = settlement tiles, Phase 2 = SPREAD_ANCHORS (fixed)."""
    obs1 = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2 = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
    ao = {}
    for a in phase1_anchors:
        for yx in AC[a]: ao[yx] = [obs1[yx]]
    for a in SPREAD_ANCHORS:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx] = [obs2[yx]]
    return ao


def get_obs_entropy(ig, gt, phase1_anchors, phase2_anchors):
    """ENTROPY strategy: Phase 1 = settlement tiles, Phase 2 = entropy-guided tiles."""
    obs1 = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
    obs2 = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
    ao = {}
    for a in phase1_anchors:
        for yx in AC[a]: ao[yx] = [obs1[yx]]
    for a in phase2_anchors:
        for yx in AC[a]:
            if yx in ao: ao[yx].append(obs2[yx])
            else: ao[yx] = [obs2[yx]]
    return ao


# ── Model ──

def run_model(ig, gt, cm_hist, ao, odm=None):
    """
    Run the prediction model given pre-computed observations (ao).
    Uses ctx_new, adaptive alpha, N_HIST=50, TEMP=1.10.

    Returns: (score, n_surprised)
    """
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_new(ig, yx[0], yx[1], odm)
            for v in vals: rc[ct].append(v)

    R = compute_global_shift_sqrt(rc, cm_hist)
    shifted = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}

    # Count surprised for adaptive alpha
    n_surprised = 0
    bl = {}
    for ct in set(list(shifted.keys()) + list(rc.keys())):
        h = shifted.get(ct, [1.0/N]*N); ol = rc.get(ct, []); nr = len(ol)
        if nr == 0: bl[ct] = h[:]; continue
        s = surp(ol, h)
        if s > SURP_THRESH and nr >= 5:
            n_surprised += 1
            nh = N_HIST_SURP
        else:
            nh = N_HIST
        rf = [0.0]*N
        for v in ol: rf[v] += 1.0/nr
        t = nr + nh
        bl[ct] = [(nr*rf[i] + nh*h[i])/t for i in range(N)]

    hard = n_surprised >= HARD_THRESH

    tensor = []
    for y in range(40):
        row = []
        for x in range(40):
            code = ig[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]; d = [FS]*N; d[pc] = 1-5*FS
            else:
                ct = ctx_new(ig, y, x, odm)
                prior = bl.get(ct, shifted.get(ct, [1.0/N]*N))[:]
                if (y, x) in ao:
                    vals = ao[(y,x)]
                    n_obs = len(vals)
                    oh = [0.0]*N
                    for v in vals: oh[v] += 1.0/n_obs
                    if hard:
                        a = 0.10 if n_obs >= 2 else 0.03
                    else:
                        a = 0.05
                    d = [(1-a)*prior[i] + a*oh[i] for i in range(N)]
                else:
                    d = prior[:]
                d = ts(d, TEMP)
                d = [max(v, FD) for v in d]
            t = sum(d); row.append([v/t for v in d])
        tensor.append(row)
    return score_t(tensor, gt), n_surprised, shifted


def build_shifted_cm(ig, gt, ao, cm_hist, odm):
    """
    Build the shifted conditional matrix from Phase 1 observations only.
    This is the intermediate model state used to compute entropy for Phase 2
    tile selection.  Returns the shifted bucket distributions.
    """
    rc = defaultdict(list)
    for yx, vals in ao.items():
        c = ig[yx[0]][yx[1]]
        if c not in STATIC_CODES:
            ct = ctx_new(ig, yx[0], yx[1], odm)
            for v in vals: rc[ct].append(v)

    R = compute_global_shift_sqrt(rc, cm_hist)
    shifted = {ct: apply_shift(dist, R) for ct, dist in cm_hist.items()}
    return shifted


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    round_ids = sorted(rounds.keys())
    print(f"Entropy Phase 2 test: {len(files)} files, {len(rounds)} rounds")
    print(f"SPREAD:  Phase 1 = settlement targeting, Phase 2 = fixed spread anchors")
    print(f"ENTROPY: Phase 1 = settlement targeting, Phase 2 = entropy-guided greedy selection")
    print(f"Model:   ctx_new (dist_ocean for Plains), adaptive alpha, N_HIST={N_HIST}, TEMP={TEMP}")
    print("=" * 100)

    # Precompute ocean dist maps
    print("Precomputing ocean distance maps...")
    odm_cache = {}
    for f in files:
        ig = all_data[f].get("initial_grid")
        if ig: odm_cache[f] = ocean_dist_map(ig)
    print(f"  Done: {len(odm_cache)} maps.")

    labels = ["SPREAD", "ENTROPY"]
    print(f"\n  {'Round':<10}", end="")
    for l in labels: print(f" {l:>14}", end="")
    print(f" {'Delta':>8} {'NSurp':>6} {'Hard?':>6} {'Entropy tiles':>30}")
    print(f"  {'-'*10}", end="")
    for _ in labels: print(f" {'-'*14}", end="")
    print(f" {'-'*8} {'-'*6} {'-'*6} {'-'*30}")

    totals = {l: [] for l in labels}
    n_hard = 0
    wins = {"SPREAD": 0, "ENTROPY": 0, "TIE": 0}

    for test_rid in round_ids:
        # Build conditional matrix (LOO — exclude test round)
        cond_acc = defaultdict(list)
        for f in files:
            if f.split("_seed")[0] == test_rid: continue
            d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_cache.get(f)
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in STATIC_CODES:
                        cond_acc[ctx_new(ig, y, x, odm)].append(gt[y][x])
        cm_hist = {c: [sum(s[i] for s in ss)/len(ss) for i in range(N)] for c, ss in cond_acc.items()}

        round_scores = {l: [] for l in labels}
        round_nsurp = 0
        round_entropy_tiles = []

        for fname in rounds[test_rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_cache.get(fname)

            # Phase 1 anchors: settlement targeting (same for both strategies)
            phase1_anchors = sel_sett(ig, 5)

            # ── SPREAD strategy ──
            rng_state = random.getstate()
            ao_spread = get_obs_spread(ig, gt, phase1_anchors)
            sc_spread, ns_spread, _ = run_model(ig, gt, cm_hist, ao_spread, odm)
            round_scores["SPREAD"].append(sc_spread)

            # ── ENTROPY strategy ──
            # Step 1: Generate Phase 1 observations (same RNG state for fair comparison)
            random.setstate(rng_state)
            obs1 = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
            phase1_ao = {}
            for a in phase1_anchors:
                for yx in AC[a]: phase1_ao[yx] = [obs1[yx]]

            # Step 2: Build shifted CM from Phase 1 observations to get intermediate predictions
            shifted_cm = build_shifted_cm(ig, gt, phase1_ao, cm_hist, odm)

            # Step 3: Select Phase 2 tiles using entropy of unobserved cells
            phase2_anchors = select_entropy_tiles(ig, phase1_anchors, shifted_cm, odm, n=5)

            # Step 4: Generate Phase 2 observations with a fresh sample
            obs2 = {(y,x): so(gt[y][x]) for y in range(40) for x in range(40)}
            ao_entropy = dict(phase1_ao)
            for a in phase2_anchors:
                for yx in AC[a]:
                    if yx in ao_entropy: ao_entropy[yx].append(obs2[yx])
                    else: ao_entropy[yx] = [obs2[yx]]

            # Step 5: Run model with entropy-guided observations
            sc_entropy, ns_entropy, _ = run_model(ig, gt, cm_hist, ao_entropy, odm)
            round_scores["ENTROPY"].append(sc_entropy)

            round_nsurp = max(round_nsurp, ns_entropy)
            round_entropy_tiles.append(phase2_anchors)

        hard = round_nsurp >= HARD_THRESH
        if hard: n_hard += 1

        avg_spread = sum(round_scores["SPREAD"]) / len(round_scores["SPREAD"])
        avg_entropy = sum(round_scores["ENTROPY"]) / len(round_scores["ENTROPY"])
        delta = avg_entropy - avg_spread

        if delta > 0.01: wins["ENTROPY"] += 1
        elif delta < -0.01: wins["SPREAD"] += 1
        else: wins["TIE"] += 1

        totals["SPREAD"].extend(round_scores["SPREAD"])
        totals["ENTROPY"].extend(round_scores["ENTROPY"])

        # Show entropy tiles from the first seed of this round
        tiles_str = ""
        if round_entropy_tiles:
            tiles_str = " ".join(f"({a[0]},{a[1]})" for a in round_entropy_tiles[0])

        short = test_rid[:8]
        marker = " +" if delta > 0.1 else (" -" if delta < -0.1 else "")
        print(f"  {short:<10} {avg_spread:14.1f} {avg_entropy:14.1f} {delta:+8.2f}{marker:2s} {round_nsurp:>6} {'HARD' if hard else '':>6} {tiles_str:>30}")

    # ── Summary ──
    print(f"\n{'='*100}")
    spread_avg = sum(totals["SPREAD"]) / len(totals["SPREAD"])
    entropy_avg = sum(totals["ENTROPY"]) / len(totals["ENTROPY"])

    print(f"\n  {'Config':<14} {'Avg':>7} {'Delta':>7} {'Min':>7} {'Max':>7} {'Std':>7}")
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for label in labels:
        scores = totals[label]
        avg = sum(scores) / len(scores)
        mn, mx = min(scores), max(scores)
        std = (sum((s - avg)**2 for s in scores) / len(scores)) ** 0.5
        delta = avg - spread_avg
        marker = " <<<" if delta > 0.5 else (" *" if delta > 0.1 else "")
        print(f"  {label:<14} {avg:7.2f} {delta:+7.2f} {mn:7.1f} {mx:7.1f} {std:7.1f}{marker}")

    print(f"\n  Overall delta (ENTROPY - SPREAD): {entropy_avg - spread_avg:+.3f}")
    print(f"  Win/Loss/Tie:  ENTROPY wins {wins['ENTROPY']}, SPREAD wins {wins['SPREAD']}, ties {wins['TIE']}")
    print(f"  Hard rounds detected: {n_hard}/{len(round_ids)}")

    # ── Per-seed breakdown ──
    print(f"\n  Per-seed breakdown:")
    print(f"  {'Seed':<10}", end="")
    for l in labels: print(f" {l:>10}", end="")
    print(f" {'Delta':>8}")
    print(f"  {'-'*10}", end="")
    for _ in labels: print(f" {'-'*10}", end="")
    print(f" {'-'*8}")

    # Group scores by seed position within each round
    n_seeds = 5
    for si in range(n_seeds):
        spread_seed = [totals["SPREAD"][i] for i in range(si, len(totals["SPREAD"]), n_seeds)]
        entropy_seed = [totals["ENTROPY"][i] for i in range(si, len(totals["ENTROPY"]), n_seeds)]
        if spread_seed and entropy_seed:
            s_avg = sum(spread_seed) / len(spread_seed)
            e_avg = sum(entropy_seed) / len(entropy_seed)
            d = e_avg - s_avg
            print(f"  seed {si:<5} {s_avg:10.2f} {e_avg:10.2f} {d:+8.2f}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
