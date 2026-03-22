"""
plains_diagnostic.py — Deep analysis of why Plains cells are hard to predict.

Plains (code 11) account for 40-60% of all prediction loss.
This script analyzes:
  1. Ground truth distributions for plains cells (how variable are they?)
  2. Plains near ocean vs inland — distribution differences
  3. Per-round variance in plains outcomes
  4. How well our conditional matrix matches actual GT per round
  5. Entropy profile of plains cells
  6. What we're predicting vs what actually happens

python -m scripts.plains_diagnostic
"""

import os, sys, json, math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import _entropy

N = 6
CLASS_NAMES = ["Empty", "Settl", "Port", "Ruin", "Forest", "Mtn"]
CODE_NAMES = {0: "Empty", 1: "Settl", 2: "Port", 3: "Ruin",
              4: "Forest", 5: "Mtn", 10: "Ocean", 11: "Plains"}


def cn(ig, y, x, tc, r=2):
    c = 0
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dy==0 and dx==0: continue
            ny, nx = y+dy, x+dx
            if 0<=ny<40 and 0<=nx<40 and ig[ny][nx]==tc: c+=1
    return c


def ctx(ig, y, x):
    code=ig[y][x]; sn=cn(ig,y,x,1); on=cn(ig,y,x,10)
    sb="sett_hi" if sn>=3 else ("sett_lo" if sn>=1 else "sett_no")
    ob="ocean" if on>=1 else "inland"
    return (code, sb, ob)


def kl(p, q):
    return sum(pi * math.log(pi / max(qi, 1e-12)) for pi, qi in zip(p, q) if pi > 1e-12)


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)

    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    round_ids = sorted(rounds.keys())
    print(f"Plains diagnostic: {len(files)} files, {len(rounds)} rounds")
    print("=" * 100)

    # ── 1. Overall plains GT distribution per context bucket ──
    print("\n1. PLAINS GROUND TRUTH DISTRIBUTION BY CONTEXT")
    print("-" * 80)

    bucket_gts = defaultdict(list)  # ctx -> list of GT distributions
    bucket_entropies = defaultdict(list)
    all_plains_gts = []

    for f in files:
        d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
        if not gt or not ig: continue
        for y in range(40):
            for x in range(40):
                if ig[y][x] == 11:  # Plains
                    ct = ctx(ig, y, x)
                    bucket_gts[ct].append(gt[y][x])
                    bucket_entropies[ct].append(_entropy(gt[y][x]))
                    all_plains_gts.append(gt[y][x])

    for ct in sorted(bucket_gts.keys(), key=lambda c: -len(bucket_gts[c])):
        gts = bucket_gts[ct]
        n = len(gts)
        avg_dist = [sum(g[i] for g in gts) / n for i in range(N)]
        avg_ent = sum(bucket_entropies[ct]) / n
        # Variance: how much do individual GTs deviate from average?
        var_per_class = [sum((g[i] - avg_dist[i])**2 for g in gts) / n for i in range(N)]
        total_var = sum(var_per_class)

        sb, ob = ct[1], ct[2]
        dist_str = "  ".join(f"{CLASS_NAMES[i]}:{avg_dist[i]:.3f}" for i in range(N) if avg_dist[i] > 0.005)
        print(f"  ({sb}, {ob}): n={n:>5}  avg_entropy={avg_ent:.3f}  variance={total_var:.5f}")
        print(f"    avg GT: {dist_str}")

    # ── 2. Per-round variation in plains distributions ──
    print(f"\n\n2. PER-ROUND VARIATION IN PLAINS GT (how much do rounds differ?)")
    print("-" * 80)

    # For each round, compute average plains GT
    round_plains_avg = {}
    for rid in round_ids:
        round_gts = []
        for fname in rounds[rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            for y in range(40):
                for x in range(40):
                    if ig[y][x] == 11:
                        round_gts.append(gt[y][x])
        if round_gts:
            n = len(round_gts)
            round_plains_avg[rid] = [sum(g[i] for g in round_gts) / n for i in range(N)]

    # Overall average
    all_n = len(all_plains_gts)
    overall_avg = [sum(g[i] for g in all_plains_gts) / all_n for i in range(N)]
    print(f"  Overall avg ({all_n} cells): {' '.join(f'{CLASS_NAMES[i]}:{overall_avg[i]:.3f}' for i in range(N))}")
    print()

    print(f"  {'Round':<10}", end="")
    for i in range(N):
        if overall_avg[i] > 0.005:
            print(f" {CLASS_NAMES[i]:>8}", end="")
    print(f" {'KL_from_avg':>11}")
    print(f"  {'-'*10}", end="")
    for i in range(N):
        if overall_avg[i] > 0.005:
            print(f" {'-'*8}", end="")
    print(f" {'-'*11}")

    kl_from_avg = {}
    for rid in sorted(round_plains_avg.keys()):
        avg = round_plains_avg[rid]
        k = kl(avg, overall_avg)
        kl_from_avg[rid] = k
        print(f"  {rid[:8]:<10}", end="")
        for i in range(N):
            if overall_avg[i] > 0.005:
                delta = avg[i] - overall_avg[i]
                marker = "*" if abs(delta) > 0.03 else " "
                print(f" {avg[i]:7.3f}{marker}", end="")
        print(f" {k:11.4f}")

    # ── 3. Plains entropy analysis ──
    print(f"\n\n3. PLAINS ENTROPY PROFILE")
    print("-" * 80)

    # Histogram of plains cell entropies
    ent_bins = [0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    ent_counts = [0] * (len(ent_bins))
    all_ents = []
    for f in files:
        d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
        if not gt or not ig: continue
        for y in range(40):
            for x in range(40):
                if ig[y][x] == 11:
                    e = _entropy(gt[y][x])
                    all_ents.append(e)
                    for bi in range(len(ent_bins) - 1):
                        if e < ent_bins[bi + 1]:
                            ent_counts[bi] += 1
                            break
                    else:
                        ent_counts[-1] += 1

    total = len(all_ents)
    avg_ent = sum(all_ents) / total
    print(f"  Total plains cells: {total}")
    print(f"  Average entropy: {avg_ent:.3f} (max possible: {math.log(N):.3f})")
    print(f"  Entropy distribution:")
    for bi in range(len(ent_bins) - 1):
        pct = 100 * ent_counts[bi] / total
        bar = "#" * int(pct)
        print(f"    [{ent_bins[bi]:.1f}, {ent_bins[bi+1]:.1f}): {ent_counts[bi]:>6} ({pct:5.1f}%) {bar}")
    pct = 100 * ent_counts[-1] / total
    bar = "#" * int(pct)
    print(f"    [{ent_bins[-1]:.1f}, inf):  {ent_counts[-1]:>6} ({pct:5.1f}%) {bar}")

    # ── 4. What class dominates plains cells? ──
    print(f"\n\n4. DOMINANT CLASS ANALYSIS (what outcome is most likely per plains cell?)")
    print("-" * 80)

    dom_counts = defaultdict(int)  # dominant class -> count
    dom_prob = defaultdict(list)   # dominant class -> list of dominant probs
    for g in all_plains_gts:
        max_i = max(range(N), key=lambda i: g[i])
        dom_counts[max_i] += 1
        dom_prob[max_i].append(g[max_i])

    for i in sorted(dom_counts.keys(), key=lambda i: -dom_counts[i]):
        avg_p = sum(dom_prob[i]) / len(dom_prob[i])
        pct = 100 * dom_counts[i] / total
        print(f"  {CLASS_NAMES[i]:<8}: dominates {dom_counts[i]:>6} cells ({pct:5.1f}%), avg dominant P = {avg_p:.3f}")

    # ── 5. Ocean-adjacent vs inland plains ──
    print(f"\n\n5. OCEAN-ADJACENT vs INLAND PLAINS")
    print("-" * 80)

    ocean_gts = []
    inland_gts = []
    for f in files:
        d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
        if not gt or not ig: continue
        for y in range(40):
            for x in range(40):
                if ig[y][x] == 11:
                    on = cn(ig, y, x, 10)
                    if on >= 1:
                        ocean_gts.append(gt[y][x])
                    else:
                        inland_gts.append(gt[y][x])

    def print_group_stats(name, gts):
        n = len(gts)
        avg = [sum(g[i] for g in gts) / n for i in range(N)]
        avg_ent = sum(_entropy(g) for g in gts) / n
        print(f"  {name}: n={n}, avg_entropy={avg_ent:.3f}")
        print(f"    avg GT: {' '.join(f'{CLASS_NAMES[i]}:{avg[i]:.3f}' for i in range(N))}")
        # Dominant class breakdown
        for ci in range(N):
            if avg[ci] > 0.01:
                # How often is this class within 0.1 of max?
                close_count = sum(1 for g in gts if g[ci] > max(g) - 0.1)
                print(f"    {CLASS_NAMES[ci]}: avg P={avg[ci]:.3f}, near-dominant in {100*close_count/n:.1f}% of cells")

    print_group_stats("Ocean-adjacent", ocean_gts)
    print()
    print_group_stats("Inland", inland_gts)

    # ── 6. The prediction gap: what could a perfect predictor achieve? ──
    print(f"\n\n6. PREDICTION CEILING FOR PLAINS")
    print("-" * 80)

    # Precompute per-bucket averages once
    bucket_avg = {}
    for ct, gts in bucket_gts.items():
        n = len(gts)
        bucket_avg[ct] = [sum(g[i] for g in gts) / n for i in range(N)]

    # If we predict the average GT distribution for each context bucket,
    # what KL divergence do we get? (This is the irreducible variance.)
    bucket_kl = defaultdict(list)
    for f in files:
        d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
        if not gt or not ig: continue
        for y in range(40):
            for x in range(40):
                if ig[y][x] == 11:
                    ct = ctx(ig, y, x)
                    avg = bucket_avg[ct]
                    k = kl(gt[y][x], avg)
                    e = _entropy(gt[y][x])
                    bucket_kl[ct].append((e, k))

    print(f"  Irreducible KL (using per-bucket average as prediction):")
    total_wkl = 0.0
    total_te = 0.0
    for ct in sorted(bucket_kl.keys(), key=lambda c: -sum(ek[0]*ek[1] for ek in bucket_kl[c])):
        pairs = bucket_kl[ct]
        wkl = sum(e * k for e, k in pairs)
        te = sum(e for e, k in pairs)
        avg_kl = sum(k for _, k in pairs) / len(pairs)
        total_wkl += wkl
        total_te += te
        sb, ob = ct[1], ct[2]
        print(f"    ({sb}, {ob}): weighted_kl={wkl:.2f}, avg_kl={avg_kl:.4f}, n={len(pairs)}")

    # What score would we get if ONLY plains cells contributed?
    plains_score = max(0, min(100, 100 * math.exp(-3 * (total_wkl / total_te)))) if total_te > 0 else 100
    print(f"\n  Plains-only score (with perfect per-bucket avg): {plains_score:.1f}")
    print(f"  Total weighted KL from plains: {total_wkl:.2f}")
    print(f"  Total entropy from plains: {total_te:.2f}")

    # ── 7. Per-round deviation: which rounds have unusual plains? ──
    print(f"\n\n7. PER-ROUND PLAINS DEVIATION (which rounds break our predictions?)")
    print("-" * 80)

    # For each round, compute how much its plains GT deviates from the
    # leave-one-out average
    for rid in sorted(kl_from_avg.keys(), key=lambda r: -kl_from_avg[r]):
        k = kl_from_avg[rid]
        avg = round_plains_avg[rid]
        # Which classes deviate most from overall?
        deviations = [(i, avg[i] - overall_avg[i]) for i in range(N)]
        deviations.sort(key=lambda x: -abs(x[1]))
        top = deviations[:3]
        dev_str = ", ".join(f"{CLASS_NAMES[i]}:{d:+.3f}" for i, d in top if abs(d) > 0.005)
        marker = " <<<" if k > 0.01 else (" *" if k > 0.005 else "")
        print(f"  {rid[:8]}: KL={k:.4f}  deviations: {dev_str}{marker}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
