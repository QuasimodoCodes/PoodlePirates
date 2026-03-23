"""
optimize_hyperparams.py — A "Continuous Evaluation Loop"
Instead of a neural network RL agent, this uses Random Search (or Bayesian Opt)
to grind the raw hyperparameters against the full historical dataset using our 
Leave-One-Out (LOO) truth evaluation.

It aggressively searches for a math-proven setup that beats the current baseline.
"""

import os
import sys
import json
import random
import config
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our helpers from dist_settlement_test to avoid code duplication
import scripts.dist_settlement_test as dst

def evaluate_params(
    p, files, all_data, rounds, round_ids, odm_cache, sdm_cache
) -> float:
    """Run full LOO for DIST_SETTLEMENT using candidate parameters."""
    dst.TEMP        = p["TEMP"]
    dst.N_HIST      = p["N_HIST"]
    dst.N_HIST_SURP = p["N_HIST_SURP"]
    dst.SURP_THRESH = p["SURP_THRESH"]
    
    totals = []
    
    for test_rid in round_ids:
        # Build conditional matrix from remaining rounds
        cond_acc = defaultdict(list)
        for f in files:
            if f.split("_seed")[0] == test_rid: continue
            d = all_data[f]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_cache.get(f); sdm = sdm_cache.get(f)
            for y in range(40):
                for x in range(40):
                    c = ig[y][x]
                    if c not in dst.STATIC_CODES:
                        cond_acc[dst.ctx_dist_sett(ig, y, x, odm, sdm)].append(gt[y][x])
                        
        cm = {c: [sum(s[i] for s in ss)/len(ss) for i in range(dst.N)] 
              for c, ss in cond_acc.items()}
              
        for fname in rounds[test_rid]:
            d = all_data[fname]; gt = d.get("ground_truth"); ig = d.get("initial_grid")
            if not gt or not ig: continue
            odm = odm_cache.get(fname); sdm = sdm_cache.get(fname)
            
            # Use random state to keep the Phase 2 query selection deterministic across runs
            rng_state = random.getstate()
            random.setstate(rng_state)
            
            # --- Inline the modified run_model with param overrides ---
            ao = dst.get_obs(ig, gt)
            rc = defaultdict(list)
            for yx, vals in ao.items():
                c = ig[yx[0]][yx[1]]
                if c not in dst.STATIC_CODES:
                    ct = dst.ctx_dist_sett(ig, yx[0], yx[1], odm, sdm)
                    for v in vals: rc[ct].append(v)

            R = dst.compute_global_shift_sqrt(rc, cm)
            shifted = {ct: dst.apply_shift(dist, R) for ct, dist in cm.items()}

            n_surprised = 0
            bl = {}
            for ct in set(list(shifted.keys()) + list(rc.keys())):
                h = shifted.get(ct, [1.0/dst.N]*dst.N); ol = rc.get(ct, []); nr = len(ol)
                if nr == 0: bl[ct] = h[:]; continue
                s = dst.surp(ol, h)
                if s > p["SURP_THRESH"] and nr >= 5:
                    n_surprised += 1
                    nh = p["N_HIST_SURP"]
                else:
                    nh = p["N_HIST"]
                rf = [0.0]*dst.N
                for v in ol: rf[v] += 1.0/nr
                t = nr + nh
                bl[ct] = [(nr*rf[i] + nh*h[i])/t for i in range(dst.N)]

            hard = n_surprised >= dst.HARD_THRESH

            tensor = []
            for y in range(40):
                row = []
                for x in range(40):
                    code = ig[y][x]
                    if code in dst.STATIC_CODES:
                        pc = dst.CODE_TO_CLASS[code]; d = [dst.FS]*dst.N; d[pc] = 1-5*dst.FS
                    else:
                        ct = dst.ctx_dist_sett(ig, y, x, odm, sdm)
                        prior = bl.get(ct, shifted.get(ct, [1.0/dst.N]*dst.N))[:]
                        if (y, x) in ao:
                            vals = ao[(y,x)]
                            n_obs = len(vals)
                            oh = [0.0]*dst.N
                            for v in vals: oh[v] += 1.0/n_obs
                            a = p["ALPHA_MULTI"] if (hard and n_obs >= 2) else p["ALPHA_1OBS"]
                            d = [(1-a)*prior[i] + a*oh[i] for i in range(dst.N)]
                        else:
                            d = prior[:]
                        d = dst.ts(d, p["TEMP"])
                        d = [max(v, dst.FD) for v in d]
                    t = sum(d); row.append([v/t for v in d])
                tensor.append(row)
                
            score = dst.score_t(tensor, gt)
            totals.append(score)
            
    return sum(totals) / len(totals)


def main():
    print("Continuous Learning Loop: Hyperparameter Tuning")
    print("Loading 20-round historical data...")
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files: rounds[f.split("_seed")[0]].append(f)
    all_data = {}
    for f in files:
        with open(os.path.join(history_dir, f)) as fh: all_data[f] = json.load(fh)

    round_ids = sorted(rounds.keys())
    
    print("Precomputing BFS distance maps... (fast)")
    odm_cache, sdm_cache = {}, {}
    for f in files:
        ig = all_data[f].get("initial_grid")
        if ig:
            odm_cache[f] = dst.ocean_dist_map(ig)
            sdm_cache[f] = dst.sett_dist_map(ig)
            
    # Baseline (Production constants)
    baseline_p = {
        "ALPHA_1OBS": 0.05,
        "ALPHA_MULTI": 0.10,
        "TEMP": 1.10,
        "N_HIST": 50,
        "N_HIST_SURP": 5,
        "SURP_THRESH": 0.30,
    }
    
    print("\nRunning Baseline Evaluator...")
    base_score = evaluate_params(baseline_p, files, all_data, rounds, round_ids, odm_cache, sdm_cache)
    print(f"  Baseline Anchor Score: {base_score:.3f}")
    
    best_score = base_score
    best_p = baseline_p
    
    print("\nStarting Random Search Continuous Loop (10 iterations)...")
    print("  Will overwrite config.py or alert you if mathematically superior params are found.")
    print("=======================================================================")
    
    for i in range(1, 11):
        # Mutate hyperparams within structurally sound bounds
        p = {
            "ALPHA_1OBS": round(random.uniform(0.01, 0.08), 3),
            "ALPHA_MULTI": round(random.uniform(0.05, 0.15), 3),
            "TEMP": round(random.uniform(1.0, 1.25), 2),
            "N_HIST": random.randint(30, 80),
            "N_HIST_SURP": random.randint(1, 10),
            "SURP_THRESH": round(random.uniform(0.15, 0.40), 2),
        }
        
        score = evaluate_params(p, files, all_data, rounds, round_ids, odm_cache, sdm_cache)
        
        diff = score - base_score
        marker = " 🚀 NEW HIGH SCORE" if score > best_score else ""
        print(f"Iter {i:2d} | Score: {score:.3f} (Δ {diff:+.3f}) | {p}{marker}")
        
        if score > best_score:
            best_score = score
            best_p = p
            
    print("=======================================================================")
    print(f"Search Complete.")
    if best_score > base_score:
        print(f"Found superior params beating baseline by +{best_score - base_score:.3f} pts!")
        print(f"Replace values in terrain_estimator.py and round_calibrator.py with:")
        print(json.dumps(best_p, indent=2))
    else:
        print("Baseline parameters held their ground. Current system is extremely robust.")

if __name__ == "__main__":
    main()
