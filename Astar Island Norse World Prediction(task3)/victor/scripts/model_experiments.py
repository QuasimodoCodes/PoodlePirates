"""
model_experiments.py - Test advanced models on historical data.

Experiments:
  1. Feature-based (sklearn) - spatial features per cell
  2. Conditional transition matrix - split by neighbor context
  3. Dirichlet smoothing - better uncertainty than flat floor
  4. Observation agreement weighting
  5. Combined best

Run from victor/ folder:
    python -m scripts.model_experiments
"""

import os, sys, json, math, random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model.initial_analyzer import CODE_TO_CLASS, STATIC_CODES
from src.observation.adaptive_planner import PHASE1_ANCHORS, TILE_W, TILE_H, _entropy, _covered_cells

N_CLASSES = config.NUM_TERRAIN_CLASSES
FLOOR_STATIC = 1e-5
N_MC = 5
random.seed(42)
np.random.seed(42)

# =============================================================================
# SCORING
# =============================================================================

def kl_divergence(p, q):
    result = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-12:
            result += pi * math.log(pi / max(qi, 1e-12))
    return result

def score_tensor(prediction, ground_truth):
    H, W = len(ground_truth), len(ground_truth[0])
    weighted_kl = total_entropy = 0.0
    for y in range(H):
        for x in range(W):
            ent = _entropy(ground_truth[y][x])
            kl  = kl_divergence(ground_truth[y][x], prediction[y][x])
            weighted_kl   += ent * kl
            total_entropy += ent
    if total_entropy < 1e-12:
        return 100.0
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * (weighted_kl / total_entropy))))

# =============================================================================
# SHARED HELPERS
# =============================================================================

def tile_cells(ax, ay):
    return set(_covered_cells(ax, ay))

def sample_observation(gt_dist):
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(gt_dist):
        cumsum += p
        if r < cumsum:
            return i
    return len(gt_dist) - 1

def build_matrix_excluding_round(all_files, exclude_round_id, history_dir):
    counts = defaultdict(lambda: [0.0] * N_CLASSES)
    for fname in all_files:
        if fname.split("_seed")[0] == exclude_round_id:
            continue
        with open(os.path.join(history_dir, fname)) as f:
            data = json.load(f)
        gt, igrid = data.get("ground_truth"), data.get("initial_grid")
        if not gt or not igrid:
            continue
        for y in range(len(igrid)):
            for x in range(len(igrid[0])):
                code = igrid[y][x]
                if code not in STATIC_CODES:
                    counts[code][max(range(N_CLASSES), key=lambda i: gt[y][x][i])] += 1.0
    matrix = {}
    for code, freq in counts.items():
        total = sum(freq)
        matrix[code] = [f/total for f in freq] if total > 0 else [1.0/N_CLASSES]*N_CLASSES
    for code in [0,1,2,3,4,5,10,11]:
        if code not in matrix:
            matrix[code] = [1.0/N_CLASSES]*N_CLASSES
    return matrix

def simulate_calibration(igrid, observed_cells, obs_samples, hist_matrix, n_hist=50):
    round_counts = defaultdict(list)
    for y, x in observed_cells:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            round_counts[code].append(obs_samples[(y, x)])
    blended = {}
    for code in [0,1,2,3,4,5,10,11]:
        hist = hist_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES:
            blended[code] = hist[:]; continue
        obs_list = round_counts.get(code, [])
        n_round = len(obs_list)
        if n_round == 0:
            blended[code] = hist[:]; continue
        rf = [0.0]*N_CLASSES
        for c in obs_list: rf[c] += 1.0/n_round
        t = n_round + n_hist
        blended[code] = [(n_round*rf[i]+n_hist*hist[i])/t for i in range(N_CLASSES)]
    return blended

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(igrid, y, x):
    """Extract spatial features for a single cell."""
    H, W = len(igrid), len(igrid[0])
    code = igrid[y][x]

    # Distance to ocean (BFS would be ideal but expensive, use simple scan)
    dist_ocean = 99
    dist_settlement = 99
    dist_mountain = 99
    n_settlement_1 = 0  # neighbors in 1-ring
    n_forest_1 = 0
    n_ocean_1 = 0
    n_settlement_2 = 0  # neighbors in 2-ring
    n_forest_2 = 0

    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                nc = igrid[ny][nx]
                dist = abs(dy) + abs(dx)  # Manhattan distance

                if nc == 10:  # Ocean
                    dist_ocean = min(dist_ocean, dist)
                    if abs(dy) <= 1 and abs(dx) <= 1:
                        n_ocean_1 += 1
                elif nc == 1:  # Settlement
                    dist_settlement = min(dist_settlement, dist)
                    if abs(dy) <= 1 and abs(dx) <= 1:
                        n_settlement_1 += 1
                    if abs(dy) <= 2 and abs(dx) <= 2:
                        n_settlement_2 += 1
                elif nc == 5:  # Mountain
                    dist_mountain = min(dist_mountain, dist)
                elif nc == 4:  # Forest
                    if abs(dy) <= 1 and abs(dx) <= 1:
                        n_forest_1 += 1
                    if abs(dy) <= 2 and abs(dx) <= 2:
                        n_forest_2 += 1

    # Edge of map (near ocean border)
    edge_dist = min(y, x, H-1-y, W-1-x)

    return [
        code,                    # 0: initial terrain code
        n_settlement_1,          # 1: settlement neighbors (1-ring)
        n_settlement_2,          # 2: settlement neighbors (2-ring)
        n_forest_1,              # 3: forest neighbors (1-ring)
        n_forest_2,              # 4: forest neighbors (2-ring)
        n_ocean_1,               # 5: ocean neighbors (1-ring)
        min(dist_ocean, 10),     # 6: distance to ocean (capped)
        min(dist_settlement, 10),# 7: distance to nearest settlement
        min(dist_mountain, 10),  # 8: distance to mountain
        edge_dist,               # 9: distance to map edge
        1 if code == 1 else 0,   # 10: is_settlement
        1 if code == 4 else 0,   # 11: is_forest
        1 if code == 11 else 0,  # 12: is_plains
    ]

FEATURE_NAMES = [
    "code", "sett_n1", "sett_n2", "forest_n1", "forest_n2",
    "ocean_n1", "dist_ocean", "dist_sett", "dist_mtn",
    "edge_dist", "is_sett", "is_forest", "is_plains"
]


# =============================================================================
# EXPERIMENT 1: Feature-based model (sklearn)
# =============================================================================

def train_feature_model(all_files, exclude_round_id, history_dir):
    """Train a model that predicts class distribution from spatial features."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.multiclass import OneVsRestClassifier
    except ImportError:
        print("  sklearn not installed, skipping feature model")
        return None

    X_train, y_train = [], []

    for fname in all_files:
        if fname.split("_seed")[0] == exclude_round_id:
            continue
        with open(os.path.join(history_dir, fname)) as f:
            data = json.load(f)
        gt, igrid = data.get("ground_truth"), data.get("initial_grid")
        if not gt or not igrid:
            continue
        H, W = len(igrid), len(igrid[0])
        for y in range(H):
            for x in range(W):
                if igrid[y][x] in STATIC_CODES:
                    continue
                features = extract_features(igrid, y, x)
                gt_class = max(range(N_CLASSES), key=lambda i: gt[y][x][i])
                X_train.append(features)
                y_train.append(gt_class)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Use RandomForest with probability estimates
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_with_feature_model(igrid, gt, model, obs_cells, obs_samples, alpha, floor_dyn):
    """Build prediction using feature-based model as prior."""
    H, W = len(igrid), len(igrid[0])

    # Extract features for all dynamic cells
    coords = []
    X_test = []
    for y in range(H):
        for x in range(W):
            if igrid[y][x] not in STATIC_CODES:
                coords.append((y, x))
                X_test.append(extract_features(igrid, y, x))

    X_test = np.array(X_test)
    proba = model.predict_proba(X_test)

    # Map model classes to our 6 classes
    model_classes = list(model.classes_)
    prior_map = {}
    for idx, (y, x) in enumerate(coords):
        dist = [floor_dyn] * N_CLASSES
        for ci, cls in enumerate(model_classes):
            dist[cls] = max(proba[idx][ci], floor_dyn)
        total = sum(dist)
        prior_map[(y, x)] = [v/total for v in dist]

    # Build tensor
    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC]*N_CLASSES
                dist[pc] = 1.0 - FLOOR_STATIC*(N_CLASSES-1)
            else:
                prior = prior_map.get((y,x), [1.0/N_CLASSES]*N_CLASSES)
                if (y,x) in obs_cells and alpha > 0:
                    oh = [0.0]*N_CLASSES
                    oh[obs_samples[(y,x)]] = 1.0
                    dist = [(1-alpha)*prior[i] + alpha*oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, floor_dyn) for v in dist]
            total = sum(dist)
            row.append([v/total for v in dist])
        tensor.append(row)
    return tensor


# =============================================================================
# EXPERIMENT 2: Conditional transition matrix
# =============================================================================

def build_conditional_matrix(all_files, exclude_round_id, history_dir):
    """Build transition matrix conditioned on (initial_code, n_settlement_neighbors)."""
    # Key: (initial_code, n_sett_neighbors_binned) -> class counts
    counts = defaultdict(lambda: [0.0] * N_CLASSES)

    for fname in all_files:
        if fname.split("_seed")[0] == exclude_round_id:
            continue
        with open(os.path.join(history_dir, fname)) as f:
            data = json.load(f)
        gt, igrid = data.get("ground_truth"), data.get("initial_grid")
        if not gt or not igrid:
            continue
        H, W = len(igrid), len(igrid[0])
        for y in range(H):
            for x in range(W):
                code = igrid[y][x]
                if code in STATIC_CODES:
                    continue
                # Count settlement neighbors
                n_sett = 0
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        if dy==0 and dx==0: continue
                        ny, nx = y+dy, x+dx
                        if 0<=ny<H and 0<=nx<W and igrid[ny][nx] == 1:
                            n_sett += 1

                # Bin: 0, 1, 2+
                sett_bin = min(n_sett, 2)

                # Also check ocean adjacency
                near_ocean = 0
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        if dy==0 and dx==0: continue
                        ny, nx = y+dy, x+dx
                        if 0<=ny<H and 0<=nx<W and igrid[ny][nx] == 10:
                            near_ocean = 1
                            break

                key = (code, sett_bin, near_ocean)
                gt_class = max(range(N_CLASSES), key=lambda i: gt[y][x][i])
                counts[key][gt_class] += 1.0

    matrix = {}
    for key, freq in counts.items():
        total = sum(freq)
        if total >= 5:  # Need minimum samples
            matrix[key] = [f/total for f in freq]
    return matrix


def predict_conditional(igrid, gt, cond_matrix, fallback_matrix, obs_cells, obs_samples, alpha, floor_dyn):
    """Use conditional matrix with fallback to simple matrix."""
    H, W = len(igrid), len(igrid[0])
    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC]*N_CLASSES
                dist[pc] = 1.0 - FLOOR_STATIC*(N_CLASSES-1)
            else:
                n_sett = 0
                near_ocean = 0
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        if dy==0 and dx==0: continue
                        ny, nx = y+dy, x+dx
                        if 0<=ny<H and 0<=nx<W:
                            if igrid[ny][nx] == 1: n_sett += 1
                            if igrid[ny][nx] == 10: near_ocean = 1
                sett_bin = min(n_sett, 2)
                key = (code, sett_bin, near_ocean)
                prior = cond_matrix.get(key, fallback_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES))[:]

                if (y,x) in obs_cells and alpha > 0:
                    oh = [0.0]*N_CLASSES
                    oh[obs_samples[(y,x)]] = 1.0
                    dist = [(1-alpha)*prior[i] + alpha*oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, floor_dyn) for v in dist]
            total = sum(dist)
            row.append([v/total for v in dist])
        tensor.append(row)
    return tensor


# =============================================================================
# EXPERIMENT 3: Dirichlet smoothing
# =============================================================================

def build_dirichlet_matrix(all_files, exclude_round_id, history_dir, alpha_prior=1.0):
    """Transition matrix with Dirichlet smoothing instead of flat floor."""
    counts = defaultdict(lambda: [alpha_prior] * N_CLASSES)  # Dirichlet prior

    for fname in all_files:
        if fname.split("_seed")[0] == exclude_round_id:
            continue
        with open(os.path.join(history_dir, fname)) as f:
            data = json.load(f)
        gt, igrid = data.get("ground_truth"), data.get("initial_grid")
        if not gt or not igrid:
            continue
        for y in range(len(igrid)):
            for x in range(len(igrid[0])):
                code = igrid[y][x]
                if code not in STATIC_CODES:
                    gt_class = max(range(N_CLASSES), key=lambda i: gt[y][x][i])
                    counts[code][gt_class] += 1.0

    matrix = {}
    for code, freq in counts.items():
        total = sum(freq)
        matrix[code] = [f/total for f in freq]
    for code in [0,1,2,3,4,5,10,11]:
        if code not in matrix:
            matrix[code] = [1.0/N_CLASSES]*N_CLASSES
    return matrix


# =============================================================================
# EXPERIMENT 4: Observation agreement weighting
# =============================================================================

def predict_obs_agreement(igrid, gt, matrix, obs_cells, obs_samples_1, obs_samples_2, floor_dyn):
    """When 2 observations agree, be more confident. When they disagree, trust prior more."""
    H, W = len(igrid), len(igrid[0])
    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC]*N_CLASSES
                dist[pc] = 1.0 - FLOOR_STATIC*(N_CLASSES-1)
            else:
                prior = matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                if (y,x) in obs_cells:
                    o1 = obs_samples_1[(y,x)]
                    o2 = obs_samples_2[(y,x)]
                    if o1 == o2:
                        # Both agree - use higher alpha
                        alpha = 0.12
                        oh = [0.0]*N_CLASSES
                        oh[o1] = 1.0
                    else:
                        # Disagree - use soft average, lower alpha
                        alpha = 0.04
                        oh = [0.0]*N_CLASSES
                        oh[o1] += 0.5
                        oh[o2] += 0.5
                    dist = [(1-alpha)*prior[i] + alpha*oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, floor_dyn) for v in dist]
            total = sum(dist)
            row.append([v/total for v in dist])
        tensor.append(row)
    return tensor


# =============================================================================
# PIPELINE RUNNERS
# =============================================================================

def run_baseline(igrid, gt, loo_matrix):
    """Current best: overlap + floor=0.001 + a=0.05 + N_HIST=50"""
    H, W = len(igrid), len(igrid[0])
    obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
    obs_cells = set()
    for ax, ay in PHASE1_ANCHORS:
        obs_cells |= tile_cells(ax, ay)

    # Calibrate with both observations
    cal_obs = defaultdict(list)
    for y, x in obs_cells:
        code = igrid[y][x]
        if code not in STATIC_CODES:
            cal_obs[code].append(obs1[(y,x)])
            cal_obs[code].append(obs2[(y,x)])
    blended = {}
    for code in [0,1,2,3,4,5,10,11]:
        hist = loo_matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
        if code in STATIC_CODES: blended[code] = hist[:]; continue
        ol = cal_obs.get(code, [])
        nr = len(ol)
        if nr == 0: blended[code] = hist[:]; continue
        rf = [0.0]*N_CLASSES
        for c in ol: rf[c] += 1.0/nr
        t = nr + 50
        blended[code] = [(nr*rf[i]+50*hist[i])/t for i in range(N_CLASSES)]

    # Average two observations
    tensor = []
    for y in range(H):
        row = []
        for x in range(W):
            code = igrid[y][x]
            if code in STATIC_CODES:
                pc = CODE_TO_CLASS[code]
                dist = [FLOOR_STATIC]*N_CLASSES
                dist[pc] = 1.0 - FLOOR_STATIC*(N_CLASSES-1)
            else:
                prior = blended.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                if (y,x) in obs_cells:
                    oh = [0.0]*N_CLASSES
                    oh[obs1[(y,x)]] += 0.5
                    oh[obs2[(y,x)]] += 0.5
                    dist = [0.95*prior[i]+0.05*oh[i] for i in range(N_CLASSES)]
                else:
                    dist = prior[:]
                dist = [max(v, 0.001) for v in dist]
            t = sum(dist)
            row.append([v/t for v in dist])
        tensor.append(row)
    return score_tensor(tensor, gt)


# =============================================================================
# MAIN
# =============================================================================

def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = sorted(f for f in os.listdir(history_dir) if f.endswith("_analysis.json"))
    rounds = defaultdict(list)
    for f in files:
        rounds[f.split("_seed")[0]].append(f)

    print(f"Model experiments: {len(files)} files, {len(rounds)} rounds, {N_MC} MC trials")
    print()

    # Pre-load data
    all_data = {}
    for fname in files:
        with open(os.path.join(history_dir, fname)) as f:
            all_data[fname] = json.load(f)

    def eval_strategy(label, setup_fn, run_fn):
        """setup_fn(round_id) called once per round, run_fn(igrid, gt, ctx) per seed."""
        print(f"  [{label}]...", end="", flush=True)
        all_scores = []
        for round_id, round_files in sorted(rounds.items()):
            ctx = setup_fn(round_id)
            for fname in round_files:
                d = all_data[fname]
                gt, igrid = d.get("ground_truth"), d.get("initial_grid")
                if not gt or not igrid: continue
                ts = [run_fn(igrid, gt, ctx) for _ in range(N_MC)]
                all_scores.append(sum(ts)/len(ts))
        avg = sum(all_scores)/len(all_scores)
        print(f" avg={avg:.2f}")
        return avg

    results = {}

    # -- Baseline --
    print("=" * 60)
    print("  BASELINE (current best: overlap + floor=0.001)")
    print("=" * 60)
    results["baseline"] = eval_strategy("baseline",
        lambda rid: build_matrix_excluding_round(files, rid, history_dir),
        lambda ig, gt, m: run_baseline(ig, gt, m))

    # -- Experiment 1: Feature-based model --
    print()
    print("=" * 60)
    print("  EXP 1: Random Forest with spatial features")
    print("=" * 60)
    try:
        import sklearn
        def setup_rf(rid):
            loo = build_matrix_excluding_round(files, rid, history_dir)
            model = train_feature_model(files, rid, history_dir)
            return (loo, model)

        def run_rf(igrid, gt, ctx):
            loo, model = ctx
            H, W = len(igrid), len(igrid[0])
            obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
            obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
            obs_cells = set()
            for ax, ay in PHASE1_ANCHORS:
                obs_cells |= tile_cells(ax, ay)
            # Combine obs for calibration
            avg_obs = {}
            for y,x in obs_cells:
                avg_obs[(y,x)] = obs1[(y,x)]  # just use first for calibration
            cal = simulate_calibration(igrid, obs_cells, avg_obs, loo, 50)
            # Use RF model for prior, calibration for adjustment
            # Blend RF prediction with calibrated matrix
            pred = predict_with_feature_model(igrid, gt, model, obs_cells, obs1, 0.05, 0.001)
            return score_tensor(pred, gt)

        results["RF spatial"] = eval_strategy("RF spatial", setup_rf, run_rf)
    except ImportError:
        print("  sklearn not available, skipping")

    # -- Experiment 2: Conditional transition matrix --
    print()
    print("=" * 60)
    print("  EXP 2: Conditional transition matrix (code + neighbors)")
    print("=" * 60)

    def setup_cond(rid):
        loo = build_matrix_excluding_round(files, rid, history_dir)
        cond = build_conditional_matrix(files, rid, history_dir)
        return (loo, cond)

    def run_cond(igrid, gt, ctx):
        loo, cond = ctx
        H, W = len(igrid), len(igrid[0])
        obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
        obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
        obs_cells = set()
        for ax, ay in PHASE1_ANCHORS:
            obs_cells |= tile_cells(ax, ay)
        # Calibrate (using simple matrix for calibration)
        avg_obs = {}
        for y,x in obs_cells:
            avg_obs[(y,x)] = obs1[(y,x)]
        cal_simple = simulate_calibration(igrid, obs_cells, avg_obs, loo, 50)
        # Use conditional matrix for prediction
        pred = predict_conditional(igrid, gt, cond, cal_simple, obs_cells, obs1, 0.05, 0.001)
        return score_tensor(pred, gt)

    results["conditional"] = eval_strategy("conditional", setup_cond, run_cond)

    # -- Experiment 3: Dirichlet smoothing --
    print()
    print("=" * 60)
    print("  EXP 3: Dirichlet smoothing (different alpha priors)")
    print("=" * 60)

    for dir_alpha in [0.1, 0.5, 1.0, 2.0]:
        label = f"dirichlet a={dir_alpha}"
        def setup_dir(rid, da=dir_alpha):
            return build_dirichlet_matrix(files, rid, history_dir, alpha_prior=da)

        def run_dir(igrid, gt, matrix):
            H, W = len(igrid), len(igrid[0])
            obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
            obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
            obs_cells = set()
            for ax, ay in PHASE1_ANCHORS:
                obs_cells |= tile_cells(ax, ay)
            cal_obs = defaultdict(list)
            for y, x in obs_cells:
                code = igrid[y][x]
                if code not in STATIC_CODES:
                    cal_obs[code].append(obs1[(y,x)])
                    cal_obs[code].append(obs2[(y,x)])
            blended = {}
            for code in [0,1,2,3,4,5,10,11]:
                hist = matrix.get(code, [1.0/N_CLASSES]*N_CLASSES)
                if code in STATIC_CODES: blended[code] = hist[:]; continue
                ol = cal_obs.get(code, [])
                nr = len(ol)
                if nr == 0: blended[code] = hist[:]; continue
                rf = [0.0]*N_CLASSES
                for c in ol: rf[c] += 1.0/nr
                t = nr + 50
                blended[code] = [(nr*rf[i]+50*hist[i])/t for i in range(N_CLASSES)]
            tensor = []
            for y in range(H):
                row = []
                for x in range(W):
                    code = igrid[y][x]
                    if code in STATIC_CODES:
                        pc = CODE_TO_CLASS[code]
                        dist = [FLOOR_STATIC]*N_CLASSES
                        dist[pc] = 1.0 - FLOOR_STATIC*(N_CLASSES-1)
                    else:
                        prior = blended.get(code, [1.0/N_CLASSES]*N_CLASSES)[:]
                        if (y,x) in obs_cells:
                            oh = [0.0]*N_CLASSES
                            oh[obs1[(y,x)]] += 0.5
                            oh[obs2[(y,x)]] += 0.5
                            dist = [0.95*prior[i]+0.05*oh[i] for i in range(N_CLASSES)]
                        else:
                            dist = prior[:]
                        dist = [max(v, 0.001) for v in dist]
                    t = sum(dist)
                    row.append([v/t for v in dist])
                tensor.append(row)
            return score_tensor(tensor, gt)

        results[label] = eval_strategy(label, setup_dir, run_dir)

    # -- Experiment 4: Observation agreement --
    print()
    print("=" * 60)
    print("  EXP 4: Observation agreement weighting")
    print("=" * 60)

    def run_agree(igrid, gt, loo):
        H, W = len(igrid), len(igrid[0])
        obs1 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
        obs2 = {(y,x): sample_observation(gt[y][x]) for y in range(H) for x in range(W)}
        obs_cells = set()
        for ax, ay in PHASE1_ANCHORS:
            obs_cells |= tile_cells(ax, ay)
        cal_obs = defaultdict(list)
        for y, x in obs_cells:
            code = igrid[y][x]
            if code not in STATIC_CODES:
                cal_obs[code].append(obs1[(y,x)])
                cal_obs[code].append(obs2[(y,x)])
        blended = {}
        for code in [0,1,2,3,4,5,10,11]:
            hist = loo.get(code, [1.0/N_CLASSES]*N_CLASSES)
            if code in STATIC_CODES: blended[code] = hist[:]; continue
            ol = cal_obs.get(code, [])
            nr = len(ol)
            if nr == 0: blended[code] = hist[:]; continue
            rf = [0.0]*N_CLASSES
            for c in ol: rf[c] += 1.0/nr
            t = nr + 50
            blended[code] = [(nr*rf[i]+50*hist[i])/t for i in range(N_CLASSES)]
        pred = predict_obs_agreement(igrid, gt, blended, obs_cells, obs1, obs2, 0.001)
        return score_tensor(pred, gt)

    results["obs-agreement"] = eval_strategy("obs-agreement",
        lambda rid: build_matrix_excluding_round(files, rid, history_dir),
        run_agree)

    # -- Summary --
    print()
    print("=" * 60)
    print("  SUMMARY - All Results Ranked")
    print("=" * 60)
    ranked = sorted(results.items(), key=lambda x: -x[1])
    for i, (label, score) in enumerate(ranked):
        delta = score - results["baseline"]
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1:2d}. {label:<35} {score:6.2f}  ({delta:+.2f} vs baseline){marker}")


if __name__ == "__main__":
    main()
