"""
config.py — SINGLE SOURCE OF TRUTH for all parameters.

If a number changes, change it HERE ONLY.
All other files import from this module — never hardcode these values elsewhere.
"""

import math
import os

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
API_BASE_URL    = "https://api.ainm.no/astar-island"
API_TOKEN       = os.getenv("ASTAR_API_TOKEN", "")   # set via environment variable

# ─────────────────────────────────────────────
# HARD CONSTRAINTS  ← touch these carefully
# ─────────────────────────────────────────────
TOTAL_QUERIES        = 50    # budget for the ENTIRE round across all seeds
NUM_SEEDS            = 5     # seeds per round
QUERIES_PER_SEED     = TOTAL_QUERIES // NUM_SEEDS   # = 10

MAP_WIDTH            = 40
MAP_HEIGHT           = 40
MAP_TOTAL_CELLS      = MAP_WIDTH * MAP_HEIGHT        # = 1600

VIEWPORT_MAX_WIDTH   = 15
VIEWPORT_MAX_HEIGHT  = 15
VIEWPORT_MAX_CELLS   = VIEWPORT_MAX_WIDTH * VIEWPORT_MAX_HEIGHT  # = 225

NUM_TERRAIN_CLASSES  = 6     # confirmed: see TERRAIN_CLASSES below
SIMULATION_YEARS     = 50    # how many time steps the sim runs

# ─────────────────────────────────────────────
# TERRAIN CLASSES  ← confirmed from API docs
# ─────────────────────────────────────────────
TERRAIN_CLASSES = {
    0: "Empty",
    1: "Settlement",
    2: "Port",
    3: "Ruin",
    4: "Forest",
    5: "Mountain",
}
TERRAIN_NAMES = list(TERRAIN_CLASSES.values())   # index → name
TERRAIN_INDICES = {v: k for k, v in TERRAIN_CLASSES.items()}  # name → index

# ─────────────────────────────────────────────
# DERIVED / COVERAGE MATH
# ─────────────────────────────────────────────
TILES_PER_ROW        = math.ceil(MAP_WIDTH  / VIEWPORT_MAX_WIDTH)   # = 3
TILES_PER_COL        = math.ceil(MAP_HEIGHT / VIEWPORT_MAX_HEIGHT)  # = 3
TILES_TO_COVER_MAP   = TILES_PER_ROW * TILES_PER_COL                # = 9  (full map scan)
SPARE_QUERIES        = TOTAL_QUERIES - (NUM_SEEDS * TILES_TO_COVER_MAP)  # = 5

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_DIR             = "data"
OBSERVATIONS_DIR     = f"{DATA_DIR}/observations"
PREDICTIONS_DIR      = f"{DATA_DIR}/predictions"
BUDGET_FILE          = f"{DATA_DIR}/budget.json"
SCORES_FILE          = f"{DATA_DIR}/scores.json"

# ─────────────────────────────────────────────
# SCORING & PREDICTION
# ─────────────────────────────────────────────
# Metric: entropy-weighted KL divergence — lower is better.
#
# ⚠️  CRITICAL: NEVER assign 0.0 to any class.
#     If ground truth is non-zero where you put 0.0 → KL divergence = infinity.
#     Always apply PROB_FLOOR, then renormalize.
#
PROB_FLOOR           = 0.01                                          # minimum per class
UNIFORM_PRIOR        = [1.0 / NUM_TERRAIN_CLASSES] * NUM_TERRAIN_CLASSES  # baseline ~1-5 score
SMOOTHING_EPSILON    = 1e-9  # secondary guard against log(0)
