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
# Tile anchors: [0, 15, 25] NOT [0, 15, 30]
# x=30 + w=15 = 45 → out of bounds on 40-wide map
# x=25 + w=15 = 40 → exactly fits, slight overlap at [25-29] is fine
TILE_ANCHORS         = [0, 15, 25]                                   # x and y origins for each tile
TILES_PER_ROW        = len(TILE_ANCHORS)                             # = 3
TILES_PER_COL        = len(TILE_ANCHORS)                             # = 3
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
PROB_FLOOR_DYNAMIC   = 0.005  # floor for dynamic cells — conservative to avoid overconfidence
PROB_FLOOR_STATIC    = 1e-5  # floor for static cells (Mountain, Ocean) — excluded from scoring weight anyway
UNIFORM_PRIOR        = [1.0 / NUM_TERRAIN_CLASSES] * NUM_TERRAIN_CLASSES  # baseline ~1-5/100 score
SMOOTHING_EPSILON    = 1e-9  # secondary guard against log(0)

# Internal terrain codes that map to static prediction classes
STATIC_TERRAIN_CODES = {10, 5}   # Ocean (→ class 0), Mountain (→ class 5) — never change
DYNAMIC_TERRAIN_CODES = {0, 1, 2, 3, 4, 11}  # can change after 50 years of simulation
