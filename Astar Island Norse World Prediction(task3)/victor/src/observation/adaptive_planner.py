"""
adaptive_planner.py — Two-phase query planning.

Phase 1 (25 queries): 5 volatile-targeted tiles x 5 seeds.
  Greedily picks tiles covering the most settlements/ports (high-uncertainty cells).
  Tested: volatile targeting (72.03) > fixed spread (71.34).

Phase 2 (25 queries): 5 spatially-spread tiles x 5 seeds.
  Fixed corners + centre for calibration diversity and broad coverage.
  Overlap between phases gives 2 observations for cells in both.
"""

import math
from typing import List, Dict, Tuple, Set

import config
from src.observation.query_planner import Query

TILE_W = config.VIEWPORT_MAX_WIDTH    # 15
TILE_H = config.VIEWPORT_MAX_HEIGHT   # 15

# Spread tiles: corners + centre — maximum spatial diversity (used for Phase 2)
SPREAD_ANCHORS = [(0, 0), (25, 0), (0, 25), (25, 25), (12, 12)]

PHASE1_QUERIES = 25   # 5 tiles x 5 seeds
PHASE2_QUERIES = 25   # 5 tiles x 5 seeds

# Keep old name as alias for backwards compatibility with imports
PHASE1_ANCHORS = SPREAD_ANCHORS


# ─────────────────────────────────────────────
# VOLATILE TILE SELECTION
# ─────────────────────────────────────────────

def _get_all_tile_anchors():
    """All valid 15x15 tile anchors on a 40x40 map."""
    return [
        (ax, ay)
        for ay in range(config.MAP_HEIGHT - TILE_H + 1)
        for ax in range(config.MAP_WIDTH - TILE_W + 1)
    ]


def select_volatile_tiles(seed_map, n_tiles: int = 5) -> List[Tuple[int, int]]:
    """
    Greedily select tiles covering the most volatile cells.
    Settlements/Ports are most volatile (highest entropy in ground truth).
    """
    # Volatility weights by initial terrain code
    VOLATILE_CODES = {1: 3.0, 2: 3.0, 3: 2.0}   # Settlement, Port, Ruin
    MEDIUM_CODES = {11: 0.5, 4: 0.3}              # Plains, Forest

    all_anchors = _get_all_tile_anchors()
    anchor_cells = {a: set(_covered_cells(*a)) for a in all_anchors}

    covered: Set[Tuple[int, int]] = set()
    selected = []

    for _ in range(n_tiles):
        best, best_score = None, -1.0
        for a in all_anchors:
            uncovered = anchor_cells[a] - covered
            score = 0.0
            for y, x in uncovered:
                code = seed_map.cells[y][x].initial_code
                score += VOLATILE_CODES.get(code, MEDIUM_CODES.get(code, 0.0))
            if score > best_score:
                best_score, best = score, a
        if best is None:
            break
        selected.append(best)
        covered |= anchor_cells[best]

    return selected


# ─────────────────────────────────────────────
# PHASE 1 — Volatile targeting
# ─────────────────────────────────────────────

def build_phase1_queries(seed_maps) -> List[Query]:
    """
    25 queries: 5 volatile-targeted tiles x 5 seeds.
    Each seed gets its own tile selection based on its initial map.
    Interleaved by tile then seed — crash-safe (all seeds get partial coverage).
    """
    per_seed_tiles = {}
    for sm in seed_maps:
        tiles = select_volatile_tiles(sm, n_tiles=5)
        per_seed_tiles[sm.seed_index] = tiles
        anchors_str = "  ".join(f"({a[0]},{a[1]})" for a in tiles)
        print(f"  Seed {sm.seed_index} volatile tiles: {anchors_str}")

    queries = []
    for tile_idx in range(5):
        for sm in seed_maps:
            tiles = per_seed_tiles[sm.seed_index]
            if tile_idx < len(tiles):
                ax, ay = tiles[tile_idx]
                queries.append(Query(
                    seed_index=sm.seed_index,
                    x=ax, y=ay,
                    w=TILE_W, h=TILE_H,
                    phase="phase1",
                    tile_id=f"s{sm.seed_index}_p1t{tile_idx}",
                ))
    return queries


# ─────────────────────────────────────────────
# PHASE 2 — entropy-guided
# ─────────────────────────────────────────────

def _entropy(dist: List[float]) -> float:
    return -sum(p * math.log(p) for p in dist if p > 1e-12)


def _covered_cells(ax: int, ay: int) -> List[Tuple[int, int]]:
    """All (row, col) cells covered by a 15×15 tile at anchor (ax, ay)."""
    return [
        (ay + dy, ax + dx)
        for dy in range(TILE_H)
        for dx in range(TILE_W)
        if ay + dy < config.MAP_HEIGHT and ax + dx < config.MAP_WIDTH
    ]


def build_phase2_queries(
    seed_maps,
    obs_index: Dict[Tuple[int, int, int], List[int]],
    predictions: List[List[List[List[float]]]],
    n_tiles: int = 5,
) -> List[Query]:
    """
    25 entropy-guided queries: greedily pick 5 tiles per seed that cover
    the highest-entropy unobserved cells.

    Args:
        seed_maps:   list of SeedMap
        obs_index:   {(seed, y, x): [codes]} from Phase 1 observations
        predictions: intermediate H×W×6 tensors built after Phase 1 calibration
        n_tiles:     tiles to add per seed (default 5)

    Returns:
        ordered list of Query objects (interleaved across seeds)
    """
    # All valid tile anchors: x+15 ≤ 40, y+15 ≤ 40
    all_anchors = [
        (ax, ay)
        for ay in range(config.MAP_HEIGHT - TILE_H + 1)
        for ax in range(config.MAP_WIDTH - TILE_W + 1)
    ]

    # Precompute covered cells per anchor (expensive to repeat)
    anchor_cells = {(ax, ay): _covered_cells(ax, ay) for ax, ay in all_anchors}

    per_seed_tiles: Dict[int, List[Tuple[int, int]]] = {}

    for sm in seed_maps:
        seed_idx = sm.seed_index
        tensor   = predictions[seed_idx]

        # Cells already observed in Phase 1
        observed = {(y, x) for (s, y, x) in obs_index if s == seed_idx}

        # Entropy of unobserved cells only
        entropy_map: Dict[Tuple[int, int], float] = {}
        for y in range(config.MAP_HEIGHT):
            for x in range(config.MAP_WIDTH):
                if (y, x) not in observed:
                    entropy_map[(y, x)] = _entropy(tensor[y][x])

        # Greedy selection: each round pick the tile covering most remaining entropy
        selected   = []
        remaining  = dict(entropy_map)  # shrinks as we "spend" cells

        for _ in range(n_tiles):
            best_anchor = None
            best_score  = -1.0

            for ax, ay in all_anchors:
                score = sum(remaining.get(cell, 0.0) for cell in anchor_cells[(ax, ay)])
                if score > best_score:
                    best_score  = score
                    best_anchor = (ax, ay)

            if best_anchor is None or best_score <= 1e-9:
                break

            selected.append(best_anchor)
            # Remove covered cells so next tile targets different area
            for cell in anchor_cells[best_anchor]:
                remaining.pop(cell, None)

        per_seed_tiles[seed_idx] = selected

    # Interleave: tile_0 for all seeds, then tile_1 for all seeds, ...
    queries = []
    max_tiles = max(len(v) for v in per_seed_tiles.values())
    for tile_idx in range(max_tiles):
        for sm in seed_maps:
            tiles = per_seed_tiles[sm.seed_index]
            if tile_idx < len(tiles):
                ax, ay = tiles[tile_idx]
                queries.append(Query(
                    seed_index=sm.seed_index,
                    x=ax, y=ay,
                    w=TILE_W, h=TILE_H,
                    phase="phase2",
                    tile_id=f"s{sm.seed_index}_p2t{tile_idx}",
                ))

    return queries


def build_phase2_spread_queries(seed_maps) -> List[Query]:
    """
    25 queries: fixed spread tiles (corners + centre) for calibration diversity.
    Cells overlapping with Phase 1 volatile tiles get 2 observations.
    """
    queries = []
    for tile_idx, (ax, ay) in enumerate(SPREAD_ANCHORS):
        for sm in seed_maps:
            queries.append(Query(
                seed_index=sm.seed_index,
                x=ax, y=ay,
                w=TILE_W, h=TILE_H,
                phase="phase2",
                tile_id=f"s{sm.seed_index}_p2t{tile_idx}",
            ))
    return queries


# Keep old name as alias
def build_phase2_overlap_queries(seed_maps) -> List[Query]:
    return build_phase2_spread_queries(seed_maps)


def print_phase_summary(phase: int, queries: List[Query]) -> None:
    by_seed: Dict[int, List[Query]] = {}
    for q in queries:
        by_seed.setdefault(q.seed_index, []).append(q)

    label = 'volatile targeting' if phase == 1 else 'spread coverage'
    print(f"\n  Phase {phase} — {len(queries)} queries ({label})")
    print(f"  {'─'*50}")
    for seed_idx in sorted(by_seed):
        tiles = by_seed[seed_idx]
        coords = "  ".join(f"({q.x},{q.y})" for q in tiles)
        print(f"  Seed {seed_idx}: {coords}")
