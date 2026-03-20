"""
adaptive_planner.py — Two-phase query planning.

Phase 1 (25 queries): 5 spatially-spread tiles x 5 seeds.
  Covers ~69% of the map. Enough to calibrate round dynamics.

Phase 2 (25 queries): Repeat Phase 1 tiles (overlap strategy).
  Querying the same tiles twice gives 2 independent observations per cell,
  reducing stochastic noise. Tested: overlap (73.29) > entropy-guided (72.48).
"""

import math
from typing import List, Dict, Tuple

import config
from src.observation.query_planner import Query

TILE_W = config.VIEWPORT_MAX_WIDTH    # 15
TILE_H = config.VIEWPORT_MAX_HEIGHT   # 15

# Phase 1: 5 tiles covering corners + centre — maximum spatial spread
# (0,0) top-left | (25,0) top-right | (0,25) bottom-left | (25,25) bottom-right | (12,12) centre
PHASE1_ANCHORS = [(0, 0), (25, 0), (0, 25), (25, 25), (12, 12)]

PHASE1_QUERIES = 25   # 5 tiles × 5 seeds
PHASE2_QUERIES = 25   # 5 tiles × 5 seeds


# ─────────────────────────────────────────────
# PHASE 1
# ─────────────────────────────────────────────

def build_phase1_queries(seed_maps) -> List[Query]:
    """
    25 fixed queries: 5 corner+centre tiles × 5 seeds.
    Interleaved by tile then seed — crash-safe (all seeds get partial coverage).
    """
    queries = []
    for tile_idx, (ax, ay) in enumerate(PHASE1_ANCHORS):
        for sm in seed_maps:
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


def build_phase2_overlap_queries(seed_maps) -> List[Query]:
    """
    25 queries: repeat Phase 1 tiles for a second observation.
    Two observations per cell reduces stochastic noise.
    """
    queries = []
    for tile_idx, (ax, ay) in enumerate(PHASE1_ANCHORS):
        for sm in seed_maps:
            queries.append(Query(
                seed_index=sm.seed_index,
                x=ax, y=ay,
                w=TILE_W, h=TILE_H,
                phase="phase2",
                tile_id=f"s{sm.seed_index}_p2t{tile_idx}",
            ))
    return queries


def print_phase_summary(phase: int, queries: List[Query]) -> None:
    by_seed: Dict[int, List[Query]] = {}
    for q in queries:
        by_seed.setdefault(q.seed_index, []).append(q)

    print(f"\n  Phase {phase} — {len(queries)} queries "
          f"({'fixed tiles' if phase == 1 else 'entropy-guided'})")
    print(f"  {'─'*50}")
    for seed_idx in sorted(by_seed):
        tiles = by_seed[seed_idx]
        coords = "  ".join(f"({q.x},{q.y})" for q in tiles)
        print(f"  Seed {seed_idx}: {coords}")
