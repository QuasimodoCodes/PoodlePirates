"""
query_planner.py — Step 6: Plan all viewport queries before spending a single one.

Strategy:
  Phase 1 — Grid scan: 9 tiles × 5 seeds = 45 queries (full map coverage)
  Phase 2 — Spare:     5 queries targeting highest-settlement-density zones

Tile anchors at [0, 15, 25] for both x and y.
  - NOT [0, 15, 30] — x=30 + w=15 = 45, out of bounds on 40-wide map
  - x=25 + w=15 = 40, exactly fits. Slight overlap at [25-29] is intentional.

Output: ordered list of Query objects — pass to runner.py to execute.
"""

from dataclasses import dataclass
from typing import List, Tuple
import config


@dataclass
class Query:
    """One planned simulate() call."""
    seed_index: int
    x: int
    y: int
    w: int
    h: int
    phase: str       # "grid" or "spare"
    tile_id: str     # e.g. "s0_t3" = seed 0, tile 3

    def __repr__(self):
        return (f"Query(seed={self.seed_index} tile={self.tile_id} "
                f"x={self.x} y={self.y} w={self.w} h={self.h} [{self.phase}])")


# ─────────────────────────────────────────────
# TILE GRID
# ─────────────────────────────────────────────

# Anchors at [0, 15, 25] — NOT [0, 15, 30]
# Each viewport is 15×15 = 225 cells
TILE_ANCHORS = [0, 15, 25]   # used for both x and y

def _grid_tiles() -> List[Tuple[int, int, int, int]]:
    """
    Returns list of (x, y, w, h) for 9 non-overlapping tiles covering 40×40.

    Tile layout:
      (0,0)───(15,0)───(25,0)
        │  t0  │  t1  │  t2  │
      (0,15)──(15,15)─(25,15)
        │  t3  │  t4  │  t5  │
      (0,25)──(15,25)─(25,25)
        │  t6  │  t7  │  t8  │
                              40×40
    """
    tiles = []
    for y_anchor in TILE_ANCHORS:
        for x_anchor in TILE_ANCHORS:
            tiles.append((
                x_anchor,
                y_anchor,
                config.VIEWPORT_MAX_WIDTH,   # 15
                config.VIEWPORT_MAX_HEIGHT,  # 15
            ))
    return tiles


# ─────────────────────────────────────────────
# SPARE QUERY PLACEMENT
# ─────────────────────────────────────────────

def _spare_targets(seed_maps=None) -> List[Tuple[int, int, int, int]]:
    """
    Pick x,y anchors for the 5 spare queries.
    If seed_maps provided: target highest settlement density zone.
    Fallback: centre of the map (most activity tends to cluster away from ocean borders).
    """
    if seed_maps:
        # Find 15×15 window with most initial settlements across all seeds
        best_score = -1
        best_anchor = (12, 12)  # default: centre-ish

        for anchor_x in range(0, config.MAP_WIDTH - config.VIEWPORT_MAX_WIDTH + 1, 5):
            for anchor_y in range(0, config.MAP_HEIGHT - config.VIEWPORT_MAX_HEIGHT + 1, 5):
                score = 0
                for sm in seed_maps:
                    for (sy, sx) in sm.settlement_positions:
                        if (anchor_x <= sx < anchor_x + config.VIEWPORT_MAX_WIDTH and
                                anchor_y <= sy < anchor_y + config.VIEWPORT_MAX_HEIGHT):
                            score += 1
                if score > best_score:
                    best_score = score
                    best_anchor = (anchor_x, anchor_y)

        bx, by = best_anchor
        # 5 spare queries: use best anchor and 4 neighbours spread around it
        return [
            (bx,      by,      config.VIEWPORT_MAX_WIDTH, config.VIEWPORT_MAX_HEIGHT),
            (max(0, bx - 5), by,      config.VIEWPORT_MAX_WIDTH, config.VIEWPORT_MAX_HEIGHT),
            (bx,      max(0, by - 5), config.VIEWPORT_MAX_WIDTH, config.VIEWPORT_MAX_HEIGHT),
            (min(25, bx + 5), by,     config.VIEWPORT_MAX_WIDTH, config.VIEWPORT_MAX_HEIGHT),
            (bx,      min(25, by + 5), config.VIEWPORT_MAX_WIDTH, config.VIEWPORT_MAX_HEIGHT),
        ]

    # Fallback: spread across interior of map
    return [
        (12, 12, config.VIEWPORT_MAX_WIDTH, config.VIEWPORT_MAX_HEIGHT),
        (0,  12, config.VIEWPORT_MAX_WIDTH, config.VIEWPORT_MAX_HEIGHT),
        (25, 12, config.VIEWPORT_MAX_WIDTH, config.VIEWPORT_MAX_HEIGHT),
        (12, 0,  config.VIEWPORT_MAX_WIDTH, config.VIEWPORT_MAX_HEIGHT),
        (12, 25, config.VIEWPORT_MAX_WIDTH, config.VIEWPORT_MAX_HEIGHT),
    ]


# ─────────────────────────────────────────────
# MAIN PLANNER
# ─────────────────────────────────────────────

def build_query_plan(seed_maps=None, total_budget: int = config.TOTAL_QUERIES) -> List[Query]:
    """
    Build the full ordered list of queries to execute.

    Phase 1 (grid): 9 tiles × 5 seeds = 45 queries — full map coverage
    Phase 2 (spare): 5 queries on highest-settlement zones

    Total: 50 queries = exactly the full budget.

    Args:
        seed_maps: optional list of SeedMap from initial_analyzer (used for spare targeting)
        total_budget: total queries allowed (default 50)

    Returns:
        Ordered list of Query objects — execute in this order.
    """
    queries: List[Query] = []
    tiles = _grid_tiles()  # 9 tiles

    # Phase 1: grid scan — interleave seeds so we distribute budget evenly
    # Order: seed0_tile0, seed1_tile0, ..., seed4_tile0, seed0_tile1, ...
    # This means if we crash mid-run, all seeds have partial coverage (not seed0 complete, rest nothing)
    for tile_idx, (x, y, w, h) in enumerate(tiles):
        for seed_idx in range(config.NUM_SEEDS):
            queries.append(Query(
                seed_index=seed_idx,
                x=x, y=y, w=w, h=h,
                phase="grid",
                tile_id=f"s{seed_idx}_t{tile_idx}",
            ))

    # Phase 2: spare queries — assign one per seed on the hottest zone
    spare_targets = _spare_targets(seed_maps)
    for spare_idx, (x, y, w, h) in enumerate(spare_targets):
        seed_idx = spare_idx % config.NUM_SEEDS   # distribute across seeds
        queries.append(Query(
            seed_index=seed_idx,
            x=x, y=y, w=w, h=h,
            phase="spare",
            tile_id=f"s{seed_idx}_spare{spare_idx}",
        ))

    assert len(queries) == total_budget, (
        f"Query plan has {len(queries)} queries but budget is {total_budget}"
    )
    return queries


def print_plan_summary(queries: List[Query]) -> None:
    """Print a compact summary of the query plan."""
    grid_q  = [q for q in queries if q.phase == "grid"]
    spare_q = [q for q in queries if q.phase == "spare"]

    print(f"\n  Query Plan Summary")
    print(f"  {'─'*40}")
    print(f"  Total queries planned: {len(queries)} / {config.TOTAL_QUERIES}")
    print(f"  Grid queries:          {len(grid_q)}  (9 tiles × {config.NUM_SEEDS} seeds)")
    print(f"  Spare queries:         {len(spare_q)}")
    print(f"\n  Tile grid anchors: x={TILE_ANCHORS}, y={TILE_ANCHORS}")
    print(f"  Each tile: {config.VIEWPORT_MAX_WIDTH}×{config.VIEWPORT_MAX_HEIGHT} = "
          f"{config.VIEWPORT_MAX_WIDTH * config.VIEWPORT_MAX_HEIGHT} cells")
    print(f"\n  Spare query targets:")
    for q in spare_q:
        print(f"    seed={q.seed_index} x={q.x} y={q.y}")
