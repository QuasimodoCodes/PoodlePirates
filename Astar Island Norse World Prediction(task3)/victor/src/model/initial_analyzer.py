"""
initial_analyzer.py — Step 5: Parse and save initial states. Zero query cost.

Responsibilities:
  1. Fetch initial_states from GET /rounds/{id} (free)
  2. Save raw data to data/initial_states.json (never fetch again)
  3. Classify every cell per seed into STATIC or DYNAMIC categories
  4. Print ASCII map of each seed so we can visually inspect the world
  5. Return a CellMap used by query_planner and terrain_estimator
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import config
from src.api.models import RoundDetail, Settlement


# ─────────────────────────────────────────────
# CELL CLASSIFICATION
# ─────────────────────────────────────────────

# Static = never changes → predict with near certainty from initial state alone
STATIC_CODES  = {5, 10}   # Mountain, Ocean

# Dynamic = can change over 50 years → need observation budget here
DYNAMIC_CODES = {0, 1, 2, 3, 4, 11}  # Empty, Settlement, Port, Ruin, Forest, Plains

# Maps internal code → prediction class index
CODE_TO_CLASS = {
    0:  0,   # Empty    → class 0
    1:  1,   # Settlement → class 1
    2:  2,   # Port       → class 2
    3:  3,   # Ruin       → class 3
    4:  4,   # Forest     → class 4
    5:  5,   # Mountain   → class 5  (STATIC)
    10: 0,   # Ocean      → class 0  (STATIC)
    11: 0,   # Plains     → class 0
}

# ASCII display characters for map printing
CODE_TO_CHAR = {
    0:  ".",   # Empty
    1:  "S",   # Settlement
    2:  "P",   # Port
    3:  "R",   # Ruin
    4:  "F",   # Forest
    5:  "▲",   # Mountain
    10: "~",   # Ocean
    11: " ",   # Plains (blank = open land)
}


@dataclass
class CellInfo:
    """Classification for a single cell at a given (y, x) position."""
    y: int
    x: int
    initial_code: int
    initial_class: int
    is_static: bool


@dataclass
class SeedMap:
    """
    Full cell classification for one seed.
    Use this to plan queries and build the terrain estimator.
    """
    seed_index: int
    width: int
    height: int
    cells: List[List[CellInfo]]          # cells[y][x]
    settlements: List[Settlement]
    static_cells: List[CellInfo]         # Mountain + Ocean — predict for free
    dynamic_cells: List[CellInfo]        # everything else — needs observation
    settlement_positions: List[Tuple[int, int]]   # (y, x) of all initial settlements

    def get(self, y: int, x: int) -> CellInfo:
        return self.cells[y][x]

    @property
    def n_static(self) -> int:
        return len(self.static_cells)

    @property
    def n_dynamic(self) -> int:
        return len(self.dynamic_cells)

    @property
    def pct_static(self) -> float:
        total = self.width * self.height
        return self.n_static / total * 100


# ─────────────────────────────────────────────
# MAIN ANALYZER
# ─────────────────────────────────────────────

def fetch_and_save(client, round_id: str) -> RoundDetail:
    """
    Fetch round detail (free), save raw JSON to disk.
    If already saved, load from disk instead of calling API again.
    """
    save_path = os.path.join(config.DATA_DIR, "initial_states.json")

    if os.path.exists(save_path):
        print(f"  Loading initial states from disk: {save_path}")
        with open(save_path) as f:
            raw = json.load(f)
        return raw  # raw dict, not RoundDetail — caller uses build_seed_maps()

    print(f"  Fetching initial states from API (free)...")
    detail = client.get_round_detail(round_id)

    # Save immediately
    os.makedirs(config.DATA_DIR, exist_ok=True)
    payload = {
        "round_id": round_id,
        "map_width": detail.map_width,
        "map_height": detail.map_height,
        "seeds_count": detail.seeds_count,
        "initial_states": [
            {
                "grid": state.grid,
                "settlements": [
                    {
                        "x": s.x, "y": s.y,
                        "has_port": s.has_port, "alive": s.alive
                    }
                    for s in state.settlements
                ]
            }
            for state in detail.initial_states
        ]
    }
    with open(save_path, "w") as f:
        json.dump(payload, f)
    print(f"  Saved to {save_path}")
    return payload


def build_seed_maps(raw: dict) -> List[SeedMap]:
    """
    Build SeedMap objects from raw saved dict.
    Classifies every cell as static or dynamic.
    """
    seed_maps = []
    W = raw["map_width"]
    H = raw["map_height"]

    for seed_idx, state in enumerate(raw["initial_states"]):
        grid = state["grid"]
        raw_settlements = state["settlements"]

        settlements = [
            Settlement(
                x=s["x"], y=s["y"],
                has_port=s.get("has_port", False),
                alive=s.get("alive", True)
            )
            for s in raw_settlements
        ]

        cells = []
        static_cells = []
        dynamic_cells = []

        for y in range(H):
            row = []
            for x in range(W):
                code = grid[y][x]
                cls = CODE_TO_CLASS.get(code, 0)
                is_static = code in STATIC_CODES
                cell = CellInfo(y=y, x=x, initial_code=code,
                                initial_class=cls, is_static=is_static)
                row.append(cell)
                if is_static:
                    static_cells.append(cell)
                else:
                    dynamic_cells.append(cell)
            cells.append(row)

        settlement_positions = [(s.y, s.x) for s in settlements]

        seed_maps.append(SeedMap(
            seed_index=seed_idx,
            width=W,
            height=H,
            cells=cells,
            settlements=settlements,
            static_cells=static_cells,
            dynamic_cells=dynamic_cells,
            settlement_positions=settlement_positions,
        ))

    return seed_maps


def print_map(seed_map: SeedMap) -> None:
    """Print ASCII map of initial terrain for visual inspection."""
    print(f"\n  Seed {seed_map.seed_index} — Initial Map  "
          f"({seed_map.n_static} static={seed_map.pct_static:.0f}%, "
          f"{seed_map.n_dynamic} dynamic)")
    print("  " + "─" * seed_map.width)
    for y in range(seed_map.height):
        row_str = "".join(
            CODE_TO_CHAR.get(seed_map.cells[y][x].initial_code, "?")
            for x in range(seed_map.width)
        )
        print(f"  {row_str}")
    print("  " + "─" * seed_map.width)

    # Settlement summary
    for s in seed_map.settlements:
        port_tag = "[PORT]" if s.has_port else ""
        print(f"    Settlement ({s.x:2d},{s.y:2d}) {port_tag}")

    # Terrain code frequency
    counts: Dict[int, int] = {}
    for row in seed_map.cells:
        for cell in row:
            counts[cell.initial_code] = counts.get(cell.initial_code, 0) + 1
    total = seed_map.width * seed_map.height
    code_names = {0:"Empty",1:"Settlement",2:"Port",3:"Ruin",
                  4:"Forest",5:"Mountain",10:"Ocean",11:"Plains"}
    print()
    for code, count in sorted(counts.items()):
        name = code_names.get(code, f"?{code}")
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        tag = " ← STATIC" if code in STATIC_CODES else ""
        print(f"    {CODE_TO_CHAR.get(code,'?')} code {code:2d} {name:<12} "
              f"{count:4d} ({pct:4.1f}%) {bar}{tag}")


def analyze(client, round_id: str, verbose: bool = True) -> List[SeedMap]:
    """
    Full pipeline: fetch → save → classify → print.
    Returns list of SeedMap (one per seed).
    """
    raw = fetch_and_save(client, round_id)
    seed_maps = build_seed_maps(raw)

    if verbose:
        for sm in seed_maps:
            print_map(sm)

    return seed_maps
