"""
api_discovery.py — Step 4: Probe all FREE endpoints. Zero queries spent.

Run this FIRST before any simulate() calls to:
  1. Confirm your token works
  2. Find the active round_id
  3. See the initial_states structure (terrain codes, settlements)
  4. Confirm budget (queries_used / queries_max)
  5. Verify map dimensions match config (40×40)

Run from the victor/ folder:
    python -m scripts.api_discovery

Or with token inline:
    ASTAR_API_TOKEN=eyJ... python -m scripts.api_discovery
"""

import os
import sys
import json

# Allow running from victor/ folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.client import AstarClient
import config


def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def main() -> None:
    token = os.getenv("ASTAR_API_TOKEN", "")
    if not token:
        # Try loading from .env file in the victor/ folder
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ASTAR_API_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if not token:
        print("ERROR: No API token found.")
        print("Set ASTAR_API_TOKEN environment variable or add it to .env file.")
        sys.exit(1)

    client = AstarClient(token=token)

    # ─── 1. List rounds ───────────────────────────────────────────
    section("1. GET /rounds — All rounds")
    rounds = client.get_rounds()
    for r in rounds:
        print(f"  id={r.id}  status={r.status}  map={r.map_width}×{r.map_height}  seeds={r.seeds_count}")

    active = [r for r in rounds if r.status == "active"]
    if not active:
        print("\n  ⚠️  No active round. Wait for admin to start one.")
        sys.exit(0)

    round_id = active[0].id
    print(f"\n  ✅ Active round: {round_id}")

    # ─── 2. Round detail + initial_states ─────────────────────────
    section("2. GET /rounds/{id} — Initial states (FREE)")
    detail = client.get_round_detail(round_id)

    print(f"  Map size:    {detail.map_width} × {detail.map_height}")
    print(f"  Seeds count: {detail.seeds_count}")

    # Validate against config
    if detail.map_width != config.MAP_WIDTH or detail.map_height != config.MAP_HEIGHT:
        print(f"  ⚠️  MAP SIZE MISMATCH: API says {detail.map_width}×{detail.map_height}, config says {config.MAP_WIDTH}×{config.MAP_HEIGHT}")
    else:
        print(f"  ✅ Map size matches config ({config.MAP_WIDTH}×{config.MAP_HEIGHT})")

    if detail.seeds_count != config.NUM_SEEDS:
        print(f"  ⚠️  SEEDS MISMATCH: API says {detail.seeds_count}, config says {config.NUM_SEEDS}")
    else:
        print(f"  ✅ Seeds count matches config ({config.NUM_SEEDS})")

    # Show terrain code distribution for each seed
    print()
    for seed_idx, state in enumerate(detail.initial_states):
        flat = [code for row in state.grid for code in row]
        counts = {}
        for code in flat:
            counts[code] = counts.get(code, 0) + 1

        code_names = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin",
                      4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains"}

        print(f"  Seed {seed_idx} — {len(flat)} cells, {len(state.settlements)} initial settlements")
        for code, count in sorted(counts.items()):
            name = code_names.get(code, f"Unknown({code})")
            pct = count / len(flat) * 100
            bar = "█" * int(pct / 2)
            print(f"    code {code:2d} ({name:<12}) {count:4d} cells  {pct:5.1f}%  {bar}")

        # Show first 3 settlements
        for s in state.settlements[:3]:
            port_tag = " [PORT]" if s.has_port else ""
            print(f"    Settlement at ({s.x:2d},{s.y:2d}){port_tag}")
        if len(state.settlements) > 3:
            print(f"    ... and {len(state.settlements) - 3} more")
        print()

    # ─── 3. Budget ────────────────────────────────────────────────
    section("3. GET /budget — Query budget")
    budget = client.get_budget(round_id)
    print(f"  Queries used:      {budget.queries_used} / {budget.queries_max}")
    print(f"  Queries remaining: {budget.queries_remaining}")
    print(f"  Round active:      {budget.active}")

    if budget.is_exhausted:
        print("  ☠️  BUDGET EXHAUSTED — no more simulate() calls allowed")
    elif budget.queries_remaining < 10:
        print(f"  ⚠️  Low budget — only {budget.queries_remaining} queries left")
    else:
        print(f"  ✅ Budget looks good — {budget.queries_remaining} queries available")

    # ─── 4. My rounds (scores) ────────────────────────────────────
    section("4. GET /my-rounds — Team scores")
    try:
        my_rounds = client.get_my_rounds()
        print(json.dumps(my_rounds, indent=2))
    except Exception as e:
        print(f"  (skipped: {e})")

    # ─── Summary ──────────────────────────────────────────────────
    section("SUMMARY")
    print(f"  round_id = {round_id}")
    print(f"  Budget:   {budget.queries_used} used / {budget.queries_max} max  ({budget.queries_remaining} remaining)")
    print(f"  Seeds:    {detail.seeds_count}")
    print(f"  Map:      {detail.map_width}×{detail.map_height}")
    print()
    print("  Next step: run the observation pipeline (Step 6-7)")
    print(f"  Expected query plan: 9 tiles × {detail.seeds_count} seeds = 45 queries")


if __name__ == "__main__":
    main()
