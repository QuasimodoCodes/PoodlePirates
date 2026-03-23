"""
runner.py — Step 7: Execute query plan and save every response to disk.

CRITICAL RULES:
  1. Always check GET /budget before starting — never exceed 50 total
  2. Save response JSON to disk IMMEDIATELY after each query
     (if script crashes at query 43, we keep all 43 observations)
  3. Sleep 0.2s between calls — max 5 req/sec rate limit
  4. Skip queries for already-saved observations (safe to re-run)
"""

import os
import json
import time
from typing import List, Optional

from src.api.client import AstarClient
from src.observation.query_planner import Query, build_query_plan, print_plan_summary
import config


RATE_LIMIT_SLEEP = 0.21   # slightly over 0.20 for safety (5 req/sec limit)


def _obs_path(query: Query) -> str:
    """Canonical file path for a query's saved response."""
    return os.path.join(
        config.OBSERVATIONS_DIR,
        f"seed{query.seed_index}_x{query.x}_y{query.y}_{query.phase}.json"
    )


def already_saved(query: Query) -> bool:
    return os.path.exists(_obs_path(query))


def save_response(query: Query, response_data: dict) -> None:
    os.makedirs(config.OBSERVATIONS_DIR, exist_ok=True)
    payload = {
        "query": {
            "seed_index": query.seed_index,
            "x": query.x, "y": query.y,
            "w": query.w, "h": query.h,
            "phase": query.phase,
            "tile_id": query.tile_id,
        },
        "response": response_data,
    }
    with open(_obs_path(query), "w") as f:
        json.dump(payload, f)


def load_all_observations() -> List[dict]:
    """Load all saved observation files from disk."""
    obs = []
    if not os.path.exists(config.OBSERVATIONS_DIR):
        return obs
    for fname in sorted(os.listdir(config.OBSERVATIONS_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(config.OBSERVATIONS_DIR, fname)) as f:
                obs.append(json.load(f))
    return obs


def run(
    client: AstarClient,
    round_id: str,
    queries: List[Query],
    dry_run: bool = False,
) -> int:
    """
    Execute the query plan. Returns number of new queries made.

    Args:
        client:   authenticated AstarClient
        round_id: active round ID
        queries:  ordered list from query_planner.build_query_plan()
        dry_run:  if True, print what would be queried without spending budget
    """
    os.makedirs(config.OBSERVATIONS_DIR, exist_ok=True)

    # ── 1. Check budget before starting ─────────────────────────────
    budget = client.get_budget(round_id)
    print(f"\n  Budget check: {budget.queries_used} used / {budget.queries_max} max "
          f"({budget.queries_remaining} remaining)")

    if budget.is_exhausted:
        print("  ☠️  Budget exhausted — cannot run any queries.")
        return 0

    if budget.queries_remaining < len([q for q in queries if not already_saved(q)]):
        needed = len([q for q in queries if not already_saved(q)])
        print(f"  ⚠️  Need {needed} queries but only {budget.queries_remaining} remain.")
        print(f"  Will run as many as budget allows.")

    # ── 2. Execute queries ───────────────────────────────────────────
    new_queries = 0
    skipped = 0

    for i, query in enumerate(queries):
        # Skip if already saved
        if already_saved(query):
            skipped += 1
            continue

        # Stop if budget would be exceeded
        if budget.queries_used + new_queries >= budget.queries_max:
            print(f"\n  ⚠️  Budget limit reached after {new_queries} queries. Stopping.")
            break

        if dry_run:
            print(f"  [DRY RUN] Would query: {query}")
            continue

        # Execute
        try:
            result = client.simulate(
                round_id=round_id,
                seed_index=query.seed_index,
                x=query.x, y=query.y,
                w=query.w, h=query.h,
            )

            # Save immediately — before anything else
            response_data = {
                "grid": result.grid,
                "settlements": [
                    {
                        "x": s.x, "y": s.y,
                        "has_port": s.has_port, "alive": s.alive,
                        "population": s.population, "food": s.food,
                        "wealth": s.wealth, "defense": s.defense,
                        "owner_id": s.owner_id,
                    }
                    for s in result.settlements
                ],
                "viewport": {
                    "x": result.viewport.x, "y": result.viewport.y,
                    "w": result.viewport.w, "h": result.viewport.h,
                },
                "queries_used": result.queries_used,
                "queries_max": result.queries_max,
            }
            save_response(query, response_data)
            new_queries += 1

            remaining = result.queries_remaining
            print(f"  [{new_queries:02d}] {query.tile_id:<12} "
                  f"seed={query.seed_index} x={query.x:2d} y={query.y:2d} "
                  f"| budget: {result.queries_used}/{result.queries_max} "
                  f"({remaining} left)")

            # Rate limit
            if remaining > 0:
                time.sleep(RATE_LIMIT_SLEEP)

        except RuntimeError as e:
            if "429" in str(e):
                print(f"\n  ⚠️  Rate limit or budget hit: {e}")
                print(f"  Stopping after {new_queries} queries.")
                break
            else:
                print(f"\n  ❌ Error on {query}: {e}")
                raise

    print(f"\n  Done. New queries: {new_queries} | Skipped (cached): {skipped}")
    return new_queries
