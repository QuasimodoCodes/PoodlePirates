"""
run_initial_analysis.py — Run Step 5: fetch, save, and display initial states.

Costs ZERO queries. Safe to run multiple times (loads from disk after first run).

Run from victor/ folder:
    python -m scripts.run_initial_analysis
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.client import AstarClient
from src.model.initial_analyzer import analyze
import config

ROUND_ID = "8e839974-b13b-407b-a5e7-fc749d877195"

def main():
    token = os.getenv("ASTAR_API_TOKEN", "")
    if not token:
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("ASTAR_API_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")

    client = AstarClient(token=token)
    seed_maps = analyze(client, ROUND_ID, verbose=True)

    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}")
    for sm in seed_maps:
        print(f"  Seed {sm.seed_index}: {sm.n_static:4d} static "
              f"({sm.pct_static:.0f}%)  |  {sm.n_dynamic:4d} dynamic  "
              f"|  {len(sm.settlements)} initial settlements")

    print(f"\n  Data saved to: {config.DATA_DIR}/initial_states.json")
    print("  Ready for Step 6: Query Planner")

if __name__ == "__main__":
    main()
