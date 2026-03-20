"""
main.py — Full pipeline orchestrator for Astar Island Norse World Prediction.

Stages:
  1. Auth + find active round
  2. Load initial states (free)
  3. Check budget
  4. Build query plan
  5. Run observations (simulate queries) — or skip with --no-observe
  6. Build predictions (transition matrix + Bayesian updates)
  7. Validate tensors
  8. Submit all 5 seeds

Usage:
    # Full run (spends queries + submits):
    python main.py

    # Submit with transition matrix only — zero queries, get on the board:
    python main.py --no-observe

    # Dry run — plan queries but don't execute or submit:
    python main.py --dry-run

    # Just submit existing saved tensors:
    python main.py --submit-only
"""

import os
import sys
import json
import argparse

from src.api.client import AstarClient
from src.model.initial_analyzer import analyze, build_seed_maps
from src.model.terrain_estimator import estimate_all_seeds
from src.observation.query_planner import build_query_plan, print_plan_summary
from src.observation.runner import run as run_observations, load_all_observations
from src.prediction.tensor_builder import build_and_save_all
import config

ROUND_ID = "8e839974-b13b-407b-a5e7-fc749d877195"


def load_token() -> str:
    token = os.getenv("ASTAR_API_TOKEN", "")
    if not token:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("ASTAR_API_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
    return token


def section(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-observe", action="store_true",
                        help="Skip simulate queries, use transition matrix only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan queries but don't execute or submit")
    parser.add_argument("--submit-only", action="store_true",
                        help="Load saved tensors and submit without re-running")
    args = parser.parse_args()

    # ── 1. Auth ──────────────────────────────────────────────────────
    section("1. Auth")
    token = load_token()
    client = AstarClient(token=token)
    print(f"  Token loaded. Round: {ROUND_ID}")

    # ── 2. Initial states ────────────────────────────────────────────
    section("2. Initial States (free)")
    initial_path = os.path.join(config.DATA_DIR, "initial_states.json")
    if os.path.exists(initial_path):
        with open(initial_path) as f:
            raw = json.load(f)
        seed_maps = build_seed_maps(raw)
        print(f"  Loaded from disk: {len(seed_maps)} seeds")
    else:
        seed_maps = analyze(client, ROUND_ID, verbose=False)
        print(f"  Fetched from API: {len(seed_maps)} seeds")

    for sm in seed_maps:
        print(f"  Seed {sm.seed_index}: {sm.n_static} static | "
              f"{sm.n_dynamic} dynamic | {len(sm.settlements)} settlements")

    if args.submit_only:
        # Load saved tensors and submit directly
        section("Submit Only — Loading saved tensors")
        tensors = []
        for seed_idx in range(config.NUM_SEEDS):
            path = os.path.join(config.PREDICTIONS_DIR, f"seed{seed_idx}_tensor.json")
            if not os.path.exists(path):
                print(f"  ❌ Missing tensor for seed {seed_idx}: {path}")
                sys.exit(1)
            with open(path) as f:
                data = json.load(f)
            tensors.append(data["tensor"])
            print(f"  Loaded seed {seed_idx} tensor from {path}")
        submit_all(client, tensors, args.dry_run)
        return

    # ── 3. Budget check ──────────────────────────────────────────────
    section("3. Budget Check")
    budget = client.get_budget(ROUND_ID)
    print(f"  Queries: {budget.queries_used} used / {budget.queries_max} max "
          f"({budget.queries_remaining} remaining)")
    if budget.is_exhausted:
        print("  ⚠️  Budget exhausted — switching to --no-observe mode")
        args.no_observe = True

    # ── 4. Query plan ────────────────────────────────────────────────
    section("4. Query Plan")
    queries = build_query_plan(seed_maps=seed_maps)
    print_plan_summary(queries)

    # ── 5. Observations ──────────────────────────────────────────────
    section("5. Observations")
    if args.no_observe:
        print("  Skipping simulate() queries — using transition matrix only.")
        print("  Expected score: ~63 (based on offline test).")
    elif args.dry_run:
        print("  [DRY RUN] — would execute the following queries:")
        run_observations(client, ROUND_ID, queries, dry_run=True)
    else:
        run_observations(client, ROUND_ID, queries)

    # ── 6. Build predictions ─────────────────────────────────────────
    section("6. Build Predictions")
    observations = load_all_observations()
    print(f"  Loaded {len(observations)} saved observation files.")
    tensors = estimate_all_seeds(seed_maps, observations=observations)

    # ── 7. Validate + save tensors ───────────────────────────────────
    section("7. Validate & Save Tensors")
    build_and_save_all(tensors)

    # ── 8. Submit ────────────────────────────────────────────────────
    section("8. Submit")
    if args.dry_run:
        print("  [DRY RUN] — would submit 5 tensors. Skipping.")
        return

    submit_all(client, tensors, dry_run=False)


def submit_all(client: AstarClient, tensors, dry_run: bool = False):
    import time

    scores_path = os.path.join(config.DATA_DIR, "scores.json")
    results = []

    for seed_idx, tensor in enumerate(tensors):
        if dry_run:
            print(f"  [DRY RUN] Would submit seed {seed_idx}")
            continue

        resp = client.submit(ROUND_ID, seed_idx, tensor)
        print(f"  Seed {seed_idx}: {'✅' if resp.success else '❌'}  {resp.message}")
        results.append({"seed_index": seed_idx, "success": resp.success})
        time.sleep(0.6)   # 2 req/sec rate limit for submit

    if results:
        with open(scores_path, "w") as f:
            json.dump({"round_id": ROUND_ID, "submissions": results}, f, indent=2)
        print(f"\n  All 5 seeds submitted. Check leaderboard at app.ainm.no")
        print(f"  Results saved to {scores_path}")


if __name__ == "__main__":
    main()
