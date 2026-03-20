"""
main.py — Full pipeline orchestrator for Astar Island Norse World Prediction.

Two-phase volatile+spread query strategy:
  Phase 1 (25 queries): 5 volatile-targeted tiles per seed (settlements/ports)
  Phase 2 (25 queries): 5 spread tiles per seed (corners + centre for calibration)

Usage:
    python main.py                  # Full two-phase run
    python main.py --no-observe     # Transition matrix only, zero queries
    python main.py --dry-run        # Plan but don't execute or submit
    python main.py --submit-only    # Submit existing saved tensors
"""

import os
import sys
import json
import argparse

from src.api.client import AstarClient
from src.model.initial_analyzer import analyze, build_seed_maps
from src.model.terrain_estimator import estimate_all_seeds, load_transition_matrix
from src.model.round_calibrator import calibrate, save_calibrated_matrix
from src.observation.adaptive_planner import (
    build_phase1_queries, build_phase2_spread_queries, print_phase_summary
)
from src.observation.runner import run as run_observations, load_all_observations
from src.prediction.tensor_builder import build_and_save_all
import config

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


def load_raw_initial() -> dict:
    with open(os.path.join(config.DATA_DIR, "initial_states.json")) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-observe", action="store_true",
                        help="Skip queries, use transition matrix only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan queries but don't execute or submit")
    parser.add_argument("--submit-only", action="store_true",
                        help="Load saved tensors and submit without re-running")
    args = parser.parse_args()

    # ── 1. Auth + Round Detection ─────────────────────────────────────
    section("1. Auth + Round Detection")
    token = load_token()
    client = AstarClient(token=token)
    ROUND_ID = client.get_active_round_id()
    print(f"  Token loaded. Active round: {ROUND_ID}")

    # ── Clean stale files from previous rounds ─────────────────────
    stale_files = [
        os.path.join(config.DATA_DIR, "round_calibrated_matrix.json"),
        os.path.join(config.DATA_DIR, "initial_states.json"),
    ]
    # Also clean old observations
    obs_dir = os.path.join(config.DATA_DIR, "observations")
    if os.path.exists(obs_dir):
        for f in os.listdir(obs_dir):
            stale_files.append(os.path.join(obs_dir, f))

    cleaned = 0
    for path in stale_files:
        if os.path.exists(path):
            os.remove(path)
            cleaned += 1
    if cleaned:
        print(f"  Cleaned {cleaned} stale file(s) from previous round")

    # ── 2. Initial states (always fetch fresh) ─────────────────────
    section("2. Initial States (free)")
    seed_maps = analyze(client, ROUND_ID, verbose=False)
    print(f"  Fetched from API: {len(seed_maps)} seeds")

    for sm in seed_maps:
        print(f"  Seed {sm.seed_index}: {sm.n_static} static | "
              f"{sm.n_dynamic} dynamic | {len(sm.settlements)} settlements")

    # ── Submit-only shortcut ─────────────────────────────────────────
    if args.submit_only:
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
        submit_all(client, ROUND_ID, tensors, args.dry_run)
        return

    # ── 3. Budget check ──────────────────────────────────────────────
    section("3. Budget Check")
    budget = client.get_budget(ROUND_ID)
    print(f"  Queries: {budget.queries_used} used / {budget.queries_max} max "
          f"({budget.queries_remaining} remaining)")
    if budget.is_exhausted:
        print("  ⚠️  Budget exhausted — switching to --no-observe mode")
        args.no_observe = True

    # ── No-observe shortcut ──────────────────────────────────────────
    if args.no_observe:
        section("4. Predictions (transition matrix only — no queries)")
        print("  Skipping simulate() queries.")
        tensors = estimate_all_seeds(seed_maps, observations=[])
        section("5. Validate & Save Tensors")
        build_and_save_all(tensors)
        section("6. Submit")
        if not args.dry_run:
            submit_all(client, ROUND_ID, tensors, dry_run=False)
        return

    # ════════════════════════════════════════════════════════════════
    # TWO-PHASE ADAPTIVE PIPELINE
    # ════════════════════════════════════════════════════════════════

    raw_initial = load_raw_initial()
    historical  = load_transition_matrix(calibrated=False)

    # ── 4. Phase 1 — Volatile targeting (25 queries) ─────────────────
    section("4. Phase 1 — Volatile Targeting (25 queries)")
    phase1_queries = build_phase1_queries(seed_maps)
    print_phase_summary(1, phase1_queries)

    if args.dry_run:
        print("  [DRY RUN] — would execute Phase 1 queries.")
    else:
        run_observations(client, ROUND_ID, phase1_queries)

    # ── 5. Phase 2 — Spread coverage (25 queries) ──────────────────
    section("5. Phase 2 — Spread Coverage (25 queries)")
    phase2_queries = build_phase2_spread_queries(seed_maps)
    print_phase_summary(2, phase2_queries)

    if args.dry_run:
        print("  [DRY RUN] — would execute Phase 2 queries.")
    else:
        run_observations(client, ROUND_ID, phase2_queries)

    # ── 6. Calibration (all 50 observations) ──────────────────────────
    section("6. Calibration (all 50 observations)")
    obs_all = load_all_observations()
    print(f"  All observations loaded: {len(obs_all)} files")

    blended_final = calibrate(raw_initial, obs_all, historical, verbose=True)
    save_calibrated_matrix(blended_final)

    # ── 7. Final predictions ─────────────────────────────────────────
    section("7. Final Predictions")
    tensors = estimate_all_seeds(seed_maps, observations=obs_all)

    # ── 8. Validate & save ──────────────────────────────────────────
    section("8. Validate & Save Tensors")
    build_and_save_all(tensors)

    # ── 9. Submit ────────────────────────────────────────────────────
    section("9. Submit")
    if args.dry_run:
        print("  [DRY RUN] — would submit 5 tensors. Skipping.")
        return

    submit_all(client, ROUND_ID, tensors, dry_run=False)


def submit_all(client: AstarClient, round_id: str, tensors, dry_run: bool = False):
    import time

    scores_path = os.path.join(config.DATA_DIR, "scores.json")
    results = []

    for seed_idx, tensor in enumerate(tensors):
        if dry_run:
            print(f"  [DRY RUN] Would submit seed {seed_idx}")
            continue

        resp = client.submit(round_id, seed_idx, tensor)
        print(f"  Seed {seed_idx}: {'✅' if resp.success else '❌'}  {resp.message}")
        results.append({"seed_index": seed_idx, "success": resp.success})
        time.sleep(0.6)

    if results:
        with open(scores_path, "w") as f:
            json.dump({"round_id": round_id, "submissions": results}, f, indent=2)
        print(f"\n  All 5 seeds submitted. Check leaderboard at app.ainm.no")
        print(f"  Results saved to {scores_path}")


if __name__ == "__main__":
    main()
