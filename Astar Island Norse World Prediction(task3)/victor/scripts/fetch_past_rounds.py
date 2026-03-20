"""
fetch_past_rounds.py — Try to recover ground truth from completed rounds.

Costs ZERO queries. Lists all rounds, then attempts GET /analysis for each
completed round. If successful, saves ground truth tensors to disk.
This lets us build a transition matrix WITHOUT waiting for round 4 to end.

Run from victor/ folder:
    python -m scripts.fetch_past_rounds
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.client import AstarClient
import config

ROUND_ID_CURRENT = "8e839974-b13b-407b-a5e7-fc749d877195"


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

    # ── 1. List all rounds ──────────────────────────────────────────
    print("Fetching all rounds...")
    raw_rounds = client._get("/rounds")
    print(f"Found {len(raw_rounds)} round(s):\n")
    for r in raw_rounds:
        print(f"  id={r['id']}  status={r.get('status','?')}")

    completed = [r for r in raw_rounds if r.get("status") == "completed"]
    print(f"\n{len(completed)} completed round(s) to check for ground truth.\n")

    if not completed:
        print("No completed rounds found. Nothing to recover.")
        return

    # ── 2. Try GET /my-rounds for scores ────────────────────────────
    print("Checking /my-rounds for past scores...")
    try:
        my_rounds = client.get_my_rounds()
        print(json.dumps(my_rounds, indent=2))
    except Exception as e:
        print(f"  /my-rounds failed: {e}")

    # ── 3. Try to fetch /analysis for each completed round ──────────
    os.makedirs(config.DATA_DIR + "/round_history", exist_ok=True)

    for r in completed:
        round_id = r["id"]
        print(f"\n{'─'*60}")
        print(f"Round: {round_id}")

        for seed_idx in range(config.NUM_SEEDS):
            save_path = f"{config.DATA_DIR}/round_history/{round_id}_seed{seed_idx}_analysis.json"

            if os.path.exists(save_path):
                print(f"  Seed {seed_idx}: already saved, skipping")
                continue

            try:
                analysis = client.get_analysis(round_id, seed_idx)
                payload = {
                    "round_id": round_id,
                    "seed_index": seed_idx,
                    "score": analysis.score,
                    "ground_truth": analysis.ground_truth,
                    "prediction": analysis.prediction,
                    "initial_grid": analysis.initial_grid,
                }
                with open(save_path, "w") as f:
                    json.dump(payload, f)
                print(f"  Seed {seed_idx}: ✅ saved  (score={analysis.score:.1f})")

            except Exception as e:
                print(f"  Seed {seed_idx}: ❌ {e}")

    # ── 4. Summary ──────────────────────────────────────────────────
    saved = [
        f for f in os.listdir(f"{config.DATA_DIR}/round_history")
        if f.endswith("_analysis.json")
    ]
    print(f"\n{'═'*60}")
    print(f"  Saved {len(saved)} analysis file(s) to data/round_history/")
    if saved:
        print("  Run scripts/build_transition_matrix.py next to extract patterns.")
    else:
        print("  No ground truth recovered — will collect after round 4 ends.")


if __name__ == "__main__":
    main()
