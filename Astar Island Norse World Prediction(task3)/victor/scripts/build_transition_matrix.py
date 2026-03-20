"""
build_transition_matrix.py — Learn what each terrain type becomes after 50 years.

Reads all saved analysis files from data/round_history/.
For each cell: looks up its initial terrain code, reads the ground truth
probability distribution, and averages across all cells with the same code.

Output: data/transition_matrix.json
  transition_matrix[initial_code] = [P(class0), P(class1), ..., P(class5)]

This replaces Layer C spatial rules in terrain_estimator.py with real data.

Run from victor/ folder:
    python -m scripts.build_transition_matrix
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

CODE_NAMES = {
    0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin",
    4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains"
}
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def main():
    history_dir = os.path.join(config.DATA_DIR, "round_history")
    files = [f for f in os.listdir(history_dir) if f.endswith("_analysis.json")]

    if not files:
        print("No analysis files found. Run fetch_past_rounds.py first.")
        return

    print(f"Found {len(files)} analysis file(s) in {history_dir}\n")

    # accumulator[initial_code] = list of 6-element ground truth distributions
    accumulator = {}

    total_cells = 0
    skipped_no_gt = 0

    for fname in sorted(files):
        fpath = os.path.join(history_dir, fname)
        with open(fpath) as f:
            data = json.load(f)

        ground_truth = data.get("ground_truth")
        initial_grid = data.get("initial_grid")

        if not ground_truth or not initial_grid:
            print(f"  {fname}: missing ground_truth or initial_grid — skipping")
            skipped_no_gt += 1
            continue

        H = len(initial_grid)
        W = len(initial_grid[0])
        print(f"  {fname}: {H}×{W} map", end="")

        cells_this_file = 0
        for y in range(H):
            for x in range(W):
                code = initial_grid[y][x]
                gt = ground_truth[y][x]   # list of 6 floats

                if len(gt) != config.NUM_TERRAIN_CLASSES:
                    continue

                if code not in accumulator:
                    accumulator[code] = []
                accumulator[code].append(gt)
                cells_this_file += 1

        total_cells += cells_this_file
        print(f"  -> {cells_this_file} cells")

    if skipped_no_gt == len(files):
        print("\nAll files missing ground_truth. The API may not include ground truth")
        print("when no submission was made for that round.")
        print("Strategy: submit for round 4, then collect ground truth after it ends.")
        return

    # ── Build transition matrix ──────────────────────────────────────
    print(f"\nTotal cells processed: {total_cells}")
    print(f"\n{'═'*70}")
    print("  TRANSITION MATRIX — initial terrain code -> final class distribution")
    print(f"  (averaged over {len(files)} files, {total_cells} cells)")
    print(f"{'═'*70}")
    print(f"  {'Code':<6} {'Terrain':<14} {'n':>6}  {'Empty':>8} {'Settl':>8} {'Port':>8} {'Ruin':>8} {'Forest':>8} {'Mtn':>8}")
    print(f"  {'─'*80}")

    transition_matrix = {}

    for code in sorted(accumulator.keys()):
        samples = accumulator[code]
        n = len(samples)
        avg = [sum(s[i] for s in samples) / n for i in range(config.NUM_TERRAIN_CLASSES)]
        transition_matrix[str(code)] = avg

        name = CODE_NAMES.get(code, f"?{code}")
        vals = "  ".join(f"{v:7.3f}" for v in avg)
        print(f"  {code:<6} {name:<14} {n:>6}  {vals}")

    # ── Save ────────────────────────────────────────────────────────
    save_path = os.path.join(config.DATA_DIR, "transition_matrix.json")
    with open(save_path, "w") as f:
        json.dump({
            "source_files": len(files),
            "total_cells": total_cells,
            "class_names": CLASS_NAMES,
            "code_names": {str(k): v for k, v in CODE_NAMES.items()},
            "transition_matrix": transition_matrix,
        }, f, indent=2)

    print(f"\n  Saved to {save_path}")
    print("\n  KEY INSIGHTS:")

    # Highlight interesting patterns
    for code, avg in [(int(k), v) for k, v in transition_matrix.items()]:
        name = CODE_NAMES.get(code, f"?{code}")
        top_class = max(range(6), key=lambda i: avg[i])
        top_pct = avg[top_class] * 100
        second_class = sorted(range(6), key=lambda i: avg[i], reverse=True)[1]
        second_pct = avg[second_class] * 100
        print(f"    code {code:2d} ({name:<12}): "
              f"mostly -> {CLASS_NAMES[top_class]} ({top_pct:.1f}%)  "
              f"then -> {CLASS_NAMES[second_class]} ({second_pct:.1f}%)")

    print("\n  Next: update terrain_estimator.py to use this matrix for Layer C.")


if __name__ == "__main__":
    main()
