"""
tensor_builder.py — Step 10: Validate, save, and prepare tensors for submission.

Takes raw H×W×6 tensors from terrain_estimator and:
  1. Asserts every cell sums to 1.0 ± 1e-4
  2. Asserts no zeros (floor already applied by estimator)
  3. Saves to data/predictions/seed{n}_tensor.json
  4. Returns as nested list ready for POST /submit
"""

import json
import os
from typing import List

import config


def validate_tensor(tensor: List[List[List[float]]], seed_index: int) -> None:
    """Hard checks before submission. Raises if anything is wrong."""
    H = len(tensor)
    W = len(tensor[0]) if H > 0 else 0
    errors = []

    for y in range(H):
        for x in range(W):
            cell = tensor[y][x]

            if len(cell) != config.NUM_TERRAIN_CLASSES:
                errors.append(f"  ({y},{x}): wrong length {len(cell)}, expected {config.NUM_TERRAIN_CLASSES}")
                continue

            total = sum(cell)
            if abs(total - 1.0) > 1e-4:
                errors.append(f"  ({y},{x}): sum={total:.6f}, not 1.0")

            if any(v < 0 for v in cell):
                errors.append(f"  ({y},{x}): negative value {min(cell):.6f}")

            if any(v == 0.0 for v in cell):
                errors.append(f"  ({y},{x}): zero probability — KL divergence will be infinite")

    if errors:
        print(f"\n  ❌ Seed {seed_index} tensor validation FAILED:")
        for e in errors[:10]:
            print(e)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        raise ValueError(f"Tensor for seed {seed_index} failed validation with {len(errors)} errors")


def save_tensor(tensor: List[List[List[float]]], seed_index: int) -> str:
    """Save tensor to data/predictions/seed{n}_tensor.json. Returns path."""
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    path = os.path.join(config.PREDICTIONS_DIR, f"seed{seed_index}_tensor.json")
    with open(path, "w") as f:
        json.dump({
            "seed_index": seed_index,
            "shape": [len(tensor), len(tensor[0]), config.NUM_TERRAIN_CLASSES],
            "tensor": tensor,
        }, f)
    return path


def build_and_save_all(
    tensors: List[List[List[List[float]]]],
) -> List[List[List[List[float]]]]:
    """
    Validate and save all seed tensors. Returns them ready for submission.
    Raises immediately if any tensor fails validation.
    """
    print(f"\n  Validating and saving {len(tensors)} tensor(s)...")

    for seed_idx, tensor in enumerate(tensors):
        H = len(tensor)
        W = len(tensor[0]) if H > 0 else 0

        validate_tensor(tensor, seed_idx)
        path = save_tensor(tensor, seed_idx)

        # Spot-check: sample min/max probability
        all_vals = [v for row in tensor for cell in row for v in cell]
        print(f"  Seed {seed_idx}: ✅  shape={H}×{W}×6  "
              f"min={min(all_vals):.5f}  max={max(all_vals):.5f}  "
              f"→ saved to {path}")

    return tensors
