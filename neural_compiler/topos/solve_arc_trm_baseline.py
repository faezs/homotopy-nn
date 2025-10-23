"""
Solve ARC Task 007bbfb7 with Pure TRM Baseline

Tests two-level recursive refinement (z_H, z_L) without symbolic formulas.

Author: Claude Code
Date: October 23, 2025
"""

import torch
import json
from pathlib import Path
from trm_baseline import train_trm_baseline


def load_arc_task(task_id: str = "007bbfb7"):
    """Load ARC task."""
    arc_path = Path("/Users/faezs/ARC-AGI/data/training")
    task_file = arc_path / f"{task_id}.json"

    if not task_file.exists():
        arc_path = Path("/Users/faezs/ARC-AGI-2/data/training")
        task_file = arc_path / f"{task_id}.json"

    with open(task_file) as f:
        return json.load(f)


def main():
    print("="*80)
    print("SOLVING ARC 007bbfb7 WITH PURE TRM BASELINE")
    print("="*80)
    print("\nTask: Tile 3×3 → 9×9")
    print("Method: Two-level recursion (z_H, z_L)")
    print("NO symbolic formulas - pure sequence-to-sequence\n")

    # Load task
    task = load_arc_task("007bbfb7")

    # Prepare data (use first 4 training examples)
    train_examples = task['train'][:4]
    test_example = task['test'][0]

    train_inputs = torch.tensor([ex['input'] for ex in train_examples], dtype=torch.long)
    train_outputs = torch.tensor([ex['output'] for ex in train_examples], dtype=torch.long)

    test_input = torch.tensor(test_example['input'], dtype=torch.long)
    test_output = torch.tensor(test_example['output'], dtype=torch.long)

    print(f"Training examples: {len(train_examples)}")
    print(f"Input shape: {train_inputs[0].shape}")
    print(f"Output shape: {train_outputs[0].shape}")

    # Train
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    success = train_trm_baseline(
        train_inputs,
        train_outputs,
        test_input,
        test_output,
        num_epochs=300,
        lr=1e-3,
        H_cycles=5,  # More H cycles for deeper reasoning
        L_cycles=3,  # More L cycles for better output prediction
        device=device
    )

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)

    if success:
        print("✅ TRM baseline successfully solved the task!")
        print("Two-level recursion (z_H, z_L) is working correctly.")
    else:
        print("❌ TRM baseline did not solve the task.")
        print("This may indicate the task requires:")
        print("  1. More training epochs")
        print("  2. Different H/L cycle configuration")
        print("  3. Larger model capacity")
        print("  4. Or task-specific inductive bias (e.g., tiling)")


if __name__ == "__main__":
    main()
