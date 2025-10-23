"""
Solve ARC Task 007bbfb7 Using TRM Recursion

Task: Tile 3×3 input into 9×9 output with specific placement pattern
Pattern: Place input at positions (0,0), (0,2), (2,0), (2,1) in 3×3 tile grid

Goal: Train TRM on examples 1-4, achieve perfect accuracy on test case

Focus: Validate TRM recursion mechanism learns the tiling transformation

Author: Claude Code
Date: October 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from trm_neural_symbolic import TRMNeuralSymbolicSolver
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np


def load_task_007bbfb7():
    """Load ARC task 007bbfb7."""
    task_file = Path("/Users/faezs/ARC-AGI/data/training/007bbfb7.json")
    if not task_file.exists():
        task_file = Path("/Users/faezs/ARC-AGI-2/data/training/007bbfb7.json")

    with open(task_file) as f:
        return json.load(f)


def prepare_task_data(task_data, num_train=4, device='cpu'):
    """Prepare task data for training.

    Args:
        task_data: ARC task dict
        num_train: Number of training examples (default 4, leaving 1 for validation)
        device: torch device

    Returns:
        train_inputs: [N, H, W] tensors
        train_outputs: [N, H_out, W_out] tensors
        test_input: [H, W] tensor
        test_output: [H_out, W_out] tensor
    """
    train_examples = task_data['train'][:num_train]
    test_example = task_data['test'][0]

    # Convert to tensors
    train_inputs = torch.tensor(
        [ex['input'] for ex in train_examples],
        dtype=torch.long,
        device=device
    )
    train_outputs = torch.tensor(
        [ex['output'] for ex in train_examples],
        dtype=torch.long,
        device=device
    )

    test_input = torch.tensor(
        test_example['input'],
        dtype=torch.long,
        device=device
    )
    test_output = torch.tensor(
        test_example['output'],
        dtype=torch.long,
        device=device
    )

    return train_inputs, train_outputs, test_input, test_output


def train_and_solve(
    num_epochs=200,
    num_cycles=3,
    lr=1e-3,
    device='mps'
):
    """Train TRM on task 007bbfb7 and solve test case.

    Args:
        num_epochs: Number of training epochs
        num_cycles: TRM recursive cycles
        lr: Learning rate
        device: Device
    """
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    device = torch.device(device)

    print("="*80)
    print("SOLVING ARC TASK 007bbfb7 WITH TRM")
    print("="*80)
    print(f"Device: {device}")
    print(f"TRM cycles: {num_cycles}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")

    # Load task
    print("\n=== Loading Task ===")
    task_data = load_task_007bbfb7()
    train_inputs, train_outputs, test_input, test_output = prepare_task_data(
        task_data, num_train=4, device=device
    )

    print(f"Training examples: {train_inputs.shape[0]}")
    print(f"Input shape: {train_inputs.shape[1:]}")
    print(f"Output shape: {train_outputs.shape[1:]}")
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")

    # Create model
    print("\n=== Creating TRM Model ===")
    model = TRMNeuralSymbolicSolver(
        num_colors=10,
        num_cycles=num_cycles,
        device=device
    ).to(device)

    print(f"Template library size: {len(model.templates)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Capture initial weights
    initial_weights = {
        'encoder': model.encoder.layer1.weight.data.clone(),
        'refine_z': model.refine_z.refine[0].weight.data.clone(),
        'refine_y': model.refine_y.refine[0].weight.data.clone(),
    }

    # Training loop
    print(f"\n=== Training for {num_epochs} Epochs ===")

    best_loss = float('inf')
    best_epoch = 0
    losses = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()

        # Forward pass on all training examples
        loss, info = model.compute_loss(
            train_inputs,
            train_outputs
        )

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        losses.append(loss.item())

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={loss.item():.4f}, best={best_loss:.4f} @epoch {best_epoch+1}")

    print(f"\n✅ Training complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f} at epoch {best_epoch+1}")
    print(f"Loss reduction: {losses[0]:.4f} → {losses[-1]:.4f} ({100*(1-losses[-1]/losses[0]):.1f}% improvement)")

    # Analyze weight changes
    print("\n=== Weight Changes ===")
    encoder_diff = (model.encoder.layer1.weight.data - initial_weights['encoder']).abs().sum().item()
    refine_z_diff = (model.refine_z.refine[0].weight.data - initial_weights['refine_z']).abs().sum().item()
    refine_y_diff = (model.refine_y.refine[0].weight.data - initial_weights['refine_y']).abs().sum().item()

    print(f"{'✅' if encoder_diff > 1e-6 else '❌'} Encoder:  {encoder_diff:.2e}")
    print(f"{'✅' if refine_z_diff > 1e-6 else '❌'} Refine_z: {refine_z_diff:.2e}")
    print(f"{'✅' if refine_y_diff > 1e-6 else '❌'} Refine_y: {refine_y_diff:.2e}")

    # Test on validation example (5th training example)
    print("\n=== Validation on Training Example 5 ===")
    val_input = torch.tensor(task_data['train'][4]['input'], dtype=torch.long, device=device)
    val_output_expected = torch.tensor(task_data['train'][4]['output'], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        val_output_pred, val_info = model.forward(
            val_input,
            target_size=val_output_expected.shape,
            hard_select=True
        )
        val_output_pred = val_output_pred.long()

        val_correct = (val_output_pred == val_output_expected).sum().item()
        val_total = val_output_expected.numel()
        val_accuracy = 100.0 * val_correct / val_total

        print(f"Validation accuracy: {val_accuracy:.2f}% ({val_correct}/{val_total} pixels)")
        print(f"Selected template: {str(val_info['selected_templates'][0])[:80]}...")

        if val_accuracy == 100.0:
            print("✅ PERFECT validation prediction!")
        elif val_accuracy > 80.0:
            print("✅ Very good validation prediction")
        elif val_accuracy > 50.0:
            print("⚠️  Partial validation success")
        else:
            print("❌ Poor validation performance")

    # Test on actual test case
    print("\n=== Testing on Test Case ===")
    model.eval()

    with torch.no_grad():
        output_pred, info = model.forward(
            test_input,
            target_size=test_output.shape,
            hard_select=True
        )

        # Convert to long
        output_pred = output_pred.long()

        # Calculate accuracy
        correct_pixels = (output_pred == test_output).sum().item()
        total_pixels = test_output.numel()
        accuracy = 100.0 * correct_pixels / total_pixels

        print(f"\nTest accuracy: {accuracy:.2f}% ({correct_pixels}/{total_pixels} pixels)")
        print(f"Selected template: {str(info['selected_templates'][0])[:80]}...")

        # Show embeddings
        print(f"\nFinal embeddings:")
        print(f"  y (answer): mean={info['y'].mean().item():.4f}, std={info['y'].std().item():.4f}")
        print(f"  z (reasoning): mean={info['z'].mean().item():.4f}, std={info['z'].std().item():.4f}")

        # Compare outputs
        print("\n=== Output Comparison ===")
        print("Predicted output:")
        for row in output_pred.cpu().numpy():
            print("  ", list(row))

        print("\nExpected output:")
        for row in test_output.cpu().numpy():
            print("  ", list(row))

        # Highlight differences
        diff_mask = (output_pred != test_output).cpu().numpy()
        if diff_mask.any():
            print("\nDifferences (1=mismatch):")
            for row in diff_mask.astype(int):
                print("  ", list(row))

        # Final verdict
        print("\n" + "="*80)
        print("FINAL RESULT")
        print("="*80)

        if accuracy == 100.0:
            print("🎉 ✅ PERFECT SOLUTION!")
            print("TRM successfully learned the tiling transformation!")
            return True
        elif accuracy > 90.0:
            print(f"✅ Nearly perfect ({accuracy:.1f}%)")
            print("TRM learned most of the pattern")
            return True
        elif accuracy > 70.0:
            print(f"⚠️  Partial success ({accuracy:.1f}%)")
            print("TRM learned some aspects but not complete")
            return False
        else:
            print(f"❌ Failed to solve ({accuracy:.1f}%)")
            print("TRM did not learn the transformation")
            return False


def analyze_trm_cycles(device='mps', num_cycles=5):
    """Analyze how TRM embeddings evolve across cycles on test input.

    This helps understand what the recursive refinement is doing.
    """
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    device = torch.device(device)

    print("\n" + "="*80)
    print("TRM CYCLE ANALYSIS")
    print("="*80)

    # Load task
    task_data = load_task_007bbfb7()
    train_inputs, train_outputs, test_input, test_output = prepare_task_data(
        task_data, num_train=4, device=device
    )

    # Create and train model briefly
    model = TRMNeuralSymbolicSolver(
        num_colors=10,
        num_cycles=num_cycles,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training briefly ({20} epochs)...")
    for epoch in range(20):
        optimizer.zero_grad()
        loss, _ = model.compute_loss(train_inputs, train_outputs)
        loss.backward()
        optimizer.step()

    # Now analyze cycles
    print(f"\n=== Analyzing {num_cycles} Recursive Cycles ===")

    model.eval()
    with torch.no_grad():
        test_input_batch = test_input.unsqueeze(0)

        # Initial encoding
        y, z = model.encoder(test_input_batch)
        features = model._extract_grid_features(test_input_batch)

        print(f"\nCycle 0 (initial encoding):")
        print(f"  y: mean={y.mean().item():.4f}, std={y.std().item():.4f}, norm={y.norm().item():.4f}")
        print(f"  z: mean={z.mean().item():.4f}, std={z.std().item():.4f}, norm={z.norm().item():.4f}")

        # Run cycles
        for t in range(num_cycles):
            y_prev, z_prev = y.clone(), z.clone()

            # Refine
            z_input = torch.cat([y, z, features], dim=1)
            z = model.refine_z(z_input)

            y_input = torch.cat([z, y, features], dim=1)
            y = model.refine_y(y_input)

            # Changes
            y_change = (y - y_prev).norm().item()
            z_change = (z - z_prev).norm().item()

            print(f"\nCycle {t+1}:")
            print(f"  y: mean={y.mean().item():.4f}, std={y.std().item():.4f}, norm={y.norm().item():.4f}, Δ={y_change:.4f}")
            print(f"  z: mean={z.mean().item():.4f}, std={z.std().item():.4f}, norm={z.norm().item():.4f}, Δ={z_change:.4f}")


if __name__ == "__main__":
    print("="*80)
    print("ARC TASK 007bbfb7 SOLVER")
    print("="*80)
    print("\nTask: Tile 3×3 input into 9×9 output")
    print("Pattern: Symmetric cross placement of input tiles")
    print("\nGoal: Train TRM on 4 examples, solve test case perfectly")

    # Main solve attempt
    success = train_and_solve(
        num_epochs=200,
        num_cycles=3,
        lr=1e-3,
        device='mps'
    )

    # Cycle analysis
    analyze_trm_cycles(device='mps', num_cycles=5)

    print("\n" + "="*80)
    print("DONE")
    print("="*80)
