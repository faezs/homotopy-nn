"""
Test TRM Recursive Refinement with Real ARC-AGI Data

This test validates ONLY the TRM recursion mechanism:
- Encoder: grid → (y, z) embeddings
- Refine_z: Updates z across T cycles
- Refine_y: Updates y across T cycles

EXCLUDED: Formula selection, predicates, Kripke-Joyal semantics

Focus:
1. Load real ARC-AGI task
2. Train on N=1,2,3... examples progressively
3. Inspect gradients at EACH recursive cycle
4. Track z/y embedding evolution
5. Analyze weight changes in TRM components only
6. Test inference with captured intermediate values

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


def load_arc_task(task_id: str = "007bbfb7"):
    """Load a simple ARC task for testing.

    Default task 007bbfb7: Simple color mapping (very learnable)
    """
    arc_data_path = Path("/Users/faezs/ARC-AGI/data/training")
    task_file = arc_data_path / f"{task_id}.json"

    if not task_file.exists():
        # Fallback: try ARC-AGI-2
        arc_data_path = Path("/Users/faezs/ARC-AGI-2/data/training")
        task_file = arc_data_path / f"{task_id}.json"

    with open(task_file) as f:
        task_data = json.load(f)

    return task_data


def pad_to_size(grid, target_H, target_W):
    """Pad grid to target size."""
    H, W = len(grid), len(grid[0])
    padded = [[0] * target_W for _ in range(target_H)]

    for i in range(min(H, target_H)):
        for j in range(min(W, target_W)):
            padded[i][j] = grid[i][j]

    return padded


def prepare_arc_data(task_data, num_examples=None, device='cpu'):
    """Prepare ARC data for training.

    Args:
        task_data: ARC task dict
        num_examples: Number of training examples to use (None = all)
        device: torch device

    Returns:
        input_grids: [N, H, W] tensor
        target_grids: [N, H, W] tensor
        test_input: [H, W] tensor (first test example)
        test_output: [H, W] tensor (first test example)
    """
    train_examples = task_data['train'][:num_examples] if num_examples else task_data['train']
    test_example = task_data['test'][0]

    # Find max dimensions
    max_H = max(
        max(len(ex['input']) for ex in train_examples),
        len(test_example['input'])
    )
    max_W = max(
        max(len(ex['input'][0]) for ex in train_examples),
        len(test_example['input'][0])
    )

    # Pad all grids
    input_grids = []
    target_grids = []

    for ex in train_examples:
        input_padded = pad_to_size(ex['input'], max_H, max_W)
        output_padded = pad_to_size(ex['output'], max_H, max_W)

        input_grids.append(input_padded)
        target_grids.append(output_padded)

    test_input = pad_to_size(test_example['input'], max_H, max_W)
    test_output = pad_to_size(test_example['output'], max_H, max_W)

    # Convert to tensors
    input_tensor = torch.tensor(input_grids, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_grids, dtype=torch.long, device=device)
    test_input_tensor = torch.tensor(test_input, dtype=torch.long, device=device)
    test_output_tensor = torch.tensor(test_output, dtype=torch.long, device=device)

    return input_tensor, target_tensor, test_input_tensor, test_output_tensor


class TRMGradientHook:
    """Hook to capture gradients at each TRM cycle."""

    def __init__(self):
        self.cycle_grads = {}
        self.cycle_values = {}

    def register_hooks(self, model):
        """Register forward and backward hooks on TRM components."""
        self.handles = []

        # Hook encoder
        handle = model.encoder.layer1.register_full_backward_hook(
            self._make_backward_hook('encoder_layer1')
        )
        self.handles.append(handle)

        # Hook refiners
        handle = model.refine_z.refine[0].register_full_backward_hook(
            self._make_backward_hook('refine_z_layer0')
        )
        self.handles.append(handle)

        handle = model.refine_y.refine[0].register_full_backward_hook(
            self._make_backward_hook('refine_y_layer0')
        )
        self.handles.append(handle)

    def _make_backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            # Store gradient magnitude
            if grad_output[0] is not None:
                self.cycle_grads[name] = grad_output[0].abs().mean().item()
        return hook

    def clear_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def test_trm_with_n_examples(task_id="007bbfb7", max_n=3, num_cycles=3):
    """Test TRM recursion with N training examples progressively.

    Args:
        task_id: ARC task ID
        max_n: Maximum number of training examples to use
        num_cycles: Number of TRM recursive cycles
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Task: {task_id}")
    print(f"TRM cycles: {num_cycles}")

    # Load task
    print("\n=== Loading ARC Task ===")
    task_data = load_arc_task(task_id)
    print(f"Training examples available: {len(task_data['train'])}")
    print(f"Test examples: {len(task_data['test'])}")

    # Test with N=1, 2, 3, ... examples
    for N in range(1, max_n + 1):
        print("\n" + "="*80)
        print(f"TRAINING WITH N={N} EXAMPLES")
        print("="*80)

        # Prepare data
        input_grids, target_grids, test_input, test_output = prepare_arc_data(
            task_data, num_examples=N, device=device
        )

        print(f"\nData shapes:")
        print(f"  Input: {input_grids.shape}")
        print(f"  Target: {target_grids.shape}")
        print(f"  Test input: {test_input.shape}")

        # Create fresh model
        model = TRMNeuralSymbolicSolver(
            num_colors=10,
            num_cycles=num_cycles,
            device=device
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Capture initial weights
        initial_weights = {
            'encoder': model.encoder.layer1.weight.data.clone(),
            'refine_z': model.refine_z.refine[0].weight.data.clone(),
            'refine_y': model.refine_y.refine[0].weight.data.clone(),
        }

        # Setup gradient hooks
        hook = TRMGradientHook()
        hook.register_hooks(model)

        # Training loop
        num_iterations = 50
        print(f"\n=== Training for {num_iterations} iterations ===")

        losses = []
        grad_stats = {
            'encoder': [],
            'refine_z': [],
            'refine_y': []
        }

        for iteration in tqdm(range(num_iterations), desc=f"N={N}"):
            optimizer.zero_grad()

            # Forward pass
            loss, info = model.compute_loss(input_grids, target_grids)

            # Backward pass
            loss.backward()

            # Collect gradients
            if model.encoder.layer1.weight.grad is not None:
                grad_stats['encoder'].append(
                    model.encoder.layer1.weight.grad.abs().sum().item()
                )

            if model.refine_z.refine[0].weight.grad is not None:
                grad_stats['refine_z'].append(
                    model.refine_z.refine[0].weight.grad.abs().sum().item()
                )

            if model.refine_y.refine[0].weight.grad is not None:
                grad_stats['refine_y'].append(
                    model.refine_y.refine[0].weight.grad.abs().sum().item()
                )

            # Optimizer step
            optimizer.step()

            losses.append(loss.item())

        # Analyze results
        print("\n=== Training Results ===")
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Change: {losses[-1] - losses[0]:.4f}")

        if losses[-1] < losses[0]:
            print("✅ Loss DECREASED")
        else:
            print("❌ Loss increased or flat")

        print("\n=== Gradient Statistics (Mean over iterations) ===")
        for component, grads in grad_stats.items():
            if len(grads) > 0:
                mean_grad = sum(grads) / len(grads)
                max_grad = max(grads)
                nonzero = sum(1 for g in grads if g > 1e-10)
                status = "✅" if mean_grad > 1e-10 else "❌"
                print(f"{status} {component:15s}: mean={mean_grad:.6e}, max={max_grad:.6e}, nonzero={nonzero}/{len(grads)}")

        print("\n=== Weight Changes ===")
        encoder_diff = (model.encoder.layer1.weight.data - initial_weights['encoder']).abs().sum().item()
        refine_z_diff = (model.refine_z.refine[0].weight.data - initial_weights['refine_z']).abs().sum().item()
        refine_y_diff = (model.refine_y.refine[0].weight.data - initial_weights['refine_y']).abs().sum().item()

        print(f"{'✅' if encoder_diff > 1e-10 else '❌'} Encoder:  {encoder_diff:.6e}")
        print(f"{'✅' if refine_z_diff > 1e-10 else '❌'} Refine_z: {refine_z_diff:.6e}")
        print(f"{'✅' if refine_y_diff > 1e-10 else '❌'} Refine_y: {refine_y_diff:.6e}")

        # Test inference
        print("\n=== Testing Inference ===")
        model.eval()
        with torch.no_grad():
            # Add batch dimension
            test_input_batch = test_input.unsqueeze(0)

            # Forward pass
            output, info = model.forward(test_input_batch, target_size=test_input.shape, hard_select=True)

            # Extract embeddings from info
            if 'y' in info:
                y_final = info['y']
                print(f"Final y embedding shape: {y_final.shape}")
                print(f"Final y embedding stats: mean={y_final.mean().item():.4f}, std={y_final.std().item():.4f}")

            if 'z' in info:
                z_final = info['z']
                print(f"Final z embedding shape: {z_final.shape}")
                print(f"Final z embedding stats: mean={z_final.mean().item():.4f}, std={z_final.std().item():.4f}")

            # Compare output with expected
            output_grid = output[0].long()

            # Calculate accuracy
            correct_pixels = (output_grid == test_output).sum().item()
            total_pixels = test_output.numel()
            accuracy = 100.0 * correct_pixels / total_pixels

            print(f"\nTest accuracy: {accuracy:.2f}% ({correct_pixels}/{total_pixels} pixels)")

            if accuracy == 100.0:
                print("✅ PERFECT TEST PREDICTION!")
            elif accuracy > 50.0:
                print("✅ Partial success (>50% correct)")
            else:
                print("❌ Poor test performance")

        # Cleanup hooks
        hook.clear_hooks()

        print(f"\n{'='*80}")
        print(f"N={N} COMPLETE")
        print(f"{'='*80}\n")


def deep_cycle_analysis(task_id="007bbfb7", num_cycles=5, num_iterations=20):
    """Deep analysis of embedding evolution across TRM cycles.

    This captures z and y values at EACH cycle to see how they evolve.
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("="*80)
    print("DEEP TRM CYCLE ANALYSIS")
    print("="*80)
    print(f"Device: {device}")
    print(f"Task: {task_id}")
    print(f"Cycles: {num_cycles}")

    # Load task
    task_data = load_arc_task(task_id)
    input_grids, target_grids, test_input, test_output = prepare_arc_data(
        task_data, num_examples=2, device=device
    )

    # Create model with MORE cycles for analysis
    model = TRMNeuralSymbolicSolver(
        num_colors=10,
        num_cycles=num_cycles,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n=== Training for {num_iterations} iterations ===")

    # We need to hook into the forward pass to capture intermediate z/y
    # This requires modifying the forward method temporarily
    # For now, we'll just analyze the final values

    for iteration in tqdm(range(num_iterations), desc="Training"):
        optimizer.zero_grad()
        loss, info = model.compute_loss(input_grids, target_grids)
        loss.backward()
        optimizer.step()

    print(f"\nFinal loss: {loss.item():.4f}")

    # Now run inference and capture cycle-by-cycle values
    print("\n=== Running Inference with Cycle Tracking ===")
    model.eval()

    with torch.no_grad():
        test_input_batch = test_input.unsqueeze(0)

        # We need to manually step through cycles
        # Extract from model's forward method
        y, z = model.encoder(test_input_batch)
        features = model._extract_grid_features(test_input_batch)

        print(f"\nCycle 0 (initial):")
        print(f"  y: mean={y.mean().item():.4f}, std={y.std().item():.4f}, norm={y.norm().item():.4f}")
        print(f"  z: mean={z.mean().item():.4f}, std={z.std().item():.4f}, norm={z.norm().item():.4f}")

        # Run refinement cycles
        for t in range(num_cycles):
            z_prev = z.clone()
            y_prev = y.clone()

            # Refine z: concatenate [y, z, features]
            z_input = torch.cat([y, z, features], dim=1)
            z = model.refine_z(z_input)

            # Refine y: concatenate [z, y, features]
            y_input = torch.cat([z, y, features], dim=1)
            y = model.refine_y(y_input)

            # Compute change
            z_change = (z - z_prev).norm().item()
            y_change = (y - y_prev).norm().item()

            print(f"\nCycle {t+1}:")
            print(f"  y: mean={y.mean().item():.4f}, std={y.std().item():.4f}, norm={y.norm().item():.4f}, Δ={y_change:.4f}")
            print(f"  z: mean={z.mean().item():.4f}, std={z.std().item():.4f}, norm={z.norm().item():.4f}, Δ={z_change:.4f}")

    print("\n=== Analysis ===")
    print("Look for:")
    print("  ✅ Embeddings evolve (Δ > 0) across cycles")
    print("  ✅ Statistics stabilize after several cycles")
    print("  ✅ y (answer) converges to useful representation")
    print("  ✅ z (reasoning) shows refinement progress")


if __name__ == "__main__":
    print("="*80)
    print("TRM RECURSIVE REFINEMENT TEST")
    print("="*80)
    print("\nThis test validates the TRM recursion mechanism:")
    print("  - Encoder: grid → (y, z)")
    print("  - Refine_z: z updates across T cycles")
    print("  - Refine_y: y updates across T cycles")
    print("\nEXCLUDED: Formula selection, predicates, KJ semantics")

    # Test 1: Progressive N examples
    print("\n\n" + "="*80)
    print("TEST 1: Progressive N Examples (N=1,2,3)")
    print("="*80)
    test_trm_with_n_examples(task_id="007bbfb7", max_n=3, num_cycles=3)

    # Test 2: Deep cycle analysis
    print("\n\n" + "="*80)
    print("TEST 2: Deep Cycle Analysis")
    print("="*80)
    deep_cycle_analysis(task_id="007bbfb7", num_cycles=5, num_iterations=20)

    print("\n\n" + "="*80)
    print("TRM VALIDATION COMPLETE")
    print("="*80)
    print("\nVerify:")
    print("  ✅ Gradients flow to encoder, refine_z, refine_y")
    print("  ✅ Weights change during training")
    print("  ✅ Loss decreases with more examples")
    print("  ✅ Embeddings evolve across cycles")
    print("  ✅ Test inference produces reasonable output")
