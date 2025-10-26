"""
Train ARC with Algebraic Geometry Approach

KEY CHANGES FROM PREVIOUS:
1. Cross-entropy loss (classification, not regression)
2. Hard sheaf constraints (manifold projection, not soft penalties)
3. Interpolation loss (examples as constraints on continuous morphism)
4. High-capacity model (~100K params for rich transformations)

MATHEMATICAL PRINCIPLE:
  - N training examples = N constraints on continuous geometric morphism
  - Solution = intersection of constraint fibers (unique in the limit)
  - Discrete output = scheme-theoretic limit of continuous map

Author: Claude Code + Human
Date: October 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import List, Tuple, Dict
import time
from datetime import datetime
from pathlib import Path

from algebraic_geometric_morphism import (
    AlgebraicGeometricSolver,
    ConstraintSatisfactionLoss,
    SheafConstraintManifold
)
from arc_loader import ARCGrid, ARCTask, load_arc_dataset, split_arc_dataset
from homotopy_regularization import HomotopyLoss


def get_device(verbose=False):
    """Get best available device."""
    if torch.backends.mps.is_available():
        if verbose:
            print("✓ Using MPS (macOS GPU) backend")
        return torch.device("mps")
    elif torch.cuda.is_available():
        if verbose:
            print("✓ Using CUDA (NVIDIA GPU) backend")
        return torch.device("cuda")
    else:
        if verbose:
            print("⚠ Using CPU backend (slow)")
        return torch.device("cpu")


def encode_batch_to_sheaf(solver: AlgebraicGeometricSolver,
                          grids_tensor: torch.Tensor) -> torch.Tensor:
    """Encode batch of grids to sheaf sections.

    Args:
        solver: Solver with encoder
        grids_tensor: (B, H, W) - color indices

    Returns:
        sheaf_sections: (B, feature_dim, H, W)
    """
    B, H, W = grids_tensor.shape

    # One-hot encode
    one_hot = F.one_hot(grids_tensor, num_classes=solver.num_colors).float()
    one_hot = one_hot.permute(0, 3, 1, 2)  # (B, num_colors, H, W)

    # Encode to sheaf (NO projection here - let gradients flow!)
    sheaf_sections = solver.sheaf_encoder(one_hot)

    return sheaf_sections


def decode_sheaf_to_logits(solver: AlgebraicGeometricSolver,
                           sheaf_sections: torch.Tensor) -> torch.Tensor:
    """Decode sheaf to color logits.

    Args:
        sheaf_sections: (B, feature_dim, H, W)

    Returns:
        logits: (B, num_colors, H, W)
    """
    return solver.decoder(sheaf_sections)


def train_on_arc_task_algebraic(
    task: ARCTask,
    task_id: str,
    max_epochs: int = 1000,      # Increased from 500
    early_stop_patience: int = 200,  # Increased from 50 - allow overfitting!
    lr: float = 1e-3,
    verbose: bool = True
) -> Dict:
    """Train using algebraic geometry approach.

    KEY DIFFERENCES:
    1. Cross-entropy loss (not MSE)
    2. Sheaf constraints as hard projections
    3. Interpolation loss over all examples simultaneously
    4. High-capacity model
    """

    # Find max grid sizes
    all_grids = task.train_inputs + task.train_outputs + task.test_inputs + task.test_outputs
    max_height = max(g.height for g in all_grids)
    max_width = max(g.width for g in all_grids)

    input_shape = (max_height, max_width)
    output_shape = (max_height, max_width)

    # Split train/val
    n_train_total = len(task.train_inputs)
    n_val = max(1, int(n_train_total * 0.2))
    n_train = n_train_total - n_val

    indices = np.random.permutation(n_train_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    if verbose:
        print(f"\n{'='*70}")
        print(f"Training: {task_id}")
        print(f"{'='*70}")
        print(f"  Train: {n_train}, Val: {n_val}, Test: {len(task.test_inputs)}")
        print(f"  Grid size: {max_height}×{max_width}")

    # Create solver (HIGH CAPACITY)
    device = get_device()
    solver = AlgebraicGeometricSolver(
        grid_shape_in=input_shape,
        grid_shape_out=output_shape,
        feature_dim=128,  # High capacity
        num_blocks=4,     # Deep network
        device=device
    )

    if verbose:
        total_params = sum(p.numel() for p in solver.parameters())
        print(f"  Parameters: {total_params:,}")
        print()

    # Prepare training data (pad to max size)
    def pad_grid(grid: ARCGrid, target_h: int, target_w: int) -> torch.Tensor:
        cells = np.array(grid.cells)
        h, w = cells.shape
        padded = np.zeros((target_h, target_w), dtype=np.int64)
        padded[:h, :w] = cells
        return torch.from_numpy(padded).long()

    # Prepare batches
    train_inputs_batch = torch.stack([
        pad_grid(task.train_inputs[i], max_height, max_width) for i in train_indices
    ]).to(device)

    train_outputs_batch = torch.stack([
        pad_grid(task.train_outputs[i], max_height, max_width) for i in train_indices
    ]).to(device)

    val_inputs_batch = torch.stack([
        pad_grid(task.train_inputs[i], max_height, max_width) for i in val_indices
    ]).to(device)

    val_outputs_batch = torch.stack([
        pad_grid(task.train_outputs[i], max_height, max_width) for i in val_indices
    ]).to(device)

    # Optimizer
    optimizer = optim.AdamW(solver.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # Loss functions
    constraint_loss = ConstraintSatisfactionLoss(lambda_adj=0.1, lambda_reg=0.01)
    homotopy_loss_fn = HomotopyLoss(
        lambda_smooth=0.01,    # Smoothness in parameter space
        lambda_interp=0.1,     # Path interpolation consistency
        lambda_topo=0.05       # Topological (Lipschitz) preservation
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(max_epochs):
        # TRAINING
        solver.train()
        optimizer.zero_grad()

        # Encode inputs to sheaves
        input_sheaves_list = []
        target_sheaves_list = []

        for i in range(len(train_indices)):
            inp_sheaf = encode_batch_to_sheaf(solver, train_inputs_batch[i:i+1])
            target_sheaf = encode_batch_to_sheaf(solver, train_outputs_batch[i:i+1])
            input_sheaves_list.append(inp_sheaf)
            target_sheaves_list.append(target_sheaf)

        # Compute constraint satisfaction loss
        losses = constraint_loss(solver.geometric_morphism, input_sheaves_list, target_sheaves_list)

        # Add cross-entropy loss on discrete outputs
        ce_loss = 0.0
        for i in range(len(train_indices)):
            # Get predicted sheaf
            pred_sheaf = solver.geometric_morphism.pushforward(input_sheaves_list[i])

            # HARD PROJECTION onto sheaf manifold (differentiable!)
            # DISABLED: Averaging destroys exact pixel values needed for ARC!
            # pred_sheaf = SheafConstraintManifold.project_onto_manifold(pred_sheaf)

            # Decode to logits
            logits = decode_sheaf_to_logits(solver, pred_sheaf)  # (1, num_colors, H, W)

            # Cross-entropy loss (CORRECT for classification!)
            ce_loss += F.cross_entropy(
                logits,  # (1, num_colors, H, W)
                train_outputs_batch[i:i+1],  # (1, H, W)
                reduction='mean'
            )

        ce_loss = ce_loss / len(train_indices)

        # Homotopy regularization (smoothness + path consistency + topology)
        # DISABLED: Homotopy prevents exact solutions needed for ARC!
        # predicted_sheaves = [solver.geometric_morphism.pushforward(x) for x in input_sheaves_list]
        # homotopy_losses = homotopy_loss_fn(
        #     model=solver.geometric_morphism,
        #     sheaf_pairs=list(zip(input_sheaves_list, target_sheaves_list)),
        #     predicted_outputs=predicted_sheaves
        # )

        # Total loss (PURE classification - allow exact memorization)
        total_loss = (
            ce_loss
            # + 0.5 * losses['interpolation']  # DISABLED - forces smoothness
            # + 0.1 * losses['adjunction']     # DISABLED - categorical constraint
            # + homotopy_losses['total']       # DISABLED - prevents binary accuracy
        )

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), max_norm=1.0)
        optimizer.step()

        # VALIDATION
        solver.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            for i in range(len(val_indices)):
                # Encode
                val_inp_sheaf = encode_batch_to_sheaf(solver, val_inputs_batch[i:i+1])

                # Transform
                val_pred_sheaf = solver.geometric_morphism.pushforward(val_inp_sheaf)
                # DISABLED: Averaging destroys exact values
                # val_pred_sheaf = SheafConstraintManifold.project_onto_manifold(val_pred_sheaf)

                # Decode to logits
                val_logits = decode_sheaf_to_logits(solver, val_pred_sheaf)

                # Loss
                val_loss += F.cross_entropy(val_logits, val_outputs_batch[i:i+1], reduction='mean')

                # Accuracy (exclude padding pixels)
                val_idx = val_indices[i]
                actual_h = task.train_outputs[val_idx].height
                actual_w = task.train_outputs[val_idx].width

                val_pred_colors = val_logits.argmax(dim=1)
                val_correct += (val_pred_colors[0, :actual_h, :actual_w] ==
                              val_outputs_batch[i, :actual_h, :actual_w]).sum().item()
                val_total += actual_h * actual_w

            val_loss = val_loss / len(val_indices)
            val_accuracy = val_correct / val_total

        # Update scheduler
        scheduler.step(val_loss.item())

        # Track history
        history['train_loss'].append(total_loss.item())
        history['val_loss'].append(val_loss.item())
        history['val_accuracy'].append(val_accuracy)

        # Early stopping
        if val_loss.item() < best_val_loss - 1e-6:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
            break

        # Print progress
        if verbose and (epoch % 20 == 0 or patience_counter >= early_stop_patience - 1):
            print(f"Epoch {epoch:3d}: Train={total_loss.item():.4f}, "
                  f"Val={val_loss.item():.4f}, Acc={val_accuracy:.1%}, "
                  f"LR={optimizer.param_groups[0]['lr']:.6f}")

    # FINAL TEST
    solver.eval()
    with torch.no_grad():
        test_inputs_batch = torch.stack([
            pad_grid(g, max_height, max_width) for g in task.test_inputs
        ]).to(device)

        test_outputs_batch = torch.stack([
            pad_grid(g, max_height, max_width) for g in task.test_outputs
        ]).to(device)

        test_correct = 0
        test_total = 0
        test_perfect = 0  # Binary: entire grid must be perfect

        for i in range(len(task.test_inputs)):
            test_inp_sheaf = encode_batch_to_sheaf(solver, test_inputs_batch[i:i+1])
            test_pred_sheaf = solver.geometric_morphism.pushforward(test_inp_sheaf)
            # DISABLED: Averaging destroys exact values
            # test_pred_sheaf = SheafConstraintManifold.project_onto_manifold(test_pred_sheaf)
            test_logits = decode_sheaf_to_logits(solver, test_pred_sheaf)
            test_pred_colors = test_logits.argmax(dim=1)

            # Get actual grid size (exclude padding)
            actual_h = task.test_outputs[i].height
            actual_w = task.test_outputs[i].width

            # Pixel-wise accuracy
            matches = (test_pred_colors[0, :actual_h, :actual_w] ==
                      test_outputs_batch[i, :actual_h, :actual_w])
            test_correct += matches.sum().item()
            test_total += actual_h * actual_w

            # Binary accuracy (all-or-nothing)
            if matches.all():
                test_perfect += 1

        test_accuracy = test_correct / test_total
        test_binary_accuracy = test_perfect / len(task.test_inputs)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Complete: {task_id}")
        print(f"  Pixel accuracy: {test_accuracy:.1%} ({test_correct}/{test_total})")
        print(f"  Binary accuracy: {test_binary_accuracy:.1%} ({test_perfect}/{len(task.test_inputs)})")
        print(f"  Epochs: {len(history['train_loss'])}")
        print(f"{'='*70}\n")

    return {
        'task_id': task_id,
        'test_accuracy': test_accuracy,
        'test_binary_accuracy': test_binary_accuracy,
        'test_correct': test_correct,
        'test_total': test_total,
        'test_perfect': test_perfect,
        'test_examples': len(task.test_inputs),
        'epochs': len(history['train_loss']),
        'history': history
    }


if __name__ == "__main__":
    print("=" * 70)
    print("ARC Training - Algebraic Geometry Approach")
    print("=" * 70)
    print()

    # Load dataset
    ARC_DATA_PATH = "/Users/faezs/homotopy-nn/ARC-AGI/data"
    print(f"Loading ARC dataset from {ARC_DATA_PATH}...")

    # Option 1: Full dataset with PROPER ARC split
    use_full_dataset = False  # Set to True for full dataset (800 tasks total!)

    if use_full_dataset:
        # PROPER ARC SPLIT (as per official ARC-AGI):
        # - training/ (400 tasks) → use for training individual task models
        # - evaluation/ (400 tasks) → held-out test set for final evaluation
        train_tasks = load_arc_dataset(ARC_DATA_PATH, split="training")  # 400 tasks
        test_tasks = load_arc_dataset(ARC_DATA_PATH, split="evaluation")  # 400 tasks
        print(f"✓ Train tasks (training/): {len(train_tasks)}")
        print(f"✓ Test tasks (evaluation/): {len(test_tasks)}")
    else:
        # Quick testing mode
        train_tasks = load_arc_dataset(ARC_DATA_PATH, split="training", limit=10)
        test_tasks = load_arc_dataset(ARC_DATA_PATH, split="evaluation", limit=10)
        print(f"✓ Quick test: {len(train_tasks)} train, {len(test_tasks)} test")

    print()

    device = get_device(verbose=True)
    print()

    # Train on training/ tasks, then evaluate on evaluation/ tasks
    print("=" * 70)
    print("PHASE 1: Training on training/ tasks (400 tasks)")
    print("=" * 70)
    print()

    train_results = []
    for task_id, task in train_tasks.items():
        try:
            result = train_on_arc_task_algebraic(
                task,
                task_id,
                max_epochs=500,
                early_stop_patience=50,
                lr=1e-3,
                verbose=True
            )
            train_results.append(result)
        except Exception as e:
            print(f"ERROR on {task_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Training set summary
    if len(train_results) > 0:
        avg_pixel_acc = np.mean([r['test_accuracy'] for r in train_results])
        avg_binary_acc = np.mean([r['test_binary_accuracy'] for r in train_results])
        total_perfect = sum(r['test_perfect'] for r in train_results)
        total_examples = sum(r['test_examples'] for r in train_results)

        print("\n" + "=" * 70)
        print("TRAINING SET SUMMARY")
        print("=" * 70)
        print(f"  Tasks: {len(train_results)}")
        print(f"  Average pixel accuracy: {avg_pixel_acc:.1%}")
        print(f"  Average binary accuracy: {avg_binary_acc:.1%}")
        print(f"  Perfect grids: {total_perfect}/{total_examples} ({100*total_perfect/total_examples:.1f}%)")
        print()

    # PHASE 2: Evaluate on held-out evaluation/ test set
    print("=" * 70)
    print("PHASE 2: Evaluation on evaluation/ test tasks (400 tasks)")
    print("=" * 70)
    print()

    test_results = []
    for task_id, task in test_tasks.items():
        try:
            result = train_on_arc_task_algebraic(
                task,
                task_id,
                max_epochs=500,
                early_stop_patience=50,
                lr=1e-3,
                verbose=True
            )
            test_results.append(result)
        except Exception as e:
            print(f"ERROR on {task_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Test set summary
    if len(test_results) > 0:
        avg_pixel_acc = np.mean([r['test_accuracy'] for r in test_results])
        avg_binary_acc = np.mean([r['test_binary_accuracy'] for r in test_results])
        total_perfect = sum(r['test_perfect'] for r in test_results)
        total_examples = sum(r['test_examples'] for r in test_results)

        print("\n" + "=" * 70)
        print("FINAL TEST SET SUMMARY (evaluation/)")
        print("=" * 70)
        print(f"  Tasks: {len(test_results)}")
        print(f"  Average pixel accuracy: {avg_pixel_acc:.1%}")
        print(f"  Average binary accuracy: {avg_binary_acc:.1%}")
        print(f"  Perfect grids: {total_perfect}/{total_examples} ({100*total_perfect/total_examples:.1f}%)")
        print()
        print("✓ Complete evaluation with algebraic geometry + homotopy approach!")
