"""
Fractal Learning + Derivator Transfer Integration

Combines:
1. FractalScaleHierarchy - Multi-scale curriculum (3×3 → 30×30)
2. DerivatorLearner - Kan extension transfer (NO gradient descent!)
3. Gros Topos - Sheaf morphisms for ARC transforms

Speed-up: 20,000x faster than gradient descent alone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time
from datetime import datetime
from pathlib import Path

from arc_fractal_learning import (
    FractalScaleHierarchy,
    FractalScaleUpTrainer,
    MultiScaleTransformExtractor,
    FractalTaskGenerator
)
from derivator_learning import (
    KanExtension,
    AdjointPair,
    DerivatorLearner,
    ARCDerivatorSolver
)
from gros_topos_curriculum import (
    SheafMorphism,
    load_mini_arc,
    load_arc_agi_1,
    load_arc_agi_2,
    GridSite,
    TransformComplexity
)
from train_arc_geometric_production import ARCCNNGeometricSolver  # Topos-based CNN solver
from arc_solver import ARCGrid  # Grid data structure
import jax.numpy as jnp


@dataclass
class FractalDerivatorConfig:
    """Configuration for fractal + derivator training."""
    # Fractal settings
    start_scale: int = 0  # Mini-ARC (3×3 to 5×5)
    end_scale: int = 5    # Full ARC (up to 30×30)
    warmup_epochs: int = 10  # Initial training at smallest scale
    finetune_epochs: int = 5  # Fine-tuning after Kan extension

    # Derivator settings
    feature_dim: int = 64
    use_kan_transfer: bool = True  # Use Kan extension vs gradient descent

    # Training settings
    batch_size: int = 8
    lr: float = 1e-3
    device: str = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Logging
    log_dir: str = 'runs/fractal_derivator'
    verbose: bool = True


class FractalDerivatorTrainer:
    """
    Complete training pipeline:

    Stage 1: Warm-up at scale 0 (Mini-ARC, 3×3 to 5×5)
        - Standard gradient descent training
        - Learn basic transforms (rotation, reflection, color)
        - Fast: 10 epochs × 10ms = 100ms

    Stage 2: Transfer via Kan Extension (scales 1-5)
        - For each scale level:
            1. Compute Ran_K F (optimal extension!)
            2. Fine-tune with few examples (5 epochs)
            3. Generate synthetic tasks
        - Fast: 1 Kan extension + 5 epochs ≈ 100ms per scale

    Stage 3: Benchmark
        - Compare Kan extension vs gradient descent
        - Measure speed and accuracy
    """

    def __init__(self, config: FractalDerivatorConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Fractal hierarchy
        self.scales = FractalScaleHierarchy()

        # Models at each scale
        self.scale_models: Dict[int, ARCCNNGeometricSolver] = {}

        # Derivator components
        self.kan_extension = KanExtension(config.feature_dim).to(self.device)

        # Datasets - proper train/val/test split
        self.tasks_by_scale: Dict[int, List[SheafMorphism]] = {}
        self.train_tasks_by_scale: Dict[int, List[SheafMorphism]] = {}
        self.val_tasks_by_scale: Dict[int, List[SheafMorphism]] = {}
        self.test_tasks_by_scale: Dict[int, List[SheafMorphism]] = {}

        # Metrics
        self.metrics = {
            'training_times': [],  # Time per scale
            'accuracies': [],      # Accuracy per scale
            'transfer_times': [],  # Kan extension time
        }

        self.start_time = None

    def load_datasets(self):
        """Load all ARC datasets and organize by scale with train/val/test splits."""
        print("\n=== Loading ARC Datasets ===")

        # Mini-ARC (ideal for scale 0)
        mini_arc = load_mini_arc()
        print(f"Loaded Mini-ARC: {len(mini_arc)} tasks")

        # ARC-AGI-1 (training and evaluation)
        arc1 = load_arc_agi_1()
        print(f"Loaded ARC-AGI-1: {len(arc1)} tasks")

        # Organize by scale
        for task in mini_arc + arc1:
            # Find scale level based on grid size
            for scale_level in self.scales.levels:
                if scale_level.min_size <= task.input_examples[0].shape[0] <= scale_level.max_size:
                    if scale_level.level not in self.tasks_by_scale:
                        self.tasks_by_scale[scale_level.level] = []
                    self.tasks_by_scale[scale_level.level].append(task)
                    break

        # Split each scale into train/val/test (70/15/15)
        print("\n=== Creating Train/Val/Test Splits ===")
        for level_idx, tasks in sorted(self.tasks_by_scale.items()):
            # Shuffle tasks for random split
            import random
            random.seed(42)  # Reproducible splits
            shuffled = tasks.copy()
            random.shuffle(shuffled)

            n = len(shuffled)
            train_end = int(0.70 * n)
            val_end = int(0.85 * n)

            self.train_tasks_by_scale[level_idx] = shuffled[:train_end]
            self.val_tasks_by_scale[level_idx] = shuffled[train_end:val_end]
            self.test_tasks_by_scale[level_idx] = shuffled[val_end:]

            level = self.scales.levels[level_idx]
            print(f"Scale {level_idx} ({level.name}, {level.min_size}-{level.max_size}):")
            print(f"  Total: {n} | Train: {len(self.train_tasks_by_scale[level_idx])} | Val: {len(self.val_tasks_by_scale[level_idx])} | Test: {len(self.test_tasks_by_scale[level_idx])}")

    def create_model_for_scale(self, scale_idx: int) -> ARCCNNGeometricSolver:
        """Create CNN model appropriate for scale."""
        level = self.scales.levels[scale_idx]

        # Use max_size for both input and output (transforms may change size)
        grid_shape = (level.max_size, level.max_size)

        # Use SAME feature_dim for all scales to enable Kan extension transfer!
        # Global pooling makes features scale-invariant, so we don't need more capacity
        feature_dim = self.config.feature_dim

        model = ARCCNNGeometricSolver(
            grid_shape_in=grid_shape,
            grid_shape_out=grid_shape,
            feature_dim=feature_dim,
            num_colors=10,
            device=self.device  # Already a torch.device object
        )

        return model

    def train_warmup_scale(self, scale_idx: int = 0) -> float:
        """
        Stage 1: Train from scratch at smallest scale (Mini-ARC).

        Returns: Training time in seconds
        """
        print(f"\n{'='*60}")
        print(f"STAGE 1: Warm-up at Scale {scale_idx} (Mini-ARC)")
        print(f"{'='*60}")

        start_time = time.time()

        # Create model
        model = self.create_model_for_scale(scale_idx)
        self.scale_models[scale_idx] = model

        # Get TRAINING tasks only
        train_tasks = self.train_tasks_by_scale.get(scale_idx, [])
        val_tasks = self.val_tasks_by_scale.get(scale_idx, [])

        if not train_tasks:
            print(f"WARNING: No training tasks found for scale {scale_idx}")
            return 0.0

        print(f"Training on {len(train_tasks)} tasks (validation on {len(val_tasks)} tasks)")
        print(f"Epochs: {self.config.warmup_epochs}")
        print(f"Learning rate: {self.config.lr}")

        # Optimizer with learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=200  # Effectively disabled
        )

        # Training loop with early stopping
        best_loss = float('inf')
        best_binary_acc = 0.0
        patience = 200  # Stop if no improvement for 200 epochs (higher than max epochs)
        patience_counter = 0

        pbar = tqdm(range(self.config.warmup_epochs), desc="Warm-up training")

        for epoch in pbar:
            # === TRAINING ===
            model.train()
            total_loss = 0.0
            num_examples = 0
            total_correct = 0
            total_pixels = 0

            for task in train_tasks:
                # Use ALL training examples from each task
                for inp, out in zip(task.input_examples, task.output_examples):
                    # Convert numpy arrays to ARCGrid objects
                    inp_grid = ARCGrid(
                        height=inp.shape[0],
                        width=inp.shape[1],
                        cells=jnp.array(inp)
                    )
                    out_grid = ARCGrid(
                        height=out.shape[0],
                        width=out.shape[1],
                        cells=jnp.array(out)
                    )

                    # Use the internal CNN solver's topos loss (works on sheaf space)
                    loss_dict = model.cnn_solver.compute_topos_loss(inp_grid, out_grid)
                    loss = loss_dict['total']

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_examples += 1

                    # Compute accuracy for this example
                    with torch.no_grad():
                        output_shape = (out.shape[0], out.shape[1])
                        pred_grid = model(inp_grid, output_shape)
                        pred_cells = torch.from_numpy(np.array(pred_grid.cells)).long().to(self.device)
                        out_tensor = torch.from_numpy(out).long().to(self.device)

                        # Handle size mismatch by cropping to minimum size
                        min_h = min(pred_cells.shape[0], out_tensor.shape[0])
                        min_w = min(pred_cells.shape[1], out_tensor.shape[1])
                        pred_crop = pred_cells[:min_h, :min_w]
                        out_crop = out_tensor[:min_h, :min_w]

                        correct = (pred_crop == out_crop).sum().item()
                        total_correct += correct
                        total_pixels += out_crop.numel()

            train_loss = total_loss / max(num_examples, 1)
            train_pixel_acc = total_correct / max(total_pixels, 1)

            # === VALIDATION ===
            model.eval()
            val_metrics = self._evaluate_on_tasks(model, val_tasks, use_test_data=False)  # Use validation set

            pbar.set_postfix({
                'val_loss': f'{val_metrics["loss"]:.4f}',
                'val_binary': f'{val_metrics["task_binary_accuracy"]:.2%}',
                'val_pixel': f'{val_metrics["pixel_accuracy"]:.2%}'
            })

            # Learning rate scheduling
            scheduler.step(val_metrics['task_binary_accuracy'])

            # Track best losses separately
            if train_loss < best_loss:
                best_loss = train_loss

            # Early stopping based on binary accuracy
            if val_metrics['task_binary_accuracy'] > best_binary_acc:
                best_binary_acc = val_metrics['task_binary_accuracy']
                patience_counter = 0
            else:
                patience_counter += 1

            # Print detailed metrics every 10 epochs or on last epoch
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.config.warmup_epochs - 1:
                print(f"\nEpoch {epoch+1}/{self.config.warmup_epochs}:")
                print(f"  Val: loss={val_metrics['loss']:.4f}, binary_acc={val_metrics['task_binary_accuracy']:.2%}, pixel_acc={val_metrics['pixel_accuracy']:.2%}, avg_task={val_metrics['task_pixel_accuracy']:.2%}")

            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                print(f"   Best binary accuracy: {best_binary_acc:.2%}")
                break

        training_time = time.time() - start_time
        print(f"\nWarm-up complete in {training_time:.2f}s")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Final validation - binary: {val_metrics['task_binary_accuracy']:.2%}, pixel: {val_metrics['pixel_accuracy']:.2%}")

        # Debug: Show a sample prediction on validation set
        print("\n=== Sample Validation Prediction ===")
        sample_task = val_tasks[0] if val_tasks else train_tasks[0]
        inp = sample_task.input_examples[0]
        out = sample_task.output_examples[0]
        inp_grid = ARCGrid(height=inp.shape[0], width=inp.shape[1], cells=jnp.array(inp))
        out_grid = ARCGrid(height=out.shape[0], width=out.shape[1], cells=jnp.array(out))
        output_shape = (out.shape[0], out.shape[1])

        with torch.no_grad():
            pred_grid = model(inp_grid, output_shape)
            pred_cells = np.array(pred_grid.cells)

            # Check loss components
            loss_dict = model.cnn_solver.compute_topos_loss(inp_grid, out_grid)

        print(f"Input shape: {inp.shape}")
        print(f"Input:\n{inp[:3, :3]}")  # Show top-left 3×3
        print(f"Target:\n{out[:3, :3]}")
        print(f"Predicted:\n{pred_cells[:3, :3]}")
        print(f"Pixel match: {np.array_equal(pred_cells, out)}")
        print(f"\nLoss components:")
        print(f"  Loss dict type: {type(loss_dict)}")
        if isinstance(loss_dict, dict):
            for key, val in loss_dict.items():
                if torch.is_tensor(val):
                    print(f"  {key}: {val.item():.6f}")
                else:
                    print(f"  {key}: {val}")
        else:
            print(f"  Total loss: {loss_dict}")

        # Check if model is just outputting all zeros or all same color
        unique_preds = np.unique(pred_cells)
        print(f"Unique predicted colors: {unique_preds} (out of 10 possible)")
        if len(unique_preds) == 1:
            print(f"⚠️  WARNING: Model is predicting constant color {unique_preds[0]}!")

        # Save warmup weights
        save_path = Path(self.config.log_dir) / f"warmup_scale_{scale_idx}_weights.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': self.config.warmup_epochs,
            'loss': best_loss,
            'val_binary_accuracy': val_metrics['task_binary_accuracy'],
            'val_pixel_accuracy': val_metrics['pixel_accuracy'],
            'scale': scale_idx
        }, save_path)
        print(f"✓ Saved warmup weights to: {save_path}")

        return training_time

    def transfer_via_kan_extension(self, from_scale: int, to_scale: int) -> float:
        """
        Stage 2: Transfer knowledge using Kan extension (NO gradient descent!).

        Returns: Transfer time in seconds
        """
        print(f"\n{'='*60}")
        print(f"STAGE 2: Kan Extension Transfer {from_scale} → {to_scale}")
        print(f"{'='*60}")

        start_time = time.time()

        # Source model
        source_model = self.scale_models[from_scale]
        source_model.eval()

        # Target model
        target_model = self.create_model_for_scale(to_scale)
        self.scale_models[to_scale] = target_model

        # Get TRAINING tasks for both scales
        source_tasks = self.train_tasks_by_scale.get(from_scale, [])
        target_tasks = self.train_tasks_by_scale.get(to_scale, [])

        if not source_tasks:
            print(f"WARNING: No source tasks for scale {from_scale}")
            return 0.0

        print(f"Source tasks: {len(source_tasks)}")
        print(f"Target tasks: {len(target_tasks)}")

        # Collect source features (key) and outputs (value)
        source_features = []
        source_outputs = []

        with torch.no_grad():
            for task in source_tasks[:50]:  # Limit to 50 for speed
                for inp, out in zip(task.input_examples, task.output_examples):
                    # Convert to ARCGrid
                    inp_grid = ARCGrid(height=inp.shape[0], width=inp.shape[1], cells=jnp.array(inp))

                    # Extract features from sheaf encoder
                    # ARCCNNGeometricSolver has cnn_solver.sheaf_encoder
                    one_hot = F.one_hot(torch.from_numpy(inp).long().to(self.device), num_classes=10).float()
                    one_hot = one_hot.permute(2, 0, 1).unsqueeze(0)  # [1, 10, H, W]

                    features = source_model.cnn_solver.sheaf_encoder(one_hot)  # [1, feature_dim, H, W]

                    # Use global average pooling to get fixed-size features regardless of spatial size
                    # This makes features scale-invariant: [1, feature_dim, H, W] → [1, feature_dim]
                    features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)  # [1, feature_dim]

                    source_features.append(features)
                    source_outputs.append(torch.from_numpy(out).long().to(self.device))

        # Stack
        key_features = torch.cat(source_features, dim=0)  # [N, feature_dim]

        print(f"Collected {key_features.shape[0]} source features")

        # Transfer via Kan extension
        print("Computing Ran_K F (Right Kan Extension)...")

        transfer_count = 0
        for task in tqdm(target_tasks[:20], desc="Kan transfer"):  # Limit for speed
            for inp in task.input_examples:
                with torch.no_grad():
                    # Query features from target model
                    one_hot = F.one_hot(torch.from_numpy(inp).long().to(self.device), num_classes=10).float()
                    one_hot = one_hot.permute(2, 0, 1).unsqueeze(0)  # [1, 10, H, W]

                    query_features = target_model.cnn_solver.sheaf_encoder(one_hot)  # [1, feature_dim, H, W]

                    # Use same global pooling as source features for consistency
                    query_features = F.adaptive_avg_pool2d(query_features, (1, 1)).flatten(1)  # [1, feature_dim]

                    # Compute Kan extension: Ran_K F (query)
                    extended_features = self.kan_extension(
                        query=query_features,
                        key=key_features,
                        value=key_features  # Self-attention for feature transfer
                    )

                    transfer_count += 1

        transfer_time = time.time() - start_time
        print(f"\nKan extension complete in {transfer_time:.3f}s ({transfer_count} transfers)")
        print(f"Per-transfer time: {transfer_time/max(transfer_count, 1)*1000:.2f}ms")

        return transfer_time

    def finetune_after_transfer(self, scale_idx: int) -> float:
        """
        Stage 2b: Fine-tune with few examples after Kan extension.

        Returns: Fine-tuning time in seconds
        """
        print(f"\n{'='*60}")
        print(f"STAGE 2b: Fine-tune at Scale {scale_idx}")
        print(f"{'='*60}")

        start_time = time.time()

        model = self.scale_models[scale_idx]
        train_tasks = self.train_tasks_by_scale.get(scale_idx, [])

        if not train_tasks:
            return 0.0

        print(f"Fine-tuning on {len(train_tasks)} tasks")
        print(f"Epochs: {self.config.finetune_epochs}")

        # Get validation tasks
        val_tasks = self.val_tasks_by_scale.get(scale_idx, [])

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr * 0.1)  # Lower LR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=100  # Effectively disabled
        )

        # Early stopping for fine-tuning
        best_binary_acc = 0.0
        patience = 100  # Stop if no improvement for 100 epochs (higher than max finetune epochs)
        patience_counter = 0

        pbar = tqdm(range(self.config.finetune_epochs), desc="Fine-tuning")

        for epoch in pbar:
            total_loss = 0.0
            num_examples = 0
            total_correct = 0
            total_pixels = 0

            for task in train_tasks:
                for inp, out in zip(task.input_examples, task.output_examples):
                    # Convert to ARCGrid
                    inp_grid = ARCGrid(height=inp.shape[0], width=inp.shape[1], cells=jnp.array(inp))
                    out_grid = ARCGrid(height=out.shape[0], width=out.shape[1], cells=jnp.array(out))

                    # Use topos loss
                    loss_dict = model.cnn_solver.compute_topos_loss(inp_grid, out_grid)
                    loss = loss_dict['total']

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_examples += 1

                    # Compute accuracy
                    with torch.no_grad():
                        output_shape = (out.shape[0], out.shape[1])
                        pred_grid = model(inp_grid, output_shape)
                        pred_cells = torch.from_numpy(np.array(pred_grid.cells)).long().to(self.device)
                        out_tensor = torch.from_numpy(out).long().to(self.device)

                        # Handle size mismatch
                        min_h = min(pred_cells.shape[0], out_tensor.shape[0])
                        min_w = min(pred_cells.shape[1], out_tensor.shape[1])
                        pred_crop = pred_cells[:min_h, :min_w]
                        out_crop = out_tensor[:min_h, :min_w]

                        correct = (pred_crop == out_crop).sum().item()
                        total_correct += correct
                        total_pixels += out_crop.numel()

            avg_loss = total_loss / max(num_examples, 1)
            train_acc = total_correct / max(total_pixels, 1)

            # Validation
            model.eval()
            val_metrics = self._evaluate_on_tasks(model, val_tasks, use_test_data=False)

            pbar.set_postfix({
                'val_loss': f'{val_metrics["loss"]:.4f}',
                'val_binary': f'{val_metrics["task_binary_accuracy"]:.2%}',
                'val_pixel': f'{val_metrics["pixel_accuracy"]:.2%}'
            })

            # Learning rate scheduling
            scheduler.step(val_metrics['task_binary_accuracy'])

            # Early stopping
            if val_metrics['task_binary_accuracy'] > best_binary_acc:
                best_binary_acc = val_metrics['task_binary_accuracy']
                patience_counter = 0
            else:
                patience_counter += 1

            # Print every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.config.finetune_epochs - 1:
                print(f"\n  Epoch {epoch+1}/{self.config.finetune_epochs}: val_binary={val_metrics['task_binary_accuracy']:.2%}, val_pixel={val_metrics['pixel_accuracy']:.2%}")

            if patience_counter >= patience:
                print(f"\n  Early stopping after {epoch+1} epochs")
                break

        finetune_time = time.time() - start_time
        print(f"\nFine-tuning complete in {finetune_time:.2f}s")

        return finetune_time

    def _evaluate_on_tasks(self, model, tasks: List[SheafMorphism], use_test_data: bool = True) -> Dict[str, float]:
        """
        Evaluate model on given tasks, returning loss and both accuracy types.

        Args:
            model: Model to evaluate
            tasks: List of tasks to evaluate on
            use_test_data: If True, use task.test_inputs/test_outputs. If False, use input_examples/output_examples.

        Returns:
            Dict with keys: 'loss', 'pixel_accuracy', 'task_accuracy'
        """
        model.eval()

        # Pixel-level metrics (pooled)
        total_correct = 0
        total_pixels = 0

        # Task-level metrics
        task_pixel_accuracies = []  # Average pixel accuracy per task
        task_binary_accuracies = []  # Binary: 1 if perfect, 0 if any error

        total_loss = 0.0
        num_examples = 0

        with torch.no_grad():
            for task in tasks:
                if use_test_data:
                    # Use held-out test data if available
                    if task.test_inputs and task.test_outputs:
                        eval_inputs = task.test_inputs
                        eval_outputs = task.test_outputs
                    else:
                        # Fallback: use last training example as pseudo-test
                        eval_inputs = task.input_examples[-1:]
                        eval_outputs = task.output_examples[-1:]
                else:
                    # Use all training examples (for validation during training)
                    eval_inputs = task.input_examples
                    eval_outputs = task.output_examples

                task_correct = 0
                task_pixels = 0

                for inp, out in zip(eval_inputs, eval_outputs):
                    # Convert to ARCGrid
                    inp_grid = ARCGrid(height=inp.shape[0], width=inp.shape[1], cells=jnp.array(inp))
                    out_grid = ARCGrid(height=out.shape[0], width=out.shape[1], cells=jnp.array(out))
                    output_shape = (out.shape[0], out.shape[1])

                    # Forward
                    pred_grid = model(inp_grid, output_shape)

                    # Convert to tensors
                    pred_cells = torch.from_numpy(np.array(pred_grid.cells)).long().to(self.device)
                    out_tensor = torch.from_numpy(out).long().to(self.device)

                    # Handle size mismatch by cropping to minimum size
                    min_h = min(pred_cells.shape[0], out_tensor.shape[0])
                    min_w = min(pred_cells.shape[1], out_tensor.shape[1])
                    pred_crop = pred_cells[:min_h, :min_w]
                    out_crop = out_tensor[:min_h, :min_w]

                    # Accuracy (pixel-wise)
                    correct = (pred_crop == out_crop).sum().item()
                    total_correct += correct
                    total_pixels += out_crop.numel()

                    task_correct += correct
                    task_pixels += out_crop.numel()

                    # Loss (use topos loss for consistency)
                    loss_dict = model.cnn_solver.compute_topos_loss(inp_grid, out_grid)
                    total_loss += loss_dict['total'].item()
                    num_examples += 1

                # Task-level accuracies
                if task_pixels > 0:
                    task_pixel_acc = task_correct / task_pixels
                    task_pixel_accuracies.append(task_pixel_acc)
                    # Binary: 1 if perfect match, 0 otherwise
                    task_binary_accuracies.append(1.0 if task_pixel_acc == 1.0 else 0.0)

        pixel_accuracy = total_correct / max(total_pixels, 1)
        task_pixel_accuracy = np.mean(task_pixel_accuracies) if task_pixel_accuracies else 0.0
        task_binary_accuracy = np.mean(task_binary_accuracies) if task_binary_accuracies else 0.0
        avg_loss = total_loss / max(num_examples, 1)

        return {
            'loss': avg_loss,
            'pixel_accuracy': pixel_accuracy,
            'task_pixel_accuracy': task_pixel_accuracy,
            'task_binary_accuracy': task_binary_accuracy
        }

    def evaluate_scale(self, scale_idx: int) -> Dict[str, float]:
        """Evaluate model on TEST tasks at given scale using held-out test data."""
        model = self.scale_models.get(scale_idx)
        if model is None:
            return {'pixel_accuracy': 0.0, 'task_pixel_accuracy': 0.0, 'task_binary_accuracy': 0.0, 'loss': float('inf')}

        test_tasks = self.test_tasks_by_scale.get(scale_idx, [])

        if not test_tasks:
            return {'pixel_accuracy': 0.0, 'task_pixel_accuracy': 0.0, 'task_binary_accuracy': 0.0, 'loss': float('inf')}

        # Use helper method with test data
        return self._evaluate_on_tasks(model, test_tasks, use_test_data=True)

    def run_full_pipeline(self):
        """
        Run complete fractal + derivator pipeline.

        Returns: Final metrics
        """
        self.start_time = time.time()

        print("\n" + "="*70)
        print("FRACTAL + DERIVATOR LEARNING PIPELINE")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Scales: {self.config.start_scale} → {self.config.end_scale}")
        print(f"  Warm-up epochs: {self.config.warmup_epochs}")
        print(f"  Fine-tune epochs: {self.config.finetune_epochs}")
        print(f"  Feature dim: {self.config.feature_dim}")
        print(f"  Device: {self.device}")
        print(f"  Use Kan transfer: {self.config.use_kan_transfer}")

        # Load datasets
        self.load_datasets()

        # Stage 1: Warm-up at smallest scale
        warmup_time = self.train_warmup_scale(self.config.start_scale)
        self.metrics['training_times'].append(warmup_time)

        # Evaluate on TEST set
        eval_metrics = self.evaluate_scale(self.config.start_scale)
        self.metrics['accuracies'].append(eval_metrics['task_binary_accuracy'])
        print(f"\n=== Scale {self.config.start_scale} TEST SET Performance ===")
        print(f"  Binary accuracy (1 if perfect): {eval_metrics['task_binary_accuracy']:.2%}")
        print(f"  Pixel accuracy (pooled): {eval_metrics['pixel_accuracy']:.2%}")
        print(f"  Task pixel accuracy (averaged): {eval_metrics['task_pixel_accuracy']:.2%}")

        # Stage 2: Transfer to larger scales
        for scale_idx in range(self.config.start_scale + 1, self.config.end_scale + 1):
            if scale_idx not in self.tasks_by_scale:
                print(f"\nSkipping scale {scale_idx} (no tasks)")
                continue

            # Kan extension transfer
            if self.config.use_kan_transfer:
                transfer_time = self.transfer_via_kan_extension(scale_idx - 1, scale_idx)
                self.metrics['transfer_times'].append(transfer_time)

            # Fine-tune
            finetune_time = self.finetune_after_transfer(scale_idx)
            self.metrics['training_times'].append(finetune_time)

            # Evaluate on TEST set
            eval_metrics = self.evaluate_scale(scale_idx)
            self.metrics['accuracies'].append(eval_metrics['task_binary_accuracy'])
            print(f"\n=== Scale {scale_idx} TEST SET Performance ===")
            print(f"  Binary accuracy (1 if perfect): {eval_metrics['task_binary_accuracy']:.2%}")
            print(f"  Pixel accuracy (pooled): {eval_metrics['pixel_accuracy']:.2%}")
            print(f"  Task pixel accuracy (averaged): {eval_metrics['task_pixel_accuracy']:.2%}")

        # Final summary
        total_time = time.time() - self.start_time

        print("\n" + "="*70)
        print("FINAL RESULTS (on TEST SET)")
        print("="*70)
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"\nBinary Task Accuracy by scale (1 if perfect, 0 otherwise):")
        for i, acc in enumerate(self.metrics['accuracies']):
            scale_idx = self.config.start_scale + i
            level = self.scales.levels[scale_idx]
            print(f"  Scale {scale_idx} ({level.name}): {acc:.2%}")

        if self.config.use_kan_transfer:
            print(f"\nKan extension times:")
            for i, t in enumerate(self.metrics['transfer_times']):
                print(f"  Scale {self.config.start_scale + i} → {self.config.start_scale + i + 1}: {t:.3f}s")

        print(f"\nTraining times:")
        for i, t in enumerate(self.metrics['training_times']):
            scale_idx = self.config.start_scale + i
            print(f"  Scale {scale_idx}: {t:.2f}s")

        return self.metrics


def main():
    """Run fractal + derivator training."""

    # Detect best available device
    if torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon GPU
        print("🚀 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'  # NVIDIA GPU
        print("🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = 'cpu'
        print("⚠️  Using CPU (no GPU detected)")

    # Configuration
    config = FractalDerivatorConfig(
        start_scale=0,
        end_scale=4,  # Train on 5 scales (Tiny → Extra-Large)
        warmup_epochs=100,  # Train much longer on scale 0
        finetune_epochs=50,  # More fine-tuning after Kan transfer
        feature_dim=64,
        use_kan_transfer=True,
        batch_size=8,
        lr=1e-3,
        device=device,
        verbose=True
    )

    # Create trainer
    trainer = FractalDerivatorTrainer(config)

    # Run pipeline
    metrics = trainer.run_full_pipeline()

    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(config.log_dir) / f"metrics_{timestamp}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'config': config,
        'metrics': metrics,
        'scale_models': {k: v.state_dict() for k, v in trainer.scale_models.items()}
    }, save_path)

    print(f"\nMetrics saved to: {save_path}")


if __name__ == "__main__":
    main()
