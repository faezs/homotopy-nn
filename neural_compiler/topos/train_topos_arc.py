"""
Complete Training Pipeline for Topos-Theoretic ARC Solver

This script implements the FULL training loop with:
1. Differentiable soft gluing
2. Sheaf axiom losses (composition + identity)
3. Compatibility and coverage losses
4. All gradients flowing end-to-end

This is the culmination of making the topos structure ACTUALLY TRAINABLE!

Author: Claude Code + Human
Date: October 25, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
from tqdm import tqdm

from topos_arc_solver import FewShotARCLearner, SheafSection
from differentiable_gluing import soft_glue_sheaf_sections, compatibility_loss, coverage_loss
from sheaf_axiom_losses import combined_sheaf_axiom_loss
from arc_loader import load_arc_dataset, split_arc_dataset, ARCTask, ARCGrid


################################################################################
# § 1: ARC Data Loading and Preprocessing
################################################################################

def arc_grid_to_tensor(grid: ARCGrid, device='mps') -> torch.Tensor:
    """
    Convert ARCGrid to one-hot encoded torch tensor.

    Args:
        grid: ARCGrid object from arc_loader
        device: torch device

    Returns:
        tensor: [H, W, 10] one-hot encoded grid (10 colors in ARC)
    """
    cells = grid.cells  # numpy array [H, W]
    h, w = cells.shape

    # One-hot encode colors (ARC has 10 colors: 0-9)
    one_hot = np.zeros((h, w, 10), dtype=np.float32)
    for color in range(10):
        one_hot[:, :, color] = (cells == color).astype(np.float32)

    return torch.from_numpy(one_hot).to(device)


def convert_arc_task_to_tensor(arc_task: ARCTask, device='mps') -> Dict:
    """
    Convert ARCTask with ARCGrid objects to dict with torch tensors.

    This adapts the existing arc_loader.py format to our training pipeline.

    Args:
        arc_task: ARCTask from arc_loader.load_arc_dataset()
        device: torch device

    Returns:
        task: Dict with format:
            {
                'train': [(input_tensor, output_tensor), ...],
                'test': (input_tensor, output_tensor)
            }
    """
    return {
        'train': [
            (arc_grid_to_tensor(inp, device), arc_grid_to_tensor(out, device))
            for inp, out in zip(arc_task.train_inputs, arc_task.train_outputs)
        ],
        'test': (
            arc_grid_to_tensor(arc_task.test_inputs[0], device),
            arc_grid_to_tensor(arc_task.test_outputs[0], device)
        )
    }


def pad_to_max_size(grid: torch.Tensor, max_h: int, max_w: int) -> torch.Tensor:
    """
    Pad grid to maximum size (ARC grids have variable sizes).

    Args:
        grid: [H, W, C] grid tensor
        max_h: Maximum height
        max_w: Maximum width

    Returns:
        padded: [max_h, max_w, C] padded grid
    """
    h, w, c = grid.shape

    if h > max_h or w > max_w:
        # Grid too large - crop (shouldn't happen in ARC)
        return grid[:max_h, :max_w, :]

    # Pad with zeros (black background)
    padded = torch.zeros(max_h, max_w, c, device=grid.device)
    padded[:h, :w, :] = grid

    return padded


################################################################################
# § 2: Training Configuration
################################################################################

@dataclass
class TrainingConfig:
    """
    Hyperparameters for topos-theoretic training.
    """
    # Model architecture
    grid_size: Tuple[int, int] = (30, 30)  # Maximum ARC grid size
    feature_dim: int = 64
    stalk_dim: int = 8
    num_patterns: int = 16

    # Training
    num_epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 1  # Few-shot learning - 1 task at a time

    # Loss weights
    task_weight: float = 1.0
    compatibility_weight: float = 0.1
    coverage_weight: float = 0.05
    composition_weight: float = 0.01
    identity_weight: float = 0.01
    orthogonality_weight: float = 0.001
    spectral_weight: float = 0.0  # Disabled on MPS (SVD lacks autograd support)

    # Soft gluing
    gluing_temperature: float = 0.1

    # Logging
    log_dir: str = "runs/topos_arc"
    log_interval: int = 10
    save_interval: int = 50

    # Device
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


################################################################################
# § 3: Training Loop
################################################################################

class ToposARCTrainer:
    """
    Trainer for topos-theoretic ARC solver with all losses.
    """

    def __init__(self, config: TrainingConfig, tasks: List[ARCTask]):
        self.config = config
        self.tasks = tasks
        self.device = torch.device(config.device)

        # Create model
        self.model = FewShotARCLearner(
            grid_size=config.grid_size,
            feature_dim=config.feature_dim,
            stalk_dim=config.stalk_dim,
            num_patterns=config.num_patterns
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # TensorBoard logger
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Metrics
        self.global_step = 0

    def train_step(self, task: Dict) -> Dict[str, float]:
        """
        Single training step on one ARC task.

        Args:
            task: Dict with 'train' and 'test' keys (from convert_arc_task_to_tensor)

        Returns:
            metrics: Dict of loss values
        """
        # Model handles padding internally, just pass variable-sized grids
        train_pairs = [(inp, out) for inp, out in task['train']]

        # Use first test example
        test_input, test_output = task['test']

        # Forward pass through model
        # Model expects: train_pairs without batch dim, test_input with batch dim
        prediction, gluing_result = self.model(
            train_pairs,
            test_input.unsqueeze(0),
            temperature=self.config.gluing_temperature
        )

        # =====================================================================
        # COMPUTE ALL LOSSES
        # =====================================================================

        # 1. Task loss: Prediction accuracy
        # Handle size mismatch: prediction is test_input size, test_output might differ
        test_h, test_w = test_output.shape[0:2]
        pred_h, pred_w = prediction.shape[1:3]

        if (pred_h, pred_w) != (test_h, test_w):
            # Resize prediction to match test_output using interpolation
            prediction_resized = F.interpolate(
                prediction.permute(0, 3, 1, 2),  # [1, 10, pred_h, pred_w]
                size=(test_h, test_w),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # [1, test_h, test_w, 10]
        else:
            prediction_resized = prediction

        task_loss = F.mse_loss(prediction_resized, test_output.unsqueeze(0))

        # 2. Compatibility loss: Encourage sheaf gluing condition
        compat_loss = compatibility_loss(gluing_result.compatibility_score)

        # 3. Coverage loss: Ensure all cells covered
        cover_loss = coverage_loss(gluing_result.coverage)

        # 4. Sheaf axiom losses: Enforce functoriality
        # Use cached restriction maps from forward pass (computed in sheaf NN)
        # These are the EXACT maps used during prediction, ensuring consistency
        if (self.model.sheaf_nn._cached_restriction_maps is not None and
            self.model._cached_edge_index is not None):

            restriction_maps = self.model.sheaf_nn._cached_restriction_maps
            masked_edge_index = self.model._cached_edge_index
            actual_cells = len(self.model._cached_features)

            sheaf_losses = combined_sheaf_axiom_loss(
                restriction_maps,
                masked_edge_index,
                actual_cells,
                composition_weight=self.config.composition_weight,
                identity_weight=self.config.identity_weight,
                orthogonality_weight=self.config.orthogonality_weight,
                spectral_weight=self.config.spectral_weight
            )
        else:
            # No cached maps (shouldn't happen in training mode)
            sheaf_losses = {
                'composition': torch.tensor(0.0, device=prediction.device),
                'identity': torch.tensor(0.0, device=prediction.device),
                'orthogonality': torch.tensor(0.0, device=prediction.device),
                'spectral': torch.tensor(0.0, device=prediction.device)
            }

        # 5. Total loss
        total_loss = (
            self.config.task_weight * task_loss +
            self.config.compatibility_weight * compat_loss +
            self.config.coverage_weight * cover_loss +
            sheaf_losses['total']
        )

        # =====================================================================
        # BACKPROPAGATION
        # =====================================================================

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # =====================================================================
        # METRICS
        # =====================================================================

        # Compute binary accuracy: 1 if prediction matches exactly, 0 otherwise
        # Convert to discrete predictions (argmax over color dimension)
        pred_discrete = prediction_resized.argmax(dim=-1)  # [1, h, w]
        target_discrete = test_output.argmax(dim=-1)  # [h, w]

        # Binary accuracy: 1 if all pixels match, 0 otherwise
        binary_acc = float(torch.all(pred_discrete[0] == target_discrete).item())

        metrics = {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'compatibility_loss': compat_loss.item(),
            'coverage_loss': cover_loss.item(),
            'composition_loss': sheaf_losses['composition'].item(),
            'identity_loss': sheaf_losses['identity'].item(),
            'orthogonality_loss': sheaf_losses['orthogonality'].item(),
            'spectral_loss': sheaf_losses['spectral'].item(),
            'compatibility_score': gluing_result.compatibility_score.item(),
            'binary_accuracy': binary_acc,  # 1 for correct, 0 for wrong
        }

        return metrics

    def train(self):
        """
        Full training loop.
        """
        print("=" * 80)
        print("TRAINING TOPOS-THEORETIC ARC SOLVER")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Training tasks: {len(self.tasks)}")
        print(f"Epochs: {self.config.num_epochs}")
        print()

        for epoch in range(self.config.num_epochs):
            epoch_metrics = {
                'total_loss': 0.0,
                'task_loss': 0.0,
                'compatibility_loss': 0.0,
                'coverage_loss': 0.0,
                'composition_loss': 0.0,
                'compatibility_score': 0.0,
                'binary_accuracy': 0.0,
            }

            # Train on all tasks
            for task in tqdm(self.tasks, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                try:
                    metrics = self.train_step(task)

                    # Accumulate metrics
                    for key in epoch_metrics:
                        if key in metrics:
                            epoch_metrics[key] += metrics[key]

                    # Log to TensorBoard
                    if self.global_step % self.config.log_interval == 0:
                        for key, value in metrics.items():
                            self.writer.add_scalar(f'train/{key}', value, self.global_step)

                    self.global_step += 1

                except Exception as e:
                    print(f"Warning: Failed to train on task: {e}")
                    continue

            # Average metrics over epoch
            num_tasks = len(self.tasks)
            for key in epoch_metrics:
                epoch_metrics[key] /= num_tasks

            # Log epoch summary
            print(f"Epoch {epoch+1}/{self.config.num_epochs}:")
            print(f"  Total Loss:        {epoch_metrics['total_loss']:.4f}")
            print(f"  Task Loss:         {epoch_metrics['task_loss']:.4f}")
            print(f"  Binary Accuracy:   {epoch_metrics['binary_accuracy']:.4f} (1=correct, 0=wrong)")
            print(f"  Compatibility:     {epoch_metrics['compatibility_loss']:.4f} (score: {epoch_metrics['compatibility_score']:.3f})")
            print(f"  Coverage:          {epoch_metrics['coverage_loss']:.4f}")
            print(f"  Composition:       {epoch_metrics['composition_loss']:.4f}")
            print()

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = Path(self.config.log_dir) / f"checkpoint_epoch_{epoch+1}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': epoch_metrics,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        print("=" * 80)
        print("✓ Training complete!")
        print("=" * 80)

        self.writer.close()


################################################################################
# § 4: Evaluation
################################################################################

def evaluate_model(model: FewShotARCLearner, tasks: List[Dict],
                   config: TrainingConfig) -> Dict[str, float]:
    """
    Evaluate trained model on tasks.

    Args:
        model: Trained FewShotARCLearner
        tasks: List of task dicts (already in tensor format)
        config: Training configuration

    Returns:
        metrics: Dict of evaluation metrics
    """
    model.eval()

    total_accuracy = 0.0
    total_binary_acc = 0.0
    total_compatibility = 0.0
    num_tasks = 0

    with torch.no_grad():
        for task in tqdm(tasks, desc="Evaluating"):
            try:
                # Task is already in tensor format
                train_pairs = task['train']
                test_input, test_output = task['test']

                # Predict
                prediction, gluing_result = model(
                    train_pairs,
                    test_input.unsqueeze(0),
                    temperature=config.gluing_temperature
                )

                # Resize prediction if needed
                test_h, test_w = test_output.shape[0:2]
                pred_h, pred_w = prediction.shape[1:3]

                if (pred_h, pred_w) != (test_h, test_w):
                    prediction_resized = F.interpolate(
                        prediction.permute(0, 3, 1, 2),
                        size=(test_h, test_w),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                else:
                    prediction_resized = prediction

                # Compute pixel accuracy
                pred_colors = prediction_resized.argmax(dim=-1)
                true_colors = test_output.argmax(dim=-1)
                pixel_accuracy = (pred_colors[0] == true_colors).float().mean()

                # Compute binary accuracy (1 if all correct, 0 otherwise)
                binary_acc = float(torch.all(pred_colors[0] == true_colors).item())

                total_accuracy += pixel_accuracy.item()
                total_binary_acc += binary_acc
                total_compatibility += gluing_result.compatibility_score.item()
                num_tasks += 1

            except Exception as e:
                print(f"Warning: Failed to evaluate task: {e}")
                continue

    metrics = {
        'pixel_accuracy': total_accuracy / num_tasks if num_tasks > 0 else 0.0,
        'binary_accuracy': total_binary_acc / num_tasks if num_tasks > 0 else 0.0,
        'compatibility': total_compatibility / num_tasks if num_tasks > 0 else 0.0,
        'num_tasks': num_tasks
    }

    return metrics


################################################################################
# § 5: Main Entry Point
################################################################################

def main():
    """
    Main training script using existing arc_loader.py.
    """
    # Configuration
    config = TrainingConfig(
        num_epochs=50,
        learning_rate=0.001,
        compatibility_weight=0.1,
        composition_weight=0.01,
    )

    # Load ARC dataset using existing loader
    arc_data_root = Path(__file__).parent.parent.parent / "ARC-AGI" / "data"

    if not arc_data_root.exists():
        print(f"Error: ARC data directory not found: {arc_data_root}")
        print("Please download ARC-AGI dataset first:")
        print("  git clone https://github.com/fchollet/ARC-AGI.git ~/homotopy-nn/ARC-AGI")
        return

    print("=" * 80)
    print("LOADING ARC DATASET")
    print("=" * 80)
    print()

    # Load all training tasks using existing loader
    all_tasks_dict = load_arc_dataset(
        dataset_dir=str(arc_data_root),
        split="training",
        limit=None  # Load ALL tasks (400)
    )

    if len(all_tasks_dict) == 0:
        print("Error: No tasks loaded")
        return

    # Split into train/val/test using existing function
    train_dict, val_dict, test_dict = split_arc_dataset(
        all_tasks_dict,
        train_ratio=0.7,   # 70% train
        val_ratio=0.15,    # 15% val
        test_ratio=0.15,   # 15% test
        seed=42
    )

    # Convert to tensor format for our training loop
    print("\nConverting ARCGrid objects to torch tensors...")
    train_tasks = [
        convert_arc_task_to_tensor(task, device=config.device)
        for task in tqdm(train_dict.values(), desc="Converting train")
    ]
    val_tasks = [
        convert_arc_task_to_tensor(task, device=config.device)
        for task in tqdm(val_dict.values(), desc="Converting val")
    ]
    test_tasks = [
        convert_arc_task_to_tensor(task, device=config.device)
        for task in tqdm(test_dict.values(), desc="Converting test")
    ]

    print(f"\nDataset split:")
    print(f"  Training tasks:   {len(train_tasks)}")
    print(f"  Validation tasks: {len(val_tasks)}")
    print(f"  Test tasks:       {len(test_tasks)}")
    print()

    # Create trainer
    trainer = ToposARCTrainer(config, train_tasks)

    # Train
    trainer.train()

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(trainer.model, val_tasks, config)

    print("\nValidation Results:")
    print(f"  Pixel Accuracy:  {val_metrics['pixel_accuracy']:.2%}")
    print(f"  Binary Accuracy: {val_metrics['binary_accuracy']:.2%} (exact match)")
    print(f"  Compatibility:   {val_metrics['compatibility']:.3f}")
    print(f"  Tasks:           {val_metrics['num_tasks']}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(trainer.model, test_tasks, config)

    print("\nTest Results:")
    print(f"  Pixel Accuracy:  {test_metrics['pixel_accuracy']:.2%}")
    print(f"  Binary Accuracy: {test_metrics['binary_accuracy']:.2%} (exact match)")
    print(f"  Compatibility:   {test_metrics['compatibility']:.3f}")
    print(f"  Tasks:           {test_metrics['num_tasks']}")


if __name__ == "__main__":
    main()
