"""
TRM Neural-Symbolic ARC Training

Training loop for TRM-enhanced neural-symbolic solver with:
1. Recursive refinement (T=3 cycles by default)
2. EMA for training stability
3. Tiny 2-layer networks
4. Lower learning rate (1e-4 vs 1e-3)
5. More epochs (200 vs 50) due to smaller networks

Expected improvement over baseline:
- Baseline: 0-2% binary accuracy
- TRM: 10-20% initially → 30-45% after tuning

Author: Claude Code
Date: October 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time
from pathlib import Path

from trm_neural_symbolic import TRMNeuralSymbolicSolver, EMA
from gros_topos_curriculum import (
    load_mini_arc,
    load_arc_agi_1,
    load_arc_agi_2,
)
from arc_solver import ARCGrid


@dataclass
class TRMNeuralSymbolicConfig:
    """Configuration for TRM neural-symbolic training."""
    # Scale settings
    start_scale: int = 0  # Mini-ARC (3×3 to 5×5)
    end_scale: int = 2    # Small-ARC (up to 10×10)

    # Training settings
    epochs: int = 200  # More epochs for tiny networks
    batch_size: int = 4
    lr: float = 1e-4  # Lower LR for stability

    # Model settings
    num_colors: int = 10
    answer_dim: int = 128
    latent_dim: int = 64
    num_cycles: int = 3  # T=3 recursive cycles (as per TRM paper)
    max_composite_depth: int = 2

    # EMA settings
    ema_decay: float = 0.999

    # Gumbel-Softmax temperature annealing
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay: float = 0.95

    # Device
    device: str = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Logging
    log_every: int = 10
    save_every: int = 50
    verbose: bool = True


class TRMNeuralSymbolicTrainer:
    """
    TRM neural-symbolic trainer for ARC tasks.

    Training loop:
    1. Forward pass with recursive refinement (T cycles)
    2. Compute loss (pixel + template entropy)
    3. Backprop (gradients only on last cycle)
    4. Update EMA
    5. Anneal temperature
    """

    def __init__(self, config: TRMNeuralSymbolicConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create model
        self.model = TRMNeuralSymbolicSolver(
            num_colors=config.num_colors,
            answer_dim=config.answer_dim,
            latent_dim=config.latent_dim,
            num_cycles=config.num_cycles,
            device=self.device,
            max_composite_depth=config.max_composite_depth
        ).to(self.device)

        # Optimizer (lower LR for tiny networks)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        # EMA
        self.ema = EMA(self.model, decay=config.ema_decay)

        # Temperature for annealing
        self.current_temperature = config.temperature_start

        # Metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'pixel_accuracy': [],
            'binary_accuracy': [],
            'selected_templates': []
        }

    def load_data(self):
        """Load ARC tasks by scale."""
        print("Loading ARC datasets...")

        # Load all datasets
        mini_arc = load_mini_arc()
        arc_1 = load_arc_agi_1()
        arc_2 = load_arc_agi_2()

        # Combine
        all_tasks = mini_arc + arc_1 + arc_2

        # Split by grid size
        self.tasks_by_scale = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

        for task in all_tasks:
            # Get max grid size
            max_size = 0
            for inp, out in zip(task.input_examples, task.output_examples):
                max_size = max(max_size, inp.shape[0], inp.shape[1])

            # Assign to scale
            if max_size <= 5:
                scale = 0  # Tiny (3-5)
            elif max_size <= 10:
                scale = 1  # Small (6-10)
            elif max_size <= 15:
                scale = 2  # Medium (11-15)
            elif max_size <= 20:
                scale = 3  # Large (16-20)
            elif max_size <= 25:
                scale = 4  # Extra Large (21-25)
            else:
                scale = 5  # Huge (26-30)

            self.tasks_by_scale[scale].append(task)

        # Print stats
        for scale in range(6):
            print(f"  Scale {scale}: {len(self.tasks_by_scale[scale])} tasks")

        # Split train/val (80/20)
        self.train_tasks = {}
        self.val_tasks = {}

        for scale in range(self.config.start_scale, self.config.end_scale + 1):
            tasks = self.tasks_by_scale[scale]
            n_train = int(0.8 * len(tasks))
            self.train_tasks[scale] = tasks[:n_train]
            self.val_tasks[scale] = tasks[n_train:]

            print(f"  Scale {scale}: {len(self.train_tasks[scale])} train, {len(self.val_tasks[scale])} val")

    def train_epoch(self, scale: int, epoch: int) -> Dict[str, float]:
        """Train one epoch on specific scale."""
        self.model.train()

        tasks = self.train_tasks[scale]

        epoch_loss = 0.0
        epoch_pixel_acc = 0.0
        n_examples = 0

        for task in tasks:
            for inp, out in zip(task.input_examples, task.output_examples):
                # Convert to tensors
                inp_tensor = torch.from_numpy(np.array(inp)).float().to(self.device)
                out_tensor = torch.from_numpy(np.array(out)).float().to(self.device)

                # Get target size
                target_size = out_tensor.shape  # (H, W)

                # Forward pass with target size and recursion
                loss, losses = self.model.compute_loss(
                    inp_tensor,
                    out_tensor,
                    target_size=target_size,
                    hard_select=False
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update EMA
                self.ema.update()

                # Metrics
                with torch.no_grad():
                    # Predict with target size
                    pred = self.model.predict(inp_tensor, target_size=target_size)
                    pixel_acc = (pred == out_tensor).float().mean()

                epoch_loss += loss.item()
                epoch_pixel_acc += pixel_acc.item()
                n_examples += 1

        # Anneal temperature
        self.current_temperature = max(
            self.config.temperature_end,
            self.current_temperature * self.config.temperature_decay
        )
        self.model.set_temperature(self.current_temperature)

        return {
            'loss': epoch_loss / n_examples,
            'pixel_accuracy': epoch_pixel_acc / n_examples
        }

    def evaluate(self, scale: int) -> Dict[str, float]:
        """Evaluate on validation set using EMA weights."""
        # Apply EMA weights for evaluation
        self.ema.apply_shadow()
        self.model.eval()

        tasks = self.val_tasks[scale]

        total_pixel_acc = 0.0
        total_binary_acc = 0.0
        n_examples = 0

        selected_templates_counts = {}

        with torch.no_grad():
            for task in tasks:
                for inp, out in zip(task.input_examples, task.output_examples):
                    inp_tensor = torch.from_numpy(np.array(inp)).float().to(self.device)
                    out_tensor = torch.from_numpy(np.array(out)).float().to(self.device)

                    # Get target size
                    target_size = out_tensor.shape  # (H, W)

                    # Predict with target size
                    pred = self.model.predict(inp_tensor, target_size=target_size)

                    # Metrics
                    pixel_acc = (pred == out_tensor).float().mean()
                    binary_acc = 1.0 if torch.all(pred == out_tensor) else 0.0

                    total_pixel_acc += pixel_acc.item()
                    total_binary_acc += binary_acc
                    n_examples += 1

                    # Track selected templates
                    template = str(self.model.get_selected_template(inp_tensor))
                    selected_templates_counts[template] = selected_templates_counts.get(template, 0) + 1

        # Restore training weights
        self.ema.restore()

        # Top 5 templates
        top_templates = sorted(selected_templates_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'pixel_accuracy': total_pixel_acc / n_examples,
            'binary_accuracy': total_binary_acc / n_examples,
            'n_examples': n_examples,
            'top_templates': top_templates
        }

    def train_scale(self, scale: int):
        """Train on specific scale."""
        print(f"\n{'='*70}")
        print(f"Training Scale {scale} (TRM-Enhanced)")
        print(f"{'='*70}")
        print(f"Train tasks: {len(self.train_tasks[scale])}")
        print(f"Val tasks: {len(self.val_tasks[scale])}")
        print(f"Recursive cycles: {self.config.num_cycles}")
        print(f"EMA decay: {self.config.ema_decay}")

        best_val_acc = 0.0

        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(scale, epoch)

            # Evaluate
            if (epoch + 1) % self.config.log_every == 0:
                val_metrics = self.evaluate(scale)

                # Log
                print(f"Epoch {epoch+1}/{self.config.epochs}")
                print(f"  Train loss: {train_metrics['loss']:.4f}")
                print(f"  Train pixel acc: {train_metrics['pixel_accuracy']:.2%}")
                print(f"  Val pixel acc (EMA): {val_metrics['pixel_accuracy']:.2%}")
                print(f"  Val binary acc (EMA): {val_metrics['binary_accuracy']:.2%}")
                print(f"  Temperature: {self.current_temperature:.3f}")

                # Top templates
                if val_metrics['top_templates']:
                    print(f"  Top templates:")
                    for template_str, count in val_metrics['top_templates'][:3]:
                        print(f"    [{count}x] {template_str[:80]}...")

                # Track best
                if val_metrics['binary_accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['binary_accuracy']
                    print(f"  ✓ New best binary accuracy: {best_val_acc:.2%}")

                # Store metrics
                self.metrics['train_loss'].append(train_metrics['loss'])
                self.metrics['val_loss'].append(0.0)  # TODO: compute val loss
                self.metrics['pixel_accuracy'].append(val_metrics['pixel_accuracy'])
                self.metrics['binary_accuracy'].append(val_metrics['binary_accuracy'])

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(scale, epoch)

        # Final evaluation
        print(f"\n{'='*70}")
        print(f"Scale {scale} Final Results (TRM)")
        print(f"{'='*70}")
        final_metrics = self.evaluate(scale)
        print(f"  Pixel accuracy (EMA): {final_metrics['pixel_accuracy']:.2%}")
        print(f"  Binary accuracy (EMA): {final_metrics['binary_accuracy']:.2%}")
        print(f"  Examples: {final_metrics['n_examples']}")

        return final_metrics

    def save_checkpoint(self, scale: int, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints/trm")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"scale{scale}_epoch{epoch+1}.pt"

        # Save model + EMA
        torch.save({
            'epoch': epoch,
            'scale': scale,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_shadow': self.ema.shadow,
            'metrics': self.metrics,
            'config': self.config
        }, checkpoint_path)

        print(f"  Checkpoint saved: {checkpoint_path}")

    def run(self):
        """Run full training pipeline."""
        print(f"\n{'='*70}")
        print(f"TRM NEURAL-SYMBOLIC ARC TRAINING")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Scales: {self.config.start_scale} → {self.config.end_scale}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.lr}")
        print(f"  Answer dim: {self.config.answer_dim}")
        print(f"  Latent dim: {self.config.latent_dim}")
        print(f"  Recursive cycles: {self.config.num_cycles}")
        print(f"  EMA decay: {self.config.ema_decay}")
        print(f"  Device: {self.device}")
        print(f"  Template search space: {len(self.model.templates)}")

        # Load data
        self.load_data()

        # Train each scale
        start_time = time.time()

        for scale in range(self.config.start_scale, self.config.end_scale + 1):
            if scale not in self.train_tasks or len(self.train_tasks[scale]) == 0:
                print(f"\nSkipping scale {scale} (no tasks)")
                continue

            scale_start = time.time()
            scale_metrics = self.train_scale(scale)
            scale_time = time.time() - scale_start

            print(f"\nScale {scale} completed in {scale_time:.2f}s")

        total_time = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"TRM TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.2f}s")
        print(f"\nFinal accuracies (with EMA):")
        for scale in range(self.config.start_scale, self.config.end_scale + 1):
            if scale in self.train_tasks:
                metrics = self.evaluate(scale)
                print(f"  Scale {scale}: {metrics['binary_accuracy']:.2%} binary, {metrics['pixel_accuracy']:.2%} pixel")

        return self.metrics


def main():
    """Run TRM neural-symbolic training."""

    # Detect device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("🚀 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = 'cpu'
        print("⚠️  Using CPU (no GPU detected)")

    # Configuration
    config = TRMNeuralSymbolicConfig(
        start_scale=0,
        end_scale=2,
        epochs=200,  # More epochs for tiny networks
        batch_size=4,
        lr=1e-4,  # Lower LR
        answer_dim=128,
        latent_dim=64,
        num_cycles=3,  # T=3 as per TRM paper
        ema_decay=0.999,
        max_composite_depth=2,
        device=device,
        log_every=10,
        save_every=50,
        verbose=True
    )

    # Create trainer
    trainer = TRMNeuralSymbolicTrainer(config)

    # Run training
    metrics = trainer.run()

    print("\n✅ TRM Training complete!")
    print(f"\nFinal binary accuracy: {metrics['binary_accuracy'][-1]:.2%}")
    print(f"Final pixel accuracy: {metrics['pixel_accuracy'][-1]:.2%}")


if __name__ == "__main__":
    main()
