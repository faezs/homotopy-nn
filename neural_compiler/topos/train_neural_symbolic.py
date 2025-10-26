"""
Neural-Symbolic ARC Training with Fractal Learning

Integrates:
1. Neural-symbolic solver (formula templates + Kripke-Joyal)
2. Fractal scale hierarchy (3×3 → 30×30)
3. Geometric morphism training

Key innovation: Instead of black-box CNN, we learn interpretable formulas
that express ARC transformations in the internal language of the topos.

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

from neural_symbolic_arc import NeuralSymbolicARCSolver
from gros_topos_curriculum import (
    load_mini_arc,
    load_arc_agi_1,
    load_arc_agi_2,
)
from arc_solver import ARCGrid
import jax.numpy as jnp


@dataclass
class NeuralSymbolicConfig:
    """Configuration for neural-symbolic training."""
    # Scale settings
    start_scale: int = 0  # Mini-ARC (3×3 to 5×5)
    end_scale: int = 2    # Small-ARC (up to 10×10)

    # Training settings
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-3

    # Model settings
    num_colors: int = 10
    feature_dim: int = 64
    max_composite_depth: int = 2

    # Gumbel-Softmax temperature annealing
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay: float = 0.95

    # Device
    device: str = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Logging
    log_every: int = 10
    verbose: bool = True


class NeuralSymbolicTrainer:
    """
    Neural-symbolic trainer for ARC tasks.

    Training loop:
    1. Select formula template using Gumbel-Softmax
    2. Evaluate formulas using Kripke-Joyal semantics
    3. Apply transformation
    4. Compute loss (pixel + template entropy)
    5. Backprop through differentiable logic
    """

    def __init__(self, config: NeuralSymbolicConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create model
        self.model = NeuralSymbolicARCSolver(
            num_colors=config.num_colors,
            feature_dim=config.feature_dim,
            device=self.device,
            max_composite_depth=config.max_composite_depth
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

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

                # Forward pass with target size
                loss, losses = self.model.compute_loss(inp_tensor, out_tensor, target_size=target_size, hard_select=False)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
        """Evaluate on validation set."""
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
        print(f"Training Scale {scale}")
        print(f"{'='*70}")
        print(f"Train tasks: {len(self.train_tasks[scale])}")
        print(f"Val tasks: {len(self.val_tasks[scale])}")

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
                print(f"  Val pixel acc: {val_metrics['pixel_accuracy']:.2%}")
                print(f"  Val binary acc: {val_metrics['binary_accuracy']:.2%}")
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

        # Final evaluation
        print(f"\n{'='*70}")
        print(f"Scale {scale} Final Results")
        print(f"{'='*70}")
        final_metrics = self.evaluate(scale)
        print(f"  Pixel accuracy: {final_metrics['pixel_accuracy']:.2%}")
        print(f"  Binary accuracy: {final_metrics['binary_accuracy']:.2%}")
        print(f"  Examples: {final_metrics['n_examples']}")

        return final_metrics

    def run(self):
        """Run full training pipeline."""
        print(f"\n{'='*70}")
        print(f"NEURAL-SYMBOLIC ARC TRAINING")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Scales: {self.config.start_scale} → {self.config.end_scale}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.lr}")
        print(f"  Feature dim: {self.config.feature_dim}")
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
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.2f}s")
        print(f"\nFinal accuracies:")
        for scale in range(self.config.start_scale, self.config.end_scale + 1):
            if scale in self.train_tasks:
                metrics = self.evaluate(scale)
                print(f"  Scale {scale}: {metrics['binary_accuracy']:.2%} binary, {metrics['pixel_accuracy']:.2%} pixel")

        return self.metrics


def main():
    """Run neural-symbolic training."""

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
    config = NeuralSymbolicConfig(
        start_scale=0,
        end_scale=2,
        epochs=50,
        batch_size=4,
        lr=1e-3,
        feature_dim=64,
        max_composite_depth=2,
        device=device,
        log_every=5,
        verbose=True
    )

    # Create trainer
    trainer = NeuralSymbolicTrainer(config)

    # Run training
    metrics = trainer.run()

    print("\n✅ Training complete!")
    print(f"\nFinal binary accuracy: {metrics['binary_accuracy'][-1]:.2%}")
    print(f"Final pixel accuracy: {metrics['pixel_accuracy'][-1]:.2%}")


if __name__ == "__main__":
    main()
