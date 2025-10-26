"""
Topos Learner - Fast.ai-inspired API for Topos-Theoretic Neural Networks

Clean, high-level interface for training geometric morphisms on ARC tasks.

Author: Claude Code + Human
Date: October 22, 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np
from abc import ABC, abstractmethod


# ============================================================================
# Callbacks (Fast.ai-inspired)
# ============================================================================

class Callback(ABC):
    """Base callback class."""

    def on_train_begin(self, learner): pass
    def on_train_end(self, learner): pass
    def on_epoch_begin(self, learner): pass
    def on_epoch_end(self, learner): pass
    def on_batch_begin(self, learner): pass
    def on_batch_end(self, learner): pass
    def on_validate_begin(self, learner): pass
    def on_validate_end(self, learner): pass


class EarlyStoppingCallback(Callback):
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def on_validate_end(self, learner):
        val_loss = learner.history['val_loss'][-1]

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            if learner.verbose:
                print(f"Early stopping triggered at epoch {learner.epoch}")


class TensorBoardCallback(Callback):
    """Log metrics to TensorBoard."""

    def __init__(self, log_dir: str, log_verbose: bool = False):
        self.log_dir = log_dir
        self.log_verbose = log_verbose
        self.writer = None

    def on_train_begin(self, learner):
        self.writer = SummaryWriter(self.log_dir)

        # Log model graph (once)
        if hasattr(learner, 'model') and learner.train_loader:
            try:
                batch = next(iter(learner.train_loader))
                dummy_input = batch['input'][:1].to(learner.device)
                self.writer.add_graph(learner.model, dummy_input)
            except:
                pass  # Graph logging is optional

    def on_epoch_end(self, learner):
        epoch = learner.epoch

        # Essential scalars (always logged)
        for key in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
            if key in learner.history and learner.history[key]:
                self.writer.add_scalar(f'Metrics/{key}', learner.history[key][-1], epoch)

        # Learning rate
        if learner.optimizer:
            self.writer.add_scalar('Hyperparameters/lr',
                                 learner.optimizer.param_groups[0]['lr'], epoch)

        # Verbose logging (histograms, images)
        if self.log_verbose and hasattr(learner, 'model'):
            for name, param in learner.model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        self.writer.flush()

    def on_train_end(self, learner):
        if self.writer:
            self.writer.close()


class LRSchedulerCallback(Callback):
    """Learning rate scheduling."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, learner):
        self.scheduler.step()


class ProgressBarCallback(Callback):
    """Display training progress with tqdm."""

    def __init__(self):
        self.pbar = None

    def on_train_begin(self, learner):
        self.pbar = tqdm(total=learner.epochs, desc="Training", ncols=120)

    def on_epoch_end(self, learner):
        if self.pbar:
            postfix = {}
            if learner.history['train_loss']:
                postfix['train_loss'] = f"{learner.history['train_loss'][-1]:.4f}"
            if learner.history['val_loss']:
                postfix['val_loss'] = f"{learner.history['val_loss'][-1]:.4f}"
            if learner.history['val_acc']:
                postfix['val_acc'] = f"{learner.history['val_acc'][-1]:.1%}"

            self.pbar.set_postfix(postfix)
            self.pbar.update(1)

    def on_train_end(self, learner):
        if self.pbar:
            self.pbar.close()


# ============================================================================
# Topos Learner (Main Class)
# ============================================================================

@dataclass
class ToposLossWeights:
    """Loss component weights for topos-theoretic training."""
    l2: float = 1.0              # Primary reconstruction loss
    sheaf_space: float = 0.5     # Sheaf space alignment
    adjunction: float = 0.1      # Adjunction constraint
    sheaf_gluing: float = 0.01   # Sheaf condition


class ToposLearner:
    """Fast.ai-style learner for topos-theoretic neural networks.

    Example:
        learner = ToposLearner(model, train_loader, val_loader)
        learner.fit(epochs=100, lr=1e-3, callbacks=[
            EarlyStoppingCallback(patience=30),
            TensorBoardCallback(log_dir='runs/experiment1')
        ])
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: str = 'cpu',
        loss_weights: Optional[ToposLossWeights] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.loss_weights = loss_weights or ToposLossWeights()

        self.model.to(self.device)

        # Training state
        self.optimizer = None
        self.epoch = 0
        self.epochs = 0
        self.verbose = True

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': [],
        }

    def fit(
        self,
        epochs: int,
        lr: float = 1e-3,
        optimizer: Optional[torch.optim.Optimizer] = None,
        callbacks: Optional[List[Callback]] = None,
        verbose: bool = True
    ):
        """Train the model.

        Args:
            epochs: Number of epochs to train
            lr: Learning rate (if optimizer not provided)
            optimizer: Custom optimizer (if None, uses Adam)
            callbacks: List of callbacks
            verbose: Print progress
        """
        self.epochs = epochs
        self.verbose = verbose
        callbacks = callbacks or []

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        # Training loop
        self._run_callbacks(callbacks, 'on_train_begin')

        for epoch in range(epochs):
            self.epoch = epoch

            # Check early stopping
            early_stop = any(
                getattr(cb, 'should_stop', False)
                for cb in callbacks if isinstance(cb, EarlyStoppingCallback)
            )
            if early_stop:
                break

            self._run_callbacks(callbacks, 'on_epoch_begin')

            # Train
            train_loss, train_acc = self._train_epoch(callbacks)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            val_loss, val_acc = self._validate_epoch(callbacks)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            self._run_callbacks(callbacks, 'on_epoch_end')

        self._run_callbacks(callbacks, 'on_train_end')

        return self.history

    def _train_epoch(self, callbacks: List[Callback]) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_pixels = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self._run_callbacks(callbacks, 'on_batch_begin')

            # Move to device
            inputs = batch['input'].to(self.device)
            targets = batch['output'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute loss (can be overridden for custom topos losses)
            loss = self._compute_loss(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            correct = (outputs.argmax(1) == targets).sum().item()
            total_correct += correct
            total_pixels += targets.numel()

            self._run_callbacks(callbacks, 'on_batch_end')

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_pixels if total_pixels > 0 else 0

        return avg_loss, accuracy

    def _validate_epoch(self, callbacks: List[Callback]) -> tuple[float, float]:
        """Validate for one epoch."""
        self._run_callbacks(callbacks, 'on_validate_begin')

        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['output'].to(self.device)

                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)

                total_loss += loss.item()
                correct = (outputs.argmax(1) == targets).sum().item()
                total_correct += correct
                total_pixels += targets.numel()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_pixels if total_pixels > 0 else 0

        self._run_callbacks(callbacks, 'on_validate_end')

        return avg_loss, accuracy

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss (can be overridden for custom topos losses)."""
        # Default: Cross-entropy
        return nn.functional.cross_entropy(outputs, targets)

    def _run_callbacks(self, callbacks: List[Callback], method: str):
        """Run callback method on all callbacks."""
        for callback in callbacks:
            getattr(callback, method)(self)

    def predict(self, dataloader: DataLoader) -> List[torch.Tensor]:
        """Make predictions on a dataset."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu())

        return predictions

    def save(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'epoch': self.epoch,
            'history': self.history,
        }, path)

    def load(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.history = checkpoint.get('history', {})


# ============================================================================
# Topos-Specific Learner (Subclass)
# ============================================================================

class ARCToposLearner(ToposLearner):
    """Specialized learner for ARC tasks with topos-theoretic losses."""

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute topos-theoretic loss with multiple components."""

        # 1. Primary L2 loss (reconstruction)
        l2_loss = nn.functional.mse_loss(outputs, targets)

        # 2. Sheaf space loss (if model has sheaf encoder)
        sheaf_space_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.model, 'encode_batch_to_sheaf'):
            try:
                input_sheaf = self.model.encode_batch_to_sheaf(targets)
                predicted_sheaf = self.model.geometric_morphism.pushforward(input_sheaf)
                target_sheaf = self.model.encode_batch_to_sheaf(targets)
                sheaf_space_loss = nn.functional.mse_loss(
                    predicted_sheaf.sections,
                    target_sheaf.sections
                )
            except:
                pass  # Skip if not applicable

        # 3. Adjunction constraint (if model has geometric morphism)
        adj_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.model, 'geometric_morphism'):
            try:
                # Simplified adjunction check
                adj_loss = self.model.geometric_morphism.check_adjunction(
                    input_sheaf, target_sheaf
                )
            except:
                pass

        # 4. Sheaf gluing condition
        sheaf_gluing_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.model, 'encode_batch_to_sheaf'):
            try:
                sheaf_gluing_loss = predicted_sheaf.total_sheaf_violation()
            except:
                pass

        # Combined loss
        total_loss = (
            self.loss_weights.l2 * l2_loss +
            self.loss_weights.sheaf_space * sheaf_space_loss +
            self.loss_weights.adjunction * adj_loss +
            self.loss_weights.sheaf_gluing * sheaf_gluing_loss
        )

        return total_loss


# ============================================================================
# Learning Rate Finder (Fast.ai-inspired)
# ============================================================================

class LRFinder:
    """Find optimal learning rate using fast.ai's LR range test."""

    def __init__(self, learner: ToposLearner):
        self.learner = learner

    def find(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_f: float = 0.05
    ) -> tuple[List[float], List[float]]:
        """Run LR range test.

        Returns:
            (learning_rates, losses): Lists of LRs and corresponding losses
        """
        model = self.learner.model
        optimizer = self.learner.optimizer

        # Save initial state
        initial_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None
        }

        # Setup
        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
        losses = []

        model.train()
        train_iter = iter(self.learner.train_loader)

        for lr in tqdm(lrs, desc="LR Finder"):
            # Update LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.learner.train_loader)
                batch = next(train_iter)

            # Training step
            inputs = batch['input'].to(self.learner.device)
            targets = batch['output'].to(self.learner.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.learner._compute_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            # Smooth loss
            if losses:
                loss_val = smooth_f * loss.item() + (1 - smooth_f) * losses[-1]
            else:
                loss_val = loss.item()

            losses.append(loss_val)

            # Stop if loss explodes
            if len(losses) > 10 and loss_val > 4 * min(losses):
                break

        # Restore initial state
        model.load_state_dict(initial_state['model'])
        if optimizer and initial_state['optimizer']:
            optimizer.load_state_dict(initial_state['optimizer'])

        return lrs[:len(losses)], losses


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Topos Learner API Example")
    print("="*70)
    print()

    print("Usage:")
    print()
    print("# Create learner")
    print("learner = ARCToposLearner(")
    print("    model=my_topos_model,")
    print("    train_loader=train_loader,")
    print("    val_loader=val_loader,")
    print("    device='cuda'")
    print(")")
    print()
    print("# Find optimal learning rate")
    print("lr_finder = LRFinder(learner)")
    print("lrs, losses = lr_finder.find()")
    print()
    print("# Train with callbacks")
    print("learner.fit(")
    print("    epochs=100,")
    print("    lr=1e-3,")
    print("    callbacks=[")
    print("        EarlyStoppingCallback(patience=30),")
    print("        TensorBoardCallback(log_dir='runs/experiment1', log_verbose=False),")
    print("        ProgressBarCallback(),")
    print("        LRSchedulerCallback(torch.optim.lr_scheduler.CosineAnnealingLR(...))")
    print("    ]")
    print(")")
    print()
    print("# Save/load")
    print("learner.save('model.pt')")
    print("learner.load('model.pt')")
    print()
    print("="*70)
    print()
    print("Key features:")
    print("  ✓ Clean API inspired by fast.ai")
    print("  ✓ Callback system for extensibility")
    print("  ✓ Learning rate finder")
    print("  ✓ Topos-specific losses (sheaf, adjunction)")
    print("  ✓ Full control over custom architectures")
    print("  ✓ TensorBoard integration")
    print("  ✓ Early stopping, LR scheduling")
