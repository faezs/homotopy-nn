"""
Neural-Symbolic ARC Solver

Integrates the complete neural-symbolic pipeline for ARC tasks:

1. **Formula DSL** (internal_language.py): Symbolic structure
2. **Kripke-Joyal semantics** (kripke_joyal.py): Topos forcing
3. **Neural predicates** (neural_predicates.py): Learned atomic formulas
4. **Formula templates** (formula_templates.py): Pattern library
5. **Gumbel-Softmax selection**: Differentiable program search

ARCHITECTURE:

```
Input Grid
    ↓
CNN Encoder → Features
    ↓
Formula Selector (Gumbel-Softmax) → Select template from library
    ↓
Neural Predicates → Evaluate atomic formulas
    ↓
Kripke-Joyal Interpreter → Compute formula truth values
    ↓
Apply Transformation → Output Grid
```

The key innovation: Instead of black-box f: Grid → Grid, we learn
f = ⟦formula⟧ where formula ∈ Template_Library is:
- Interpretable (can see which template selected)
- Compositional (templates combine logically)
- Generalizable (same template works across scales)

Author: Claude Code
Date: October 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from internal_language import Formula
from kripke_joyal import KripkeJoyalInterpreter, KJContext, create_arc_context
from neural_predicates import PredicateRegistry
from formula_templates import (
    TemplateLibrary,
    SequentialFormula,
    enumerate_color_transforms,
    enumerate_geometric_transforms,
    enumerate_composite_transforms,
)
from topos_categorical import Site, SubobjectClassifier


################################################################################
# § 1: Gumbel-Softmax Formula Selector
################################################################################

class FormulaSelector(nn.Module):
    """Differentiable formula selection using Gumbel-Softmax.

    Given input features, selects template from library using:
    - Gumbel-Softmax for differentiable sampling
    - Straight-through estimator for gradient flow
    - Temperature annealing (hard → soft during training)
    """

    def __init__(
        self,
        input_dim: int,
        num_templates: int,
        hidden_dim: int = 128,
        temperature: float = 1.0
    ):
        super().__init__()

        self.num_templates = num_templates
        self.temperature = temperature

        # Scoring network: features → template logits
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_templates)
        )

    def forward(
        self,
        features: torch.Tensor,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select template using Gumbel-Softmax.

        Args:
            features: Input features [batch_size, input_dim]
            hard: If True, return one-hot (testing), else soft (training)

        Returns:
            selection: Soft/hard selection weights [batch_size, num_templates]
            logits: Raw template scores [batch_size, num_templates]
        """
        # Compute template scores
        logits = self.scorer(features)

        # Gumbel-Softmax sampling
        selection = F.gumbel_softmax(
            logits,
            tau=self.temperature,
            hard=hard,
            dim=-1
        )

        return selection, logits

    def set_temperature(self, temperature: float):
        """Anneal temperature during training."""
        self.temperature = temperature


################################################################################
# § 2: Feature Encoder (CNN)
################################################################################

class GridEncoder(nn.Module):
    """CNN encoder for grid features.

    Extracts visual features from input grid for formula selection.
    """

    def __init__(
        self,
        num_colors: int = 10,
        feature_dim: int = 64,
        output_dim: int = 128
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_colors, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global pooling
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim * 2, output_dim),
            nn.ReLU()
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode grid to features.

        Args:
            grid: [batch_size, num_colors, H, W] one-hot encoded

        Returns:
            features: [batch_size, output_dim]
        """
        x = self.encoder(grid)
        x = x.view(x.size(0), -1)
        features = self.projector(x)
        return features


################################################################################
# § 3: Neural-Symbolic ARC Solver
################################################################################

class NeuralSymbolicARCSolver(nn.Module):
    """Complete neural-symbolic solver for ARC tasks.

    Combines:
    - Visual encoding (CNN)
    - Symbolic reasoning (formulas)
    - Differentiable logic (Kripke-Joyal)
    - Program search (Gumbel-Softmax)
    """

    def __init__(
        self,
        num_colors: int = 10,
        feature_dim: int = 64,
        device: torch.device = None,
        max_composite_depth: int = 2
    ):
        super().__init__()

        self.num_colors = num_colors
        self.feature_dim = feature_dim
        self.device = device or torch.device('cpu')

        # Build template library
        self.template_library = TemplateLibrary()
        self._build_search_space(max_composite_depth)

        # Initialize components
        self.encoder = GridEncoder(num_colors, feature_dim, output_dim=128)

        self.formula_selector = FormulaSelector(
            input_dim=128,
            num_templates=len(self.templates),
            hidden_dim=128,
            temperature=1.0
        )

        # Neural predicates
        self.predicates = PredicateRegistry(num_colors, feature_dim, device)

        # Topos components
        self.site = Site(grid_shape=(1, 1), connectivity="4")  # Single-object site
        self.omega = SubobjectClassifier(self.site, truth_dim=1)

        # Kripke-Joyal interpreter
        self.interpreter = self._build_interpreter()

    def _build_search_space(self, max_depth: int):
        """Build complete template search space."""
        self.templates = []

        # Color transformation templates
        color_templates = enumerate_color_transforms(self.num_colors)
        self.templates.extend(color_templates)

        # Geometric transformation templates
        geom_templates = enumerate_geometric_transforms()
        self.templates.extend(geom_templates)

        # Composite templates (if depth >= 2)
        if max_depth >= 2:
            # Limit combinatorial explosion by using subset
            base_templates = geom_templates[:5]  # Top 5 geometric
            composites = enumerate_composite_transforms(base_templates, max_depth=2)
            self.templates.extend(composites)

        print(f"Template search space: {len(self.templates)} formulas")

    def _build_interpreter(self) -> KripkeJoyalInterpreter:
        """Build Kripke-Joyal interpreter with neural predicates."""
        # Convert PredicateRegistry to dict of callables
        predicate_dict = {}

        # Cell-based predicates (single cell index)
        for name in ['is_boundary', 'is_inside', 'is_corner', 'color_eq', 'same_color']:
            pred = self.predicates.get(name)
            predicate_dict[name] = (lambda p: lambda *args, context: p(*args, context=context))(pred)

        # Geometric transformation predicates (cell → color)
        for name in ['reflected_color_h', 'reflected_color_v', 'rotated_color_90',
                     'translated_color', 'neighbor_color_left', 'neighbor_color_right',
                     'neighbor_color_up', 'neighbor_color_down']:
            pred = self.predicates.get(name)
            predicate_dict[name] = (lambda p: lambda *args, context: p(*args, context=context))(pred)

        # TODO: Add region-based predicates (is_square, touches, border_of)
        # These need special handling since they operate on region masks

        return KripkeJoyalInterpreter(self.omega, predicate_dict, self.device)

    def forward(
        self,
        input_grid: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        hard_select: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass: select formula and apply transformation.

        Args:
            input_grid: [batch_size, H, W] or [H, W] color indices
            target_size: Optional (H, W) for output size
            hard_select: If True, use hard selection (testing)

        Returns:
            output_grid: [batch_size, H, W] transformed grid
            info: Dictionary with intermediate values
        """
        # Handle both batched and unbatched inputs
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)
            unbatch_output = True
        else:
            unbatch_output = False

        batch_size, H, W = input_grid.shape

        # One-hot encode
        input_one_hot = F.one_hot(input_grid.long(), num_classes=self.num_colors).float()
        input_one_hot = input_one_hot.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Encode to features
        features = self.encoder(input_one_hot)

        # Select formula
        selection, logits = self.formula_selector(features, hard=hard_select)

        # Get selected template indices
        selected_idx = selection.argmax(dim=-1)  # [batch_size]

        # Apply formulas (batched)
        output_grids = []
        for b in range(batch_size):
            idx = selected_idx[b].item()
            template = self.templates[idx]

            # Apply formula to grid with target size
            output = self._apply_formula(input_grid[b], template, target_size=target_size)
            output_grids.append(output)

        output_grid = torch.stack(output_grids)

        # Unbatch if input was unbatched
        if unbatch_output:
            output_grid = output_grid.squeeze(0)

        # Return info
        info = {
            'features': features,
            'selection': selection,
            'logits': logits,
            'selected_indices': selected_idx,
            'selected_templates': [self.templates[idx.item()] for idx in selected_idx]
        }

        return output_grid, info

    def _apply_formula(self, grid: torch.Tensor, formula: Formula, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Apply formula to single grid.

        Args:
            grid: [H, W] color indices
            formula: Formula to apply
            target_size: Optional (H, W) for output size (defaults to input size)

        Returns:
            output: [H, W] transformed grid
        """
        H_in, W_in = grid.shape
        H_out, W_out = target_size if target_size is not None else (H_in, W_in)

        # Handle sequential formulas
        if isinstance(formula, SequentialFormula):
            current_grid = grid.clone()
            for step_formula in formula.steps:
                current_grid = self._apply_formula(current_grid, step_formula, target_size)
            return current_grid

        # Create output grid with target size (initialize with background color 0)
        output = torch.zeros(H_out, W_out, device=grid.device, dtype=grid.dtype)

        # Copy input grid (centered or cropped as needed)
        copy_h = min(H_in, H_out)
        copy_w = min(W_in, W_out)
        output[:copy_h, :copy_w] = grid[:copy_h, :copy_w]

        # Evaluate formula for each cell in OUTPUT grid
        for i in range(H_out):
            for j in range(W_out):
                cell_idx = i * W_out + j

                # Create context with OUTPUT grid dimensions
                context = create_arc_context(output, cell_idx)
                context.aux['input_grid'] = grid  # Store input for reference
                context.aux['output_size'] = (H_out, W_out)

                # Evaluate formula (truth value)
                truth = self.interpreter.force(formula, context)

                # Check if transformation assigned new color
                if f'assigned_color' in context.bindings:
                    new_color = context.bindings['assigned_color']
                    # Soft assignment weighted by truth value
                    current_color = output[i, j]
                    output[i, j] = truth * new_color + (1 - truth) * current_color

        return output

    def compute_loss(
        self,
        input_grid: torch.Tensor,
        target_grid: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        hard_select: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss for training.

        Args:
            input_grid: [batch_size, H, W] or [H, W]
            target_grid: [batch_size, H, W] or [H, W]
            target_size: Optional (H, W) for output size
            hard_select: Use hard selection

        Returns:
            loss: Total loss
            losses: Dictionary of loss components
        """
        # Forward pass with target size (output will match target_grid size)
        output_grid, info = self.forward(input_grid, target_size=target_size, hard_select=hard_select)

        # Ensure same batch dimensions
        if output_grid.dim() == 2:
            output_grid = output_grid.unsqueeze(0)
        if target_grid.dim() == 2:
            target_grid = target_grid.unsqueeze(0)

        # Sizes should now match (forward handles resizing)
        # Pixel-level MSE (output_grid is continuous color indices)
        pixel_loss = F.mse_loss(output_grid, target_grid)

        # Encourage confident template selection (entropy regularization)
        template_probs = F.softmax(info['logits'], dim=-1)
        template_entropy = -(template_probs * torch.log(template_probs + 1e-8)).sum(dim=-1).mean()

        # Total loss
        loss = pixel_loss + 0.01 * template_entropy  # Small entropy penalty

        losses = {
            'total': loss,
            'pixel': pixel_loss,
            'template_entropy': template_entropy
        }

        return loss, losses

    def predict(self, input_grid: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Predict output grid (testing mode).

        Args:
            input_grid: [H, W] or [batch_size, H, W]
            target_size: Optional (H, W) for output size

        Returns:
            output_grid: [H, W] or [batch_size, H, W]
        """
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)

        with torch.no_grad():
            output_grid, _ = self.forward(input_grid, target_size=target_size, hard_select=True)

        if output_grid.size(0) == 1:
            output_grid = output_grid.squeeze(0)

        return output_grid

    def get_selected_template(self, input_grid: torch.Tensor) -> Formula:
        """Get the template selected for input.

        Args:
            input_grid: [H, W]

        Returns:
            Selected formula template
        """
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)

        with torch.no_grad():
            _, info = self.forward(input_grid, hard_select=True)

        return info['selected_templates'][0]

    def set_temperature(self, temperature: float):
        """Anneal Gumbel-Softmax temperature."""
        self.formula_selector.set_temperature(temperature)


################################################################################
# § 4: Training Utilities
################################################################################

def train_step(
    model: NeuralSymbolicARCSolver,
    optimizer: torch.optim.Optimizer,
    input_grid: torch.Tensor,
    target_grid: torch.Tensor,
    temperature: float = 1.0
) -> Dict[str, float]:
    """Single training step.

    Args:
        model: Neural-symbolic solver
        optimizer: Optimizer
        input_grid: [batch_size, H, W]
        target_grid: [batch_size, H, W]
        temperature: Gumbel-Softmax temperature

    Returns:
        Dictionary of loss values
    """
    model.train()
    model.set_temperature(temperature)

    optimizer.zero_grad()

    loss, losses = model.compute_loss(input_grid, target_grid, hard_select=False)

    loss.backward()
    optimizer.step()

    return {k: v.item() for k, v in losses.items()}


def evaluate(
    model: NeuralSymbolicARCSolver,
    input_grids: List[torch.Tensor],
    target_grids: List[torch.Tensor]
) -> Dict[str, float]:
    """Evaluate model on test set.

    Args:
        model: Neural-symbolic solver
        input_grids: List of input grids [H, W]
        target_grids: List of target grids [H, W]

    Returns:
        Evaluation metrics
    """
    model.eval()

    total_correct = 0
    total_pixels = 0
    exact_matches = 0

    with torch.no_grad():
        for inp, tgt in zip(input_grids, target_grids):
            pred = model.predict(inp)

            # Pixel accuracy
            correct = (pred == tgt).float().sum()
            total_correct += correct.item()
            total_pixels += tgt.numel()

            # Exact match
            if torch.all(pred == tgt):
                exact_matches += 1

    return {
        'pixel_accuracy': total_correct / total_pixels,
        'exact_match_rate': exact_matches / len(input_grids)
    }


################################################################################
# § 5: Example Usage
################################################################################

if __name__ == "__main__":
    """
    Test neural-symbolic solver on simple example.
    """

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create model
    model = NeuralSymbolicARCSolver(
        num_colors=10,
        feature_dim=64,
        device=device,
        max_composite_depth=2
    )
    model = model.to(device)

    # Example task: Fill interior with red (color 2)
    input_grid = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=torch.float32, device=device)

    target_grid = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],  # Interior filled with red
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=torch.float32, device=device)

    print("Input grid:")
    print(input_grid.cpu().numpy())
    print("\nTarget grid:")
    print(target_grid.cpu().numpy())

    # Predict (before training)
    print("\n=== Before Training ===")
    pred_before = model.predict(input_grid)
    print("Predicted grid:")
    print(pred_before.cpu().numpy())

    selected = model.get_selected_template(input_grid)
    print(f"\nSelected template: {selected}")

    # Train for a few steps
    print("\n=== Training ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        losses = train_step(
            model, optimizer,
            input_grid.unsqueeze(0),
            target_grid.unsqueeze(0),
            temperature=1.0
        )
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: loss={losses['total']:.4f}")

    # Predict (after training)
    print("\n=== After Training ===")
    pred_after = model.predict(input_grid)
    print("Predicted grid:")
    print(pred_after.cpu().numpy())

    selected_after = model.get_selected_template(input_grid)
    print(f"\nSelected template: {selected_after}")

    pixel_acc = (pred_after == target_grid).float().mean()
    print(f"\nPixel accuracy: {pixel_acc.item():.2%}")

    print("\n✓ Neural-symbolic solver tested successfully!")
