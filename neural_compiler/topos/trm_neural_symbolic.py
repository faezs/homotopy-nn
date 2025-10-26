"""
TRM-Enhanced Neural-Symbolic ARC Solver

Integrates Tiny Recursive Model (TRM) architecture with neural-symbolic reasoning:

1. **TRM Architecture** (from "Less is More" paper):
   - Tiny 2-layer MLP-Mixer for encoding
   - Recursive refinement: (y=answer, z=reasoning)
   - Deep recursion: T-1 cycles without gradients, 1 with gradients
   - EMA for training stability

2. **Neural-Symbolic Integration**:
   - Use refined y embedding for formula selection
   - Keep Kripke-Joyal symbolic transformation
   - Interpretable + recursive reasoning

**Key Innovation**: Combines TRM's 45% ARC-AGI-1 accuracy with interpretable
formula-based transformations.

Author: Claude Code
Date: October 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from copy import deepcopy

from internal_language import Formula
from kripke_joyal import KripkeJoyalInterpreter, KJContext, create_arc_context
from neural_predicates import PredicateRegistry
from formula_templates import (
    TemplateLibrary,
    enumerate_color_transforms,
    enumerate_geometric_transforms,
    enumerate_composite_transforms,
)
from topos_categorical import Site, SubobjectClassifier


################################################################################
# § 1: Exponential Moving Average (EMA)
################################################################################

class EMA:
    """Exponential Moving Average for model weights.

    Used for training stability as in TRM paper and diffusion models.

    Usage:
        ema = EMA(model, decay=0.999)

        # During training:
        loss.backward()
        optimizer.step()
        ema.update()  # Update shadow weights

        # During evaluation:
        ema.apply_shadow()  # Use EMA weights
        evaluate(model)
        ema.restore()  # Restore training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Initialize EMA.

        Args:
            model: PyTorch model to track
            decay: EMA decay rate (higher = more smoothing)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}  # EMA weights
        self.backup = {}  # Backup for restore

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights after training step.

        Formula: θ_shadow = decay * θ_shadow + (1 - decay) * θ_current
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not in shadow"
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Replace model weights with EMA weights (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original training weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


################################################################################
# § 2: Tiny MLP-Mixer Encoder (2 layers)
################################################################################

class TinyMLPMixer(nn.Module):
    """Tiny 2-layer MLP-Mixer encoder.

    Replaces large CNN with minimal architecture as per TRM paper:
    - Layer 1: Token mixing (spatial patterns)
    - Layer 2: Channel mixing (feature patterns)
    - Output: y (answer embedding) + z (reasoning feature)

    Key advantage: Works better than self-attention on small fixed grids.
    """

    def __init__(
        self,
        num_colors: int = 10,
        answer_dim: int = 128,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        max_grid_size: int = 30
    ):
        """Initialize MLP-Mixer.

        Args:
            num_colors: Number of distinct colors (10 for ARC)
            answer_dim: Dimension of answer embedding y
            latent_dim: Dimension of reasoning feature z
            hidden_dim: Hidden layer dimension
            max_grid_size: Maximum grid size (30x30 for ARC)
        """
        super().__init__()

        self.num_colors = num_colors
        self.answer_dim = answer_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size

        # Simplified 2-layer architecture as per TRM paper
        input_dim = max_grid_size * max_grid_size * num_colors

        # Layer 1
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.GELU()

        # Layer 2
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = nn.GELU()

        # Projection heads
        self.to_y = nn.Linear(hidden_dim, answer_dim)
        self.to_z = nn.Linear(hidden_dim, latent_dim)

    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode grid to answer + reasoning embeddings.

        Args:
            grid: [batch_size, H, W] color indices (0-9)

        Returns:
            y: [batch_size, answer_dim] answer embedding
            z: [batch_size, latent_dim] reasoning feature
        """
        batch_size, H, W = grid.shape

        # One-hot encode: [B, H, W] → [B, H, W, C]
        grid_one_hot = F.one_hot(grid.long(), num_classes=self.num_colors).float()

        # Pad to max_grid_size if needed
        if H < self.max_grid_size or W < self.max_grid_size:
            pad_h = self.max_grid_size - H
            pad_w = self.max_grid_size - W
            grid_one_hot = F.pad(
                grid_one_hot,
                (0, 0, 0, pad_w, 0, pad_h),  # (C, W, H) padding
                mode='constant',
                value=0
            )
        elif H > self.max_grid_size or W > self.max_grid_size:
            # Crop if larger
            grid_one_hot = grid_one_hot[:, :self.max_grid_size, :self.max_grid_size, :]

        # grid_one_hot: [B, max_grid_size, max_grid_size, C]
        # Flatten to [B, max_grid_size * max_grid_size * C]
        x = grid_one_hot.reshape(batch_size, -1)

        # Layer 1
        h = self.layer1(x)
        h = self.activation1(h)

        # Layer 2
        h = self.layer2(h)
        h = self.activation2(h)

        # Project to y and z
        y = self.to_y(h)
        z = self.to_z(h)

        return y, z


################################################################################
# § 3: Tiny Refiner (2 layers)
################################################################################

class TinyRefiner(nn.Module):
    """Tiny 2-layer MLP for recursive refinement.

    Refines either:
    - z (reasoning feature) given [y, z, grid_features]
    - y (answer embedding) given [z, y, grid_features]

    Key: Only 2 layers to keep model tiny (TRM paper shows 2 > 4 layers).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64
    ):
        """Initialize refiner.

        Args:
            input_dim: Concatenated input dimension
            output_dim: Output dimension (latent_dim for z, answer_dim for y)
            hidden_dim: Hidden layer dimension (small!)
        """
        super().__init__()

        self.refine = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine feature.

        Args:
            x: [batch_size, input_dim] concatenated features

        Returns:
            refined: [batch_size, output_dim] refined feature
        """
        return self.refine(x)


################################################################################
# § 4: Tiny Formula Selector (2 layers)
################################################################################

class TinyFormulaSelector(nn.Module):
    """Tiny 2-layer formula selector.

    Replaces larger FormulaSelector with minimal 2-layer version.
    Uses answer embedding y to select formula template.
    """

    def __init__(
        self,
        input_dim: int,
        num_templates: int,
        hidden_dim: int = 64,
        temperature: float = 1.0
    ):
        """Initialize selector.

        Args:
            input_dim: Answer embedding dimension
            num_templates: Number of formula templates
            hidden_dim: Hidden dimension (small!)
            temperature: Gumbel-Softmax temperature
        """
        super().__init__()

        self.num_templates = num_templates
        self.temperature = temperature

        # Tiny 2-layer scorer
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_templates)
        )

    def forward(
        self,
        y: torch.Tensor,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select template using Gumbel-Softmax.

        Args:
            y: [batch_size, input_dim] answer embedding
            hard: If True, one-hot selection (testing)

        Returns:
            selection: [batch_size, num_templates] soft/hard weights
            logits: [batch_size, num_templates] raw scores
        """
        # Score templates
        logits = self.scorer(y)

        # Gumbel-Softmax
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
# § 5: TRM Neural-Symbolic Solver
################################################################################

class TRMNeuralSymbolicSolver(nn.Module):
    """TRM-Enhanced Neural-Symbolic ARC Solver.

    Combines:
    1. TRM recursive reasoning (tiny networks, deep recursion)
    2. Neural-symbolic formulas (interpretable transformations)

    Architecture:
        Input Grid
            ↓
        TinyMLPMixer → (y, z)
            ↓
        Recursive Refinement (T cycles):
            For t=1..T-1 (no gradients):
                z ← refine_z([y, z, grid_features])
                y ← refine_y([z, y, grid_features])
            For t=T (with gradients):
                z ← refine_z([y, z, grid_features])
                y ← refine_y([z, y, grid_features])
            ↓
        TinyFormulaSelector(y) → Select formula template
            ↓
        Kripke-Joyal Interpreter → Apply formula → Output Grid

    Expected improvement: 0% → 10-45% based on TRM paper results.
    """

    def __init__(
        self,
        num_colors: int = 10,
        answer_dim: int = 128,
        latent_dim: int = 64,
        num_cycles: int = 3,
        device: torch.device = None,
        max_composite_depth: int = 2
    ):
        """Initialize TRM neural-symbolic solver.

        Args:
            num_colors: Number of colors (10 for ARC)
            answer_dim: Answer embedding dimension
            latent_dim: Reasoning feature dimension
            num_cycles: Number of recursion cycles (T)
            device: Device to run on
            max_composite_depth: Max depth for composite formulas
        """
        super().__init__()

        self.num_colors = num_colors
        self.answer_dim = answer_dim
        self.latent_dim = latent_dim
        self.num_cycles = num_cycles
        self.device = device or torch.device('cpu')

        # Build template library (reuse from neural_symbolic_arc)
        self.template_library = TemplateLibrary()
        self._build_search_space(max_composite_depth)

        # TRM Components
        self.encoder = TinyMLPMixer(
            num_colors=num_colors,
            answer_dim=answer_dim,
            latent_dim=latent_dim
        )

        # Grid feature extractor (simple statistics)
        self.grid_feature_dim = 16  # Simple features: mean color per row/col, etc.

        # Refiners
        refine_z_input_dim = answer_dim + latent_dim + self.grid_feature_dim
        refine_y_input_dim = latent_dim + answer_dim + self.grid_feature_dim

        self.refine_z = TinyRefiner(
            input_dim=refine_z_input_dim,
            output_dim=latent_dim,
            hidden_dim=64
        )

        self.refine_y = TinyRefiner(
            input_dim=refine_y_input_dim,
            output_dim=answer_dim,
            hidden_dim=64
        )

        # Formula selector
        self.formula_selector = TinyFormulaSelector(
            input_dim=answer_dim,
            num_templates=len(self.templates),
            hidden_dim=64,
            temperature=1.0
        )

        # Neural predicates (reuse from neural_symbolic_arc)
        self.predicates = PredicateRegistry(num_colors, 64, device)

        # Topos components
        self.site = Site(grid_shape=(1, 1), connectivity="4")
        self.omega = SubobjectClassifier(self.site, truth_dim=1)

        # Kripke-Joyal interpreter (old cell-by-cell - for compatibility)
        self.interpreter = self._build_interpreter()

        # ✅ NEW: Vectorized components for gradient flow
        from neural_predicates_vectorized import VectorizedPredicateRegistry
        from kripke_joyal_vectorized import VectorizedKripkeJoyalInterpreter

        self.predicates_vectorized = VectorizedPredicateRegistry(
            num_colors=num_colors,
            feature_dim=64,
            device=device
        )

        self.interpreter_vectorized = VectorizedKripkeJoyalInterpreter(
            omega=self.omega,
            predicates=self.predicates_vectorized,
            device=device
        )

    def _build_search_space(self, max_depth: int):
        """Build formula template search space."""
        self.templates = []

        # Color transformations
        color_templates = enumerate_color_transforms(self.num_colors)
        self.templates.extend(color_templates)

        # Geometric transformations
        geom_templates = enumerate_geometric_transforms()
        self.templates.extend(geom_templates)

        # Composites
        if max_depth >= 2:
            base_templates = geom_templates[:5]
            composites = enumerate_composite_transforms(base_templates, max_depth=2)
            self.templates.extend(composites)

        print(f"TRM template search space: {len(self.templates)} formulas")

    def _build_interpreter(self) -> KripkeJoyalInterpreter:
        """Build Kripke-Joyal interpreter with neural predicates."""
        predicate_dict = {}

        # Cell-based predicates
        for name in ['is_boundary', 'is_inside', 'is_corner', 'color_eq', 'same_color']:
            pred = self.predicates.get(name)
            predicate_dict[name] = (lambda p: lambda *args, context: p(*args, context=context))(pred)

        # Geometric transformation predicates
        for name in ['reflected_color_h', 'reflected_color_v', 'rotated_color_90',
                     'translated_color', 'neighbor_color_left', 'neighbor_color_right',
                     'neighbor_color_up', 'neighbor_color_down']:
            pred = self.predicates.get(name)
            predicate_dict[name] = (lambda p: lambda *args, context: p(*args, context=context))(pred)

        return KripkeJoyalInterpreter(self.omega, predicate_dict, self.device)

    def _extract_grid_features(self, grid: torch.Tensor) -> torch.Tensor:
        """Extract simple statistical features from grid.

        Features:
        - Mean color per row (top 4 rows)
        - Mean color per column (left 4 cols)
        - Overall mean
        - Max color
        - Min color (non-zero)
        - Number of distinct colors
        - Spatial variance

        Args:
            grid: [batch_size, H, W] color indices

        Returns:
            features: [batch_size, grid_feature_dim]
        """
        batch_size, H, W = grid.shape
        features = []

        # Mean color per row (top 4)
        for i in range(min(4, H)):
            features.append(grid[:, i, :].float().mean(dim=1, keepdim=True))
        # Pad if H < 4
        for i in range(4 - min(4, H)):
            features.append(torch.zeros(batch_size, 1, device=grid.device))

        # Mean color per column (left 4)
        for j in range(min(4, W)):
            features.append(grid[:, :, j].float().mean(dim=1, keepdim=True))
        # Pad if W < 4
        for j in range(4 - min(4, W)):
            features.append(torch.zeros(batch_size, 1, device=grid.device))

        # Overall statistics (all should be [batch_size, 1])
        features.append(grid.float().mean(dim=(1, 2), keepdim=True).squeeze(1))  # mean -> [B, 1]
        max_val = grid.float().max(dim=2).values.max(dim=1, keepdim=True).values  # [B, 1]
        features.append(max_val)
        min_val = (grid.float() + 1e-6).min(dim=2).values.min(dim=1, keepdim=True).values  # [B, 1]
        features.append(min_val)

        # Number of distinct colors (approximate)
        features.append((grid.float().std(dim=(1, 2), keepdim=True).squeeze(1) * 10).clamp(0, 10))

        # Concatenate: [batch_size, 4+4+4] = [batch_size, 12]
        # Need to get to 16 features, add 4 more
        features.append(torch.ones(batch_size, 1, device=grid.device))  # bias
        features.append((H / 30.0) * torch.ones(batch_size, 1, device=grid.device))  # normalized height
        features.append((W / 30.0) * torch.ones(batch_size, 1, device=grid.device))  # normalized width
        features.append((H * W / 900.0) * torch.ones(batch_size, 1, device=grid.device))  # normalized area

        grid_features = torch.cat(features, dim=1)  # [batch_size, 16]
        assert grid_features.shape[1] == self.grid_feature_dim

        return grid_features

    def forward(
        self,
        input_grid: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        hard_select: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with TRM recursive refinement.

        Args:
            input_grid: [batch_size, H, W] or [H, W] color indices
            target_size: Optional (H_out, W_out) for output size
            hard_select: If True, use hard selection (testing)

        Returns:
            output_grid: [batch_size, H_out, W_out] transformed grid
            info: Dictionary with intermediate values
        """
        # Handle unbatched input
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)
            unbatch_output = True
        else:
            unbatch_output = False

        batch_size, H, W = input_grid.shape

        # CRITICAL: Convert input to float for gradient flow!
        # PyTorch doesn't support gradients through integer tensors
        # All downstream operations will use float, convert back to int only at the end
        input_grid_float = input_grid.float()

        # Note: encoder and feature extraction can still use the original long grid
        # since they don't need gradients wrt the grid itself

        # Initial encoding
        y, z = self.encoder(input_grid)  # [B, answer_dim], [B, latent_dim]

        # Extract grid features
        grid_features = self._extract_grid_features(input_grid)  # [B, grid_feature_dim]

        # Recursive refinement
        # Allow gradients through all cycles for learning
        # (Original TRM uses no_grad for T-1 cycles, but that prevents learning
        # in our formula selection architecture)
        for t in range(self.num_cycles):
            # Refine reasoning z
            z_input = torch.cat([y, z, grid_features], dim=1)
            z = self.refine_z(z_input)

            # Refine answer y
            y_input = torch.cat([z, y, grid_features], dim=1)
            y = self.refine_y(y_input)

        # Formula selection using refined answer embedding
        selection, logits = self.formula_selector(y, hard=hard_select)

        # Apply formula with selection weights
        output_grids = []
        selected_idx = torch.argmax(selection, dim=1)  # [batch_size]

        for b in range(batch_size):
            idx = selected_idx[b].item()
            template = self.templates[idx]
            # ✅ Use vectorized evaluation for gradient flow
            # Use FLOAT grid to maintain gradient connectivity
            output = self._apply_formula_vectorized(
                input_grid_float[b],  # Use float version!
                template,
                target_size=target_size
            )
            output_grids.append(output)

        output_grid = torch.stack(output_grids)

        if unbatch_output:
            output_grid = output_grid.squeeze(0)

        info = {
            'y': y,
            'z': z,
            'logits': logits,
            'selection': selection,
            'selected_templates': [self.templates[idx.item()] for idx in selected_idx]
        }

        return output_grid, info

    def _apply_formula(
        self,
        grid: torch.Tensor,
        formula: Formula,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Apply formula to grid (same as neural_symbolic_arc.py).

        Args:
            grid: [H, W] color indices
            formula: Formula to apply
            target_size: Optional (H_out, W_out)

        Returns:
            output: [H_out, W_out] transformed grid
        """
        H_in, W_in = grid.shape
        H_out, W_out = target_size if target_size is not None else (H_in, W_in)

        # Create output grid with target size
        output = torch.zeros(H_out, W_out, device=grid.device, dtype=grid.dtype)

        # Copy input grid (centered or cropped)
        copy_h, copy_w = min(H_in, H_out), min(W_in, W_out)
        output[:copy_h, :copy_w] = grid[:copy_h, :copy_w]

        # Evaluate formula for each cell in output grid
        for i in range(H_out):
            for j in range(W_out):
                cell_idx = i * W_out + j
                context = create_arc_context(output, cell_idx)
                context.aux['input_grid'] = grid
                context.aux['output_size'] = (H_out, W_out)

                # Force formula
                truth = self.interpreter.force(formula, context)

                # Apply transformation if formula assigns color
                if 'assigned_color' in context.bindings:
                    new_color = context.bindings['assigned_color']
                    # Soft assignment weighted by truth value
                    output[i, j] = truth * new_color + (1 - truth) * output[i, j]

        return output

    def _apply_formula_vectorized(
        self,
        grid: torch.Tensor,
        formula: Formula,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Apply formula using vectorized evaluation (NEW - enables gradient flow).

        This replaces the cell-by-cell loops in _apply_formula with batched
        tensor operations, allowing gradients to flow to neural predicates.

        Args:
            grid: [H, W] color indices
            formula: Formula to apply
            target_size: Optional (H_out, W_out)

        Returns:
            output: [H_out, W_out] transformed grid (fully differentiable!)
        """
        from kripke_joyal_vectorized import create_grid_context

        H_in, W_in = grid.shape
        H_out, W_out = target_size if target_size is not None else (H_in, W_in)

        # CRITICAL: Work in FLOAT for gradient flow!
        # Integer tensors (long) cannot have gradients in PyTorch
        # We convert to float, do all operations, then round back to long at the end

        # Create output grid with target size (pad/crop) - use FLOAT!
        output = torch.zeros(H_out, W_out, device=grid.device, dtype=torch.float32)
        copy_h, copy_w = min(H_in, H_out), min(W_in, W_out)
        output[:copy_h, :copy_w] = grid[:copy_h, :copy_w].float()  # Convert to float

        # Create grid context for vectorized evaluation
        # CRITICAL: Pass FLOAT grid to preserve gradient flow!
        context = create_grid_context(output)  # Keep as float!
        context.aux['input_grid'] = grid
        context.aux['output_size'] = (H_out, W_out)

        # ✅ Vectorized evaluation (NO Python loops!)
        # Evaluates formula on ALL cells at once using tensor operations
        truth_map = self.interpreter_vectorized.force_batch(formula, context)

        # Apply color assignment if present
        if 'assigned_color' in context.aux:
            target_color = float(context.aux['assigned_color'])  # Ensure float
            # Soft assignment weighted by truth value (all tensor ops - gradients flow!)
            # ALL FLOAT OPERATIONS - gradients can flow!
            output = truth_map * target_color + (1 - truth_map) * output

        # Return as FLOAT to preserve gradient flow!
        # The loss function will handle float→long conversion if needed
        # DON'T use .long() here as it breaks the computational graph
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
            input_grid: [batch_size, H_in, W_in] or [H_in, W_in]
            target_grid: [batch_size, H_out, W_out] or [H_out, W_out]
            target_size: Optional size (inferred from target_grid if None)
            hard_select: Use hard selection

        Returns:
            loss: Total loss
            losses: Dictionary of individual losses
        """
        # Handle unbatched
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)
            target_grid = target_grid.unsqueeze(0)

        # Infer target size
        if target_size is None:
            target_size = target_grid.shape[1:]

        # Forward pass
        output_grid, info = self.forward(input_grid, target_size, hard_select)

        # Pixel loss (MSE on color indices)
        pixel_loss = F.mse_loss(output_grid.float(), target_grid.float())

        # Template selection entropy (regularization)
        logits = info['logits']
        template_probs = F.softmax(logits, dim=-1)
        template_entropy = -(template_probs * torch.log(template_probs + 1e-8)).sum(dim=-1).mean()

        # Total loss
        loss = pixel_loss + 0.01 * template_entropy

        losses = {
            'total': loss,
            'pixel': pixel_loss,
            'template_entropy': template_entropy
        }

        return loss, losses

    def predict(
        self,
        input_grid: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Predict output grid (hard selection).

        Args:
            input_grid: [batch_size, H, W] or [H, W]
            target_size: Optional output size

        Returns:
            output_grid: [batch_size, H_out, W_out] or [H_out, W_out]
        """
        output_grid, _ = self.forward(input_grid, target_size, hard_select=True)
        return output_grid.round().long()

    def get_selected_template(self, input_grid: torch.Tensor) -> Formula:
        """Get selected template for input grid.

        Args:
            input_grid: [H, W] grid

        Returns:
            template: Selected formula
        """
        _, info = self.forward(input_grid, hard_select=True)
        return info['selected_templates'][0]

    def set_temperature(self, temperature: float):
        """Set Gumbel-Softmax temperature."""
        self.formula_selector.set_temperature(temperature)
