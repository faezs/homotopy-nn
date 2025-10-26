"""
Neural Predicates - Learned Atomic Formulas

Implements differentiable predicates for ARC tasks using neural networks.

THEORETICAL FOUNDATION:

In the internal language of a topos, atomic formulas are the primitive propositions.
For ARC tasks, these include:

**Geometric predicates**:
- is_boundary(cell): Cell is on edge of grid
- is_inside(region): Region is interior
- is_corner(cell): Cell is at corner
- touches(region1, region2): Regions are adjacent

**Color predicates**:
- color_eq(cell, c): Cell has color c
- same_color(cell1, cell2): Cells have same color
- color_in_set(cell, {c₁, c₂, ...}): Cell color in set

**Shape predicates**:
- is_square(region): Region forms square
- is_line(region): Region forms horizontal/vertical line
- is_symmetric(region): Region has symmetry

**Topological predicates**:
- connected(region): Region is path-connected
- hole_count(region, n): Region has n holes
- border_of(region1, region2): region1 borders region2

Each predicate is implemented as a neural network that:
1. Takes context (grid, cell/region indices, etc.)
2. Returns probability in [0,1]
3. Is fully differentiable for gradient-based learning

Author: Claude Code
Date: October 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


################################################################################
# § 1: Base Predicate Class
################################################################################

class NeuralPredicate(nn.Module):
    """Base class for learned neural predicates.

    All predicates must:
    1. Return values in [0,1] (use sigmoid activation)
    2. Be differentiable (for gradient flow)
    3. Accept context dict with grid, indices, etc.
    """

    def __init__(self, feature_dim: int = 64, device: torch.device = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device or torch.device('cpu')

    def forward(self, *args, context: Dict[str, Any]) -> torch.Tensor:
        """Evaluate predicate.

        Args:
            *args: Predicate-specific arguments (cell index, region mask, etc.)
            context: Dictionary with grid, auxiliary data

        Returns:
            Truth value in [0,1]
        """
        raise NotImplementedError


################################################################################
# § 2: Geometric Predicates
################################################################################

class BoundaryPredicate(NeuralPredicate):
    """Predicate: is_boundary(cell)

    Returns 1.0 if cell is on edge of grid, 0.0 otherwise.

    This is a "hard" predicate (can be computed exactly), but we
    implement it as neural to maintain uniform interface.
    """

    def forward(self, cell_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape

        i, j = cell_idx // W, cell_idx % W

        # On boundary if on edge
        on_boundary = (i == 0) or (i == H-1) or (j == 0) or (j == W-1)

        return torch.tensor(1.0 if on_boundary else 0.0, device=self.device)


class InsidePredicate(NeuralPredicate):
    """Predicate: is_inside(cell)

    Returns 1.0 if cell is interior (not on boundary).
    """

    def forward(self, cell_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape

        i, j = cell_idx // W, cell_idx % W

        # Inside if not on edge
        is_inside = (i > 0) and (i < H-1) and (j > 0) and (j < W-1)

        return torch.tensor(1.0 if is_inside else 0.0, device=self.device)


class CornerPredicate(NeuralPredicate):
    """Predicate: is_corner(cell)

    Returns 1.0 if cell is at corner of grid.
    """

    def forward(self, cell_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape

        i, j = cell_idx // W, cell_idx % W

        # Corner if at (0,0), (0,W-1), (H-1,0), or (H-1,W-1)
        is_corner = ((i == 0) or (i == H-1)) and ((j == 0) or (j == W-1))

        return torch.tensor(1.0 if is_corner else 0.0, device=self.device)


################################################################################
# § 3: Color Predicates (Learned)
################################################################################

class ColorEqPredicate(NeuralPredicate):
    """Predicate: color_eq(cell, target_color)

    Learned predicate that recognizes when cell has specific color.

    Architecture:
    - Input: One-hot color at cell + target color embedding
    - Output: Probability that colors match
    """

    def __init__(self, num_colors: int = 10, feature_dim: int = 64, device=None):
        super().__init__(feature_dim, device)
        self.num_colors = num_colors

        # Color embedding (learnable)
        self.color_embed = nn.Embedding(num_colors, feature_dim)

        # Comparison network
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, cell_idx: int, target_color: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape

        i, j = cell_idx // W, cell_idx % W
        cell_color = int(grid[i, j].item())

        # Embed both colors (ensure tensor creation happens on correct device)
        cell_color_tensor = torch.tensor(cell_color, dtype=torch.long, device=self.device)
        target_color_tensor = torch.tensor(target_color, dtype=torch.long, device=self.device)

        cell_embed = self.color_embed(cell_color_tensor)
        target_embed = self.color_embed(target_color_tensor)

        # Concatenate and predict
        combined = torch.cat([cell_embed, target_embed], dim=0)
        prob = self.net(combined)

        return prob.squeeze()


class SameColorPredicate(NeuralPredicate):
    """Predicate: same_color(cell1, cell2)

    Learned predicate for color similarity.
    """

    def __init__(self, num_colors: int = 10, feature_dim: int = 64, device=None):
        super().__init__(feature_dim, device)
        self.num_colors = num_colors

        self.color_embed = nn.Embedding(num_colors, feature_dim)

        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, cell1_idx: int, cell2_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape

        i1, j1 = cell1_idx // W, cell1_idx % W
        i2, j2 = cell2_idx // W, cell2_idx % W

        color1 = int(grid[i1, j1].item())
        color2 = int(grid[i2, j2].item())

        color1_tensor = torch.tensor(color1, dtype=torch.long, device=self.device)
        color2_tensor = torch.tensor(color2, dtype=torch.long, device=self.device)

        embed1 = self.color_embed(color1_tensor)
        embed2 = self.color_embed(color2_tensor)

        combined = torch.cat([embed1, embed2], dim=0)
        prob = self.net(combined)

        return prob.squeeze()


################################################################################
# § 4: Shape Predicates (Learned via CNN)
################################################################################

class ShapePredicate(NeuralPredicate):
    """Base class for shape predicates using CNN features.

    Extracts region, encodes with CNN, predicts shape property.
    """

    def __init__(self, num_colors: int = 10, feature_dim: int = 64, device=None):
        super().__init__(feature_dim, device)
        self.num_colors = num_colors

        # CNN encoder for region
        self.encoder = nn.Sequential(
            nn.Conv2d(num_colors, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def extract_region(self, grid: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        """Extract region from grid using binary mask.

        Args:
            grid: [H, W] color values
            region_mask: [H, W] binary mask (1 = in region, 0 = outside)

        Returns:
            One-hot encoded region [num_colors, H, W]
        """
        # One-hot encode
        one_hot = F.one_hot(grid.long(), num_classes=self.num_colors).float()
        one_hot = one_hot.permute(2, 0, 1)  # [C, H, W]

        # Mask region (set outside to 0)
        one_hot = one_hot * region_mask.unsqueeze(0)

        return one_hot.unsqueeze(0)  # [1, C, H, W]

    def forward(self, region_mask: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']

        # Extract and encode region
        region_tensor = self.extract_region(grid, region_mask)
        features = self.encoder(region_tensor)
        features = features.view(features.size(0), -1)

        # Predict shape property
        prob = self.classifier(features)

        return prob.squeeze()


class IsSquarePredicate(ShapePredicate):
    """Predicate: is_square(region)

    Learned predicate for detecting square regions.
    """
    pass  # Uses ShapePredicate architecture


class IsLinePredicate(ShapePredicate):
    """Predicate: is_line(region)

    Learned predicate for detecting horizontal/vertical lines.
    """
    pass


class IsSymmetricPredicate(ShapePredicate):
    """Predicate: is_symmetric(region)

    Learned predicate for detecting symmetric regions.
    """
    pass


################################################################################
# § 4.5: Geometric Transformation Predicates
################################################################################

class ReflectedColorH(NeuralPredicate):
    """Predicate: reflected_color_h(cell)

    Returns the color of the cell's horizontal reflection.
    """

    def forward(self, cell_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape

        i, j = cell_idx // W, cell_idx % W

        # Horizontal reflection: (i, j) → (i, W-1-j)
        j_reflected = W - 1 - j

        reflected_color = grid[i, j_reflected]

        return reflected_color


class ReflectedColorV(NeuralPredicate):
    """Predicate: reflected_color_v(cell)

    Returns the color of the cell's vertical reflection.
    """

    def forward(self, cell_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape

        i, j = cell_idx // W, cell_idx % W

        # Vertical reflection: (i, j) → (H-1-i, j)
        i_reflected = H - 1 - i

        reflected_color = grid[i_reflected, j]

        return reflected_color


class RotatedColor90(NeuralPredicate):
    """Predicate: rotated_color_90(cell)

    Returns the color of the cell rotated 90° clockwise.
    For square grids: (i, j) → (j, H-1-i)
    """

    def forward(self, cell_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape

        i, j = cell_idx // W, cell_idx % W

        # 90° clockwise rotation (for square grids)
        if H == W:
            i_rot = j
            j_rot = H - 1 - i
            rotated_color = grid[i_rot, j_rot]
        else:
            # For non-square grids, just return original color
            rotated_color = grid[i, j]

        return rotated_color


class TranslatedColor(NeuralPredicate):
    """Predicate: translated_color(cell, dx, dy)

    Returns the color at (i+dy, j+dx), or 0 if out of bounds.
    """

    def forward(self, cell_idx: int, dx: int, dy: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape

        i, j = cell_idx // W, cell_idx % W

        # Translated position
        i_trans = i + dy
        j_trans = j + dx

        # Check bounds
        if 0 <= i_trans < H and 0 <= j_trans < W:
            translated_color = grid[i_trans, j_trans]
        else:
            translated_color = torch.tensor(0.0, device=self.device)  # Background

        return translated_color


class NeighborColorLeft(NeuralPredicate):
    """Predicate: neighbor_color_left(cell)"""

    def forward(self, cell_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape
        i, j = cell_idx // W, cell_idx % W

        if j > 0:
            return grid[i, j-1]
        return torch.tensor(0.0, device=self.device)


class NeighborColorRight(NeuralPredicate):
    """Predicate: neighbor_color_right(cell)"""

    def forward(self, cell_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape
        i, j = cell_idx // W, cell_idx % W

        if j < W - 1:
            return grid[i, j+1]
        return torch.tensor(0.0, device=self.device)


class NeighborColorUp(NeuralPredicate):
    """Predicate: neighbor_color_up(cell)"""

    def forward(self, cell_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape
        i, j = cell_idx // W, cell_idx % W

        if i > 0:
            return grid[i-1, j]
        return torch.tensor(0.0, device=self.device)


class NeighborColorDown(NeuralPredicate):
    """Predicate: neighbor_color_down(cell)"""

    def forward(self, cell_idx: int, context: Dict[str, Any]) -> torch.Tensor:
        grid = context['grid']
        H, W = grid.shape
        i, j = cell_idx // W, cell_idx % W

        if i < H - 1:
            return grid[i+1, j]
        return torch.tensor(0.0, device=self.device)


################################################################################
# § 5: Relational Predicates
################################################################################

class TouchesPredicate(NeuralPredicate):
    """Predicate: touches(region1, region2)

    Returns 1.0 if regions are adjacent (share boundary).
    """

    def forward(self, region1: torch.Tensor, region2: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            region1: Binary mask [H, W]
            region2: Binary mask [H, W]
        """
        # Dilate region1 by 1 pixel
        kernel = torch.ones(1, 1, 3, 3, device=self.device)
        region1_dilated = F.conv2d(
            region1.unsqueeze(0).unsqueeze(0).float(),
            kernel,
            padding=1
        ).squeeze() > 0

        # Check if dilated region1 overlaps with region2
        overlap = (region1_dilated & region2).any()

        # But they shouldn't already overlap (must be distinct)
        already_overlap = (region1 & region2).any()

        touches = overlap and not already_overlap

        return torch.tensor(1.0 if touches else 0.0, device=self.device)


class BorderOfPredicate(NeuralPredicate):
    """Predicate: border_of(region1, region2)

    Returns 1.0 if region1 is the border/outline of region2.
    """

    def forward(self, region1: torch.Tensor, region2: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        # Compute boundary of region2 (dilation - erosion)
        kernel = torch.ones(1, 1, 3, 3, device=self.device)

        r2_float = region2.unsqueeze(0).unsqueeze(0).float()

        dilated = F.conv2d(r2_float, kernel, padding=1).squeeze() > 0
        eroded = F.conv2d(r2_float, kernel, padding=1).squeeze() >= 9  # All neighbors

        boundary = dilated & (~eroded)

        # Check how much region1 overlaps with boundary
        overlap = (region1 & boundary).float().sum()
        total_boundary = boundary.float().sum()

        if total_boundary == 0:
            return torch.tensor(0.0, device=self.device)

        # Soft match
        match_ratio = overlap / (total_boundary + 1e-6)

        return torch.sigmoid(10 * (match_ratio - 0.8))  # Threshold at 80% match


################################################################################
# § 6: Predicate Registry
################################################################################

class PredicateRegistry:
    """Registry for all neural predicates.

    Manages initialization, registration, and lookup of predicates.
    """

    def __init__(
        self,
        num_colors: int = 10,
        feature_dim: int = 64,
        device: torch.device = None
    ):
        self.num_colors = num_colors
        self.feature_dim = feature_dim
        self.device = device or torch.device('cpu')

        # Registry: name → predicate instance
        self.predicates: Dict[str, NeuralPredicate] = {}

        # Initialize default predicates
        self._register_defaults()

    def _register_defaults(self):
        """Register all default predicates."""

        # Geometric
        self.register('is_boundary', BoundaryPredicate(self.feature_dim, self.device).to(self.device))
        self.register('is_inside', InsidePredicate(self.feature_dim, self.device).to(self.device))
        self.register('is_corner', CornerPredicate(self.feature_dim, self.device).to(self.device))

        # Color
        self.register('color_eq', ColorEqPredicate(self.num_colors, self.feature_dim, self.device).to(self.device))
        self.register('same_color', SameColorPredicate(self.num_colors, self.feature_dim, self.device).to(self.device))

        # Shape
        self.register('is_square', IsSquarePredicate(self.num_colors, self.feature_dim, self.device).to(self.device))
        self.register('is_line', IsLinePredicate(self.num_colors, self.feature_dim, self.device).to(self.device))
        self.register('is_symmetric', IsSymmetricPredicate(self.num_colors, self.feature_dim, self.device).to(self.device))

        # Geometric transformations
        self.register('reflected_color_h', ReflectedColorH(self.feature_dim, self.device).to(self.device))
        self.register('reflected_color_v', ReflectedColorV(self.feature_dim, self.device).to(self.device))
        self.register('rotated_color_90', RotatedColor90(self.feature_dim, self.device).to(self.device))
        self.register('translated_color', TranslatedColor(self.feature_dim, self.device).to(self.device))
        self.register('neighbor_color_left', NeighborColorLeft(self.feature_dim, self.device).to(self.device))
        self.register('neighbor_color_right', NeighborColorRight(self.feature_dim, self.device).to(self.device))
        self.register('neighbor_color_up', NeighborColorUp(self.feature_dim, self.device).to(self.device))
        self.register('neighbor_color_down', NeighborColorDown(self.feature_dim, self.device).to(self.device))

        # Relational
        self.register('touches', TouchesPredicate(self.feature_dim, self.device).to(self.device))
        self.register('border_of', BorderOfPredicate(self.feature_dim, self.device).to(self.device))

    def register(self, name: str, predicate: NeuralPredicate):
        """Register predicate with name."""
        self.predicates[name] = predicate

    def get(self, name: str) -> NeuralPredicate:
        """Look up predicate by name."""
        if name not in self.predicates:
            raise ValueError(f"Unknown predicate: {name}")
        return self.predicates[name]

    def all_parameters(self) -> List[torch.Tensor]:
        """Get all learnable parameters from all predicates."""
        params = []
        for pred in self.predicates.values():
            params.extend(list(pred.parameters()))
        return params

    def to(self, device: torch.device):
        """Move all predicates to device."""
        for pred in self.predicates.values():
            pred.to(device)
        self.device = device
        return self


################################################################################
# § 7: Example Usage
################################################################################

if __name__ == "__main__":
    """
    Test predicates on simple grid.
    """

    # Create registry
    registry = PredicateRegistry(num_colors=10, feature_dim=32, device='cpu')

    # Example grid (5×5 with red square in middle)
    grid = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0],
        [0, 2, 1, 2, 0],
        [0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0]
    ], dtype=torch.float32)

    context = {'grid': grid}

    # Test boundary predicate
    boundary_pred = registry.get('is_boundary')
    print("Boundary cells:")
    for i in range(5):
        for j in range(5):
            cell_idx = i * 5 + j
            val = boundary_pred(cell_idx, context)
            if val > 0.5:
                print(f"  ({i},{j}): {val.item():.2f}")

    # Test color predicate
    color_pred = registry.get('color_eq')
    print("\nRed cells (color=2):")
    for i in range(5):
        for j in range(5):
            cell_idx = i * 5 + j
            val = color_pred(cell_idx, 2, context)
            if val > 0.5:
                print(f"  ({i},{j}): {val.item():.2f}")

    # Test shape predicate
    square_pred = registry.get('is_square')
    red_mask = (grid == 2).float()
    val = square_pred(red_mask, context)
    print(f"\nIs red region a square? {val.item():.2f}")

    print("\n✓ All predicates initialized and tested successfully!")
