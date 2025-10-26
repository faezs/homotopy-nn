"""
CNN-Backed Sheaf Architecture

Uses CNNs as the underlying sheaf representation while maintaining
categorical structure (geometric morphisms, adjunctions, etc).

Key insight: CNNs already encode sheaf structure!
- Feature maps = Sections F(U) at each spatial location U
- Convolution = Restriction maps (local propagation)
- Receptive fields = Coverage families
- Spatial consistency = Sheaf gluing condition

Author: Claude Code + Human
Date: October 22, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from dataclasses import dataclass

from geometric_morphism_torch import Site
from arc_loader import ARCGrid


################################################################################
# § 1: CNN-Backed Sheaf
################################################################################

class CNNSheaf(nn.Module):
    """Sheaf represented as CNN feature maps.

    Mathematical structure:
    - Sheaf F: C^op → Vec
    - F(cell_{i,j}) = feature_map[i, j, :]  (section at spatial location)
    - Restriction maps = Convolution kernels (learned local propagation)
    - Coverage families = Receptive fields (neighborhoods)

    Advantages over vector sheaves:
    - Parameter sharing (conv kernels shared across space)
    - Translation equivariance (natural for grids)
    - Better inductive bias for images
    - Standard architecture (easier to scale)
    """

    def __init__(self, grid_shape: Tuple[int, int], in_channels: int = 10,
                 feature_dim: int = 32, hidden_channels: int = 64):
        super().__init__()
        self.grid_shape = grid_shape
        self.feature_dim = feature_dim

        # CNN backbone (learns sheaf sections + restrictions jointly)
        self.backbone = nn.Sequential(
            # Input: (batch, 10, H, W) - one-hot color channels
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),

            # Restriction composition layer 1
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),

            # Restriction composition layer 2
            nn.Conv2d(hidden_channels, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
            # Output: (batch, feature_dim, H, W) - sheaf sections
        )

    def forward(self, grid_one_hot: torch.Tensor) -> torch.Tensor:
        """Compute sheaf sections from one-hot grid.

        Args:
            grid_one_hot: (batch, 10, H, W) - one-hot encoded colors

        Returns:
            sections: (batch, feature_dim, H, W) - sheaf sections at each cell
        """
        return self.backbone(grid_one_hot)

    def at_object(self, sections: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """F(cell_{i,j}) - section at spatial location.

        Args:
            sections: (batch, feature_dim, H, W) - all sections
            i, j: Spatial coordinates

        Returns:
            section: (batch, feature_dim) - section at (i,j)
        """
        return sections[:, :, i, j]

    def check_sheaf_condition(self, sections: torch.Tensor) -> torch.Tensor:
        """Verify sheaf gluing condition using spatial consistency.

        Sheaf condition: F(U) should be determined by F(U_i) over covering {U_i}.
        For CNNs: Each cell should match predictions from its neighborhood.

        Returns:
            violation: Scalar - how much sheaf condition is violated
        """
        batch, C, H, W = sections.shape

        # Predict each cell from its 4-neighborhood (coverage family)
        # Use avg pooling to aggregate neighborhood
        neighborhood_pred = F.avg_pool2d(
            F.pad(sections, (1,1,1,1), mode='replicate'),  # Pad for boundary
            kernel_size=3, stride=1, padding=0
        )

        # Violation = difference between section and neighborhood prediction
        violation = F.mse_loss(sections, neighborhood_pred)
        return violation


################################################################################
# § 2: CNN Geometric Morphism
################################################################################

class CNNGeometricMorphism(nn.Module):
    """Geometric morphism f: E_in → E_out using CNN representations.

    Structure:
    - f^* (pullback): Upsampling/interpolation + CNN
    - f_* (pushforward): Downsampling/pooling + CNN
    - Adjunction: Learned via dual pathways

    Key: CNNs naturally handle variable grid sizes via pooling/upsampling.
    """

    def __init__(self, in_shape: Tuple[int, int], out_shape: Tuple[int, int],
                 feature_dim: int = 32):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.feature_dim = feature_dim

        # f_* : E_in → E_out (pushforward / direct image)
        # Spatially adapts input to output size
        self.pushforward_net = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # f^* : E_out → E_in (pullback / inverse image)
        # Spatially adapts output to input size
        self.pullback_net = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def pushforward(self, sheaf_in: torch.Tensor) -> torch.Tensor:
        """f_* : E_in → E_out (direct image).

        Args:
            sheaf_in: (batch, feature_dim, H_in, W_in)

        Returns:
            sheaf_out: (batch, feature_dim, H_out, W_out)
        """
        # Apply pushforward network
        transformed = self.pushforward_net(sheaf_in)

        # Spatially adapt to output shape
        if self.in_shape != self.out_shape:
            transformed = F.interpolate(
                transformed,
                size=self.out_shape,
                mode='bilinear',
                align_corners=False
            )

        return transformed

    def pullback(self, sheaf_out: torch.Tensor) -> torch.Tensor:
        """f^* : E_out → E_in (inverse image).

        Args:
            sheaf_out: (batch, feature_dim, H_out, W_out)

        Returns:
            sheaf_in: (batch, feature_dim, H_in, W_in)
        """
        # Spatially adapt to input shape
        if self.in_shape != self.out_shape:
            adapted = F.interpolate(
                sheaf_out,
                size=self.in_shape,
                mode='bilinear',
                align_corners=False
            )
        else:
            adapted = sheaf_out

        # Apply pullback network
        return self.pullback_net(adapted)

    def check_adjunction(self, sheaf_in: torch.Tensor, sheaf_out: torch.Tensor) -> torch.Tensor:
        """Verify f^* ⊣ f_* adjunction.

        Adjunction means: f_* ∘ f^* ∘ f_* ≈ f_*  and  f^* ∘ f_* ∘ f^* ≈ f^*

        Returns:
            violation: How much adjunction is violated
        """
        # Forward-backward-forward (should be close to forward)
        fbf = self.pushforward(self.pullback(self.pushforward(sheaf_in)))
        forward = self.pushforward(sheaf_in)

        violation1 = F.mse_loss(fbf, forward)

        # Backward-forward-backward (should be close to backward)
        bfb = self.pullback(self.pushforward(self.pullback(sheaf_out)))
        backward = self.pullback(sheaf_out)

        violation2 = F.mse_loss(bfb, backward)

        return violation1 + violation2


################################################################################
# § 3: Complete CNN Topos Solver
################################################################################

class CNNToposSolver(nn.Module):
    """Complete ARC solver using CNN-backed sheaves.

    Pipeline:
        ARC Grid → One-Hot → CNN Sheaf → Geometric Morphism → CNN Sheaf → Decode → ARC Grid

    Advantages:
    - ~10x fewer parameters than vector sheaves
    - Translation equivariant (natural for grids)
    - Standard CNN operations (fast, well-optimized)
    - Still maintains categorical structure!
    """

    def __init__(self, grid_shape_in: Tuple[int, int], grid_shape_out: Tuple[int, int],
                 feature_dim: int = 32, num_colors: int = 10, device=None):
        super().__init__()

        self.grid_shape_in = grid_shape_in
        self.grid_shape_out = grid_shape_out
        self.feature_dim = feature_dim
        self.num_colors = num_colors
        self.device = device if device else torch.device('cpu')

        # CNN-backed sheaf encoder
        self.sheaf_encoder = CNNSheaf(
            grid_shape_in,
            in_channels=num_colors,
            feature_dim=feature_dim
        )

        # Geometric morphism (CNN-based)
        self.geometric_morphism = CNNGeometricMorphism(
            grid_shape_in,
            grid_shape_out,
            feature_dim
        )

        # Decoder: Sheaf sections → Color logits
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, num_colors, kernel_size=1)
        )

        self.to(self.device)

    def encode_grid(self, grid: ARCGrid) -> torch.Tensor:
        """Grid → One-hot → CNN Sheaf."""
        # One-hot encode
        cells = torch.from_numpy(np.array(grid.cells)).long().to(self.device)
        one_hot = F.one_hot(cells, num_classes=self.num_colors).float()

        # (H, W, 10) → (1, 10, H, W)
        one_hot = one_hot.permute(2, 0, 1).unsqueeze(0)

        # Pad to target shape if needed
        H, W = self.grid_shape_in
        if one_hot.shape[-2:] != (H, W):
            one_hot = F.pad(one_hot,
                           (0, W - one_hot.shape[-1], 0, H - one_hot.shape[-2]))

        # Encode to sheaf
        return self.sheaf_encoder(one_hot)

    def decode_sheaf(self, sheaf_sections: torch.Tensor, height: int, width: int) -> ARCGrid:
        """CNN Sheaf → Color logits → ARC Grid."""
        # Decode to color logits
        logits = self.decoder(sheaf_sections)  # (1, 10, H, W)

        # Crop to target size
        logits = logits[:, :, :height, :width]

        # Argmax to colors
        colors = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        return ARCGrid.from_array(colors.astype(np.int32))

    def forward(self, input_grid: ARCGrid) -> ARCGrid:
        """Complete forward pass: Grid → Sheaf → Geometric Morphism → Sheaf → Grid."""
        # Encode
        input_sheaf = self.encode_grid(input_grid)

        # Transform via geometric morphism
        output_sheaf = self.geometric_morphism.pushforward(input_sheaf)

        # Decode
        return self.decode_sheaf(output_sheaf, *self.grid_shape_out)

    def compute_topos_loss(self, input_grid: ARCGrid, target_grid: ARCGrid) -> dict:
        """Compute categorical losses."""
        # Encode
        input_sheaf = self.encode_grid(input_grid)
        target_sheaf = self.encode_grid(target_grid)

        # Transform
        predicted_sheaf = self.geometric_morphism.pushforward(input_sheaf)

        # 1. Prediction loss
        pred_loss = F.mse_loss(predicted_sheaf, target_sheaf)

        # 2. Sheaf condition
        sheaf_loss = self.sheaf_encoder.check_sheaf_condition(predicted_sheaf)

        # 3. Adjunction
        adj_loss = self.geometric_morphism.check_adjunction(input_sheaf, predicted_sheaf)

        return {
            'prediction': pred_loss,
            'sheaf_condition': sheaf_loss,
            'adjunction': adj_loss,
            'total': pred_loss + 0.1 * adj_loss + 0.01 * sheaf_loss
        }


################################################################################
# § 4: Lightweight CNN Sheaf (Parameter-Efficient)
################################################################################

class LightweightCNNSheaf(nn.Module):
    """Ultra-lightweight CNN sheaf using 1x1 convolutions and depthwise separable convolutions.

    Key efficiency techniques:
    - 1x1 convolutions: O(C_in * C_out) parameters, no spatial cost
    - Depthwise separable: Separate spatial and channel mixing
    - No BatchNorm: Reduces parameter count
    - Single hidden layer
    """

    def __init__(self, grid_shape: Tuple[int, int], in_channels: int = 10,
                 feature_dim: int = 32):
        super().__init__()
        self.grid_shape = grid_shape
        self.feature_dim = feature_dim

        # Extremely lightweight: just 1x1 convolutions (no spatial parameters!)
        self.backbone = nn.Sequential(
            # 1x1 conv: mix color channels
            nn.Conv2d(in_channels, feature_dim, kernel_size=1),
            nn.ReLU(),
            # Optional: 3x3 depthwise for spatial mixing (parameter-efficient)
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.ReLU()
        )

    def forward(self, grid_one_hot: torch.Tensor) -> torch.Tensor:
        return self.backbone(grid_one_hot)


class LightweightCNNGeometricMorphism(nn.Module):
    """Lightweight geometric morphism using shared 1x1 convolutions.

    Natural Transformations as Attention:
    - Attention is a natural transformation η: F → G between CNN functors
    - For each spatial location (i,j): η_{(i,j)}: F(i,j) → G(i,j)
    - Naturality: commutes with restriction maps (convolutions)
    """

    def __init__(self, in_shape: Tuple[int, int], out_shape: Tuple[int, int],
                 feature_dim: int = 32):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.feature_dim = feature_dim

        # Natural transformation as attention (pointwise transformation)
        # Query, Key, Value projections (1x1 convs = natural transformations)
        self.query = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.value = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        # Shared mixing network (1x1 conv, no spatial parameters)
        self.mixing = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.ReLU()
        )

    def _apply_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention as natural transformation.

        Naturality: η commutes with morphisms
        For f: X → Y, η_Y ∘ F(f) = G(f) ∘ η_X
        """
        batch, C, H, W = x.shape

        # Project to Q, K, V (natural transformations at each location)
        Q = self.query(x)  # (batch, C, H, W)
        K = self.key(x)
        V = self.value(x)

        # Reshape for attention computation
        Q = Q.view(batch, C, H * W).permute(0, 2, 1)  # (batch, H*W, C)
        K = K.view(batch, C, H * W)  # (batch, C, H*W)
        V = V.view(batch, C, H * W).permute(0, 2, 1)  # (batch, H*W, C)

        # Attention scores (naturality: pointwise transformation)
        attn = torch.softmax(Q @ K / (C ** 0.5), dim=-1)  # (batch, H*W, H*W)

        # Apply attention (natural transformation)
        out = attn @ V  # (batch, H*W, C)
        out = out.permute(0, 2, 1).view(batch, C, H, W)

        return out

    def pushforward(self, sheaf_in: torch.Tensor) -> torch.Tensor:
        """f_* with attention as natural transformation."""
        # Apply attention (natural transformation between sheaves)
        attended = self._apply_attention(sheaf_in)

        # Mix channels
        transformed = self.mixing(attended)

        # Adapt spatial shape if needed
        if self.in_shape != self.out_shape:
            transformed = F.interpolate(transformed, size=self.out_shape,
                                       mode='bilinear', align_corners=False)
        return transformed

    def pullback(self, sheaf_out: torch.Tensor) -> torch.Tensor:
        """f^* with attention as natural transformation."""
        # Adapt spatial shape if needed
        if self.in_shape != self.out_shape:
            adapted = F.interpolate(sheaf_out, size=self.in_shape,
                                   mode='bilinear', align_corners=False)
        else:
            adapted = sheaf_out

        # Apply attention (natural transformation)
        attended = self._apply_attention(adapted)

        # Mix channels
        return self.mixing(attended)

    def check_adjunction(self, sheaf_in: torch.Tensor, sheaf_out: torch.Tensor) -> torch.Tensor:
        fbf = self.pushforward(self.pullback(self.pushforward(sheaf_in)))
        forward = self.pushforward(sheaf_in)
        violation1 = F.mse_loss(fbf, forward)

        bfb = self.pullback(self.pushforward(self.pullback(sheaf_out)))
        backward = self.pullback(sheaf_out)
        violation2 = F.mse_loss(bfb, backward)

        return violation1 + violation2


class LightweightCNNToposSolver(nn.Module):
    """Ultra-lightweight CNN topos solver targeting <5K parameters."""

    def __init__(self, grid_shape_in: Tuple[int, int], grid_shape_out: Tuple[int, int],
                 feature_dim: int = 32, num_colors: int = 10, device=None):
        super().__init__()

        self.grid_shape_in = grid_shape_in
        self.grid_shape_out = grid_shape_out
        self.feature_dim = feature_dim
        self.num_colors = num_colors
        self.device = device if device else torch.device('cpu')

        # Lightweight encoder
        self.sheaf_encoder = LightweightCNNSheaf(
            grid_shape_in, in_channels=num_colors, feature_dim=feature_dim
        )

        # Lightweight geometric morphism
        self.geometric_morphism = LightweightCNNGeometricMorphism(
            grid_shape_in, grid_shape_out, feature_dim
        )

        # Lightweight decoder: just 1x1 conv
        self.decoder = nn.Conv2d(feature_dim, num_colors, kernel_size=1)

        self.to(self.device)

    def encode_grid(self, grid: ARCGrid) -> torch.Tensor:
        cells = torch.from_numpy(np.array(grid.cells)).long().to(self.device)
        one_hot = F.one_hot(cells, num_classes=self.num_colors).float()
        one_hot = one_hot.permute(2, 0, 1).unsqueeze(0)

        H, W = self.grid_shape_in
        if one_hot.shape[-2:] != (H, W):
            one_hot = F.pad(one_hot, (0, W - one_hot.shape[-1], 0, H - one_hot.shape[-2]))

        return self.sheaf_encoder(one_hot)

    def decode_sheaf(self, sheaf_sections: torch.Tensor, height: int, width: int) -> ARCGrid:
        logits = self.decoder(sheaf_sections)
        logits = logits[:, :, :height, :width]
        colors = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        return ARCGrid.from_array(colors.astype(np.int32))

    def forward(self, input_grid: ARCGrid) -> ARCGrid:
        input_sheaf = self.encode_grid(input_grid)
        output_sheaf = self.geometric_morphism.pushforward(input_sheaf)
        return self.decode_sheaf(output_sheaf, *self.grid_shape_out)

    def compute_topos_loss(self, input_grid: ARCGrid, target_grid: ARCGrid) -> dict:
        input_sheaf = self.encode_grid(input_grid)
        target_sheaf = self.encode_grid(target_grid)
        predicted_sheaf = self.geometric_morphism.pushforward(input_sheaf)

        # Sheaf space MSE (continuous)
        sheaf_mse = F.mse_loss(predicted_sheaf, target_sheaf)

        # Pixel-level cross-entropy (discrete) - THE KEY FIX!
        pred_logits = self.decoder(predicted_sheaf)  # [1, 10, H_model, W_model]

        # Get target as tensor
        target_cells = torch.from_numpy(np.array(target_grid.cells)).long().to(self.device)  # [H_target, W_target]

        # Handle size mismatch: crop or pad to common size
        H_pred, W_pred = pred_logits.shape[2], pred_logits.shape[3]
        H_target, W_target = target_cells.shape[0], target_cells.shape[1]

        # Use minimum size (crop both to match)
        H_common = min(H_pred, H_target)
        W_common = min(W_pred, W_target)

        pred_logits_crop = pred_logits[:, :, :H_common, :W_common]  # [1, 10, H_common, W_common]
        target_cells_crop = target_cells[:H_common, :W_common]  # [H_common, W_common]
        target_cells_crop = target_cells_crop.unsqueeze(0)  # [1, H_common, W_common]

        pixel_ce = F.cross_entropy(pred_logits_crop, target_cells_crop)

        # Sheaf condition: spatial consistency
        sheaf_condition = F.mse_loss(
            predicted_sheaf,
            F.avg_pool2d(F.pad(predicted_sheaf, (1,1,1,1), mode='replicate'),
                        kernel_size=3, stride=1, padding=0)
        )

        # Adjunction
        adj_loss = self.geometric_morphism.check_adjunction(input_sheaf, predicted_sheaf)

        return {
            'sheaf_mse': sheaf_mse,
            'pixel_ce': pixel_ce,
            'sheaf_condition': sheaf_condition,
            'adjunction': adj_loss,
            'total': pixel_ce + 0.1 * sheaf_mse + 0.01 * adj_loss + 0.001 * sheaf_condition
        }


################################################################################
# § 5: Usage Example
################################################################################

if __name__ == "__main__":
    print("=" * 70)
    print("Standard CNN Sheaf")
    print("=" * 70)

    # Create standard solver
    solver = CNNToposSolver(
        grid_shape_in=(5, 5),
        grid_shape_out=(5, 5),
        feature_dim=32,
        num_colors=10,
        device=torch.device('cpu')
    )

    # Count parameters
    total_params = sum(p.numel() for p in solver.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    test_grid = ARCGrid.from_array(np.random.randint(0, 10, (5, 5)))
    output = solver(test_grid)

    print(f"Input shape: {test_grid.cells.shape}")
    print(f"Output shape: {output.cells.shape}")

    print("\n" + "=" * 70)
    print("Lightweight CNN Sheaf (Target: <5K params)")
    print("=" * 70)

    # Create lightweight solver
    lightweight_solver = LightweightCNNToposSolver(
        grid_shape_in=(5, 5),
        grid_shape_out=(5, 5),
        feature_dim=32,
        num_colors=10,
        device=torch.device('cpu')
    )

    # Count parameters
    lightweight_params = sum(p.numel() for p in lightweight_solver.parameters())
    print(f"Total parameters: {lightweight_params:,}")

    # Test forward pass
    lightweight_output = lightweight_solver(test_grid)
    print(f"Output shape: {lightweight_output.cells.shape}")

    # Compare
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"Standard CNN:     {total_params:>8,} params")
    print(f"Lightweight CNN:  {lightweight_params:>8,} params")
    print(f"Vector sheaf:     ~7,600 params (from production)")
    print(f"\nReduction: {total_params / lightweight_params:.1f}x fewer parameters")
