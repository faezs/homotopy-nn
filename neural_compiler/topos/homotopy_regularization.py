"""
Homotopy Theory for Geometric Morphism Learning

MATHEMATICAL FRAMEWORK:
  - Morphism space: Mor(Sh(X), Sh(Y)) forms a topological space
  - Homotopy H: f ≃ g is continuous deformation f_0 → f_1
  - Fundamental group π₁(Mor) captures equivalence classes of transformations
  - Van Kampen theorem: compose transformations topologically

IMPLEMENTATION:
  1. Homotopy regularization: Penalize non-smooth morphisms (∇_θ smoothness)
  2. Path interpolation: Enforce geometric consistency along paths
  3. Topological constraints: Preserve connectivity and continuity

REFERENCES:
  - Neural/Homotopy/VanKampen.agda (formal proof in codebase)
  - Belfiore & Bennequin 2022 (topos-theoretic DNNs)
  - Algebraic topology (Hatcher, 2002)

Author: Claude Code + Human
Date: October 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


################################################################################
# § 1: Homotopy Regularization (Smoothness in Parameter Space)
################################################################################

class HomotopyRegularization(nn.Module):
    """Regularize geometric morphism to be smooth (continuous deformation).

    MATHEMATICAL PRINCIPLE:
      A homotopy H: [0,1] × X → Y between morphisms f₀, f₁ should be continuous.
      In parameter space θ, this means:
        ||∇_θ f_θ||² should be bounded (smooth variation)

    IMPLEMENTATION:
      Penalize large parameter gradients → smooth morphism family
    """

    def __init__(self, lambda_smooth: float = 0.01):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute smoothness penalty on model parameters.

        Args:
            model: Neural network representing geometric morphism

        Returns:
            loss: Smoothness regularization loss
        """
        # Compute gradient norm penalty
        # This encourages smooth deformations in parameter space
        param_norms = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param_norms.append(torch.norm(param.grad))

        if len(param_norms) == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        # Smoothness penalty: penalize large gradients
        smoothness_loss = sum(param_norms) / len(param_norms)

        return self.lambda_smooth * smoothness_loss


################################################################################
# § 2: Path Interpolation (Geometric Consistency)
################################################################################

class PathInterpolationLoss(nn.Module):
    """Enforce that interpolated inputs map to interpolated outputs.

    MATHEMATICAL PRINCIPLE:
      Linear homotopy between examples (x₁, y₁) and (x₂, y₂):
        H_t(x) = (1-t)·f(x₁) + t·f(x₂)  for t ∈ [0,1]

      Geometric consistency:
        f(x_t) ≈ H_t(x)  where x_t = (1-t)·x₁ + t·x₂

      This enforces that the morphism respects linear structure.

    CATEGORICAL INTERPRETATION:
      The sheaf morphism should be a natural transformation,
      which means it commutes with all morphisms in the base category.
      Path interpolation is a weak form of naturality.
    """

    def __init__(self, num_interpolation_points: int = 3, lambda_interp: float = 0.1):
        super().__init__()
        self.num_points = num_interpolation_points
        self.lambda_interp = lambda_interp

    def forward(self,
                model: nn.Module,
                sheaf_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Compute path interpolation loss between example pairs.

        Args:
            model: Geometric morphism (callable: sheaf_in → sheaf_out)
            sheaf_pairs: List of (input_sheaf, output_sheaf) pairs

        Returns:
            loss: Path interpolation consistency loss
        """
        if len(sheaf_pairs) < 2:
            return torch.tensor(0.0, device=sheaf_pairs[0][0].device)

        total_loss = 0.0
        num_pairs = 0

        # Sample pairs of examples
        for i in range(len(sheaf_pairs) - 1):
            x1, y1 = sheaf_pairs[i]
            x2, y2 = sheaf_pairs[i + 1]

            # Sample points along linear path
            for t in torch.linspace(0.1, 0.9, self.num_points):
                # Interpolated input
                x_t = (1 - t) * x1 + t * x2

                # Expected output (linear interpolation in output space)
                y_t_expected = (1 - t) * y1 + t * y2

                # Actual output from morphism
                y_t_actual = model(x_t)

                # Consistency loss
                total_loss += F.mse_loss(y_t_actual, y_t_expected)
                num_pairs += 1

        if num_pairs == 0:
            return torch.tensor(0.0, device=sheaf_pairs[0][0].device)

        return self.lambda_interp * total_loss / num_pairs


################################################################################
# § 3: Topological Consistency (Connectivity Preservation)
################################################################################

class TopologicalConsistencyLoss(nn.Module):
    """Preserve topological structure (connectivity, holes) through morphism.

    MATHEMATICAL PRINCIPLE:
      A good geometric morphism should preserve:
        - Connected components (π₀)
        - Fundamental group (π₁) [future work]
        - Homology groups (H_n) [future work]

      Current implementation: Preserve local connectivity via Lipschitz constraint.

    LIPSCHITZ CONSTRAINT:
      ||f(x) - f(y)|| ≤ L·||x - y||
      Ensures continuous morphism (no tearing/gluing)
    """

    def __init__(self, lipschitz_bound: float = 10.0, lambda_topo: float = 0.05):
        super().__init__()
        self.lipschitz_bound = lipschitz_bound
        self.lambda_topo = lambda_topo

    def forward(self,
                model: nn.Module,
                sheaf_inputs: List[torch.Tensor],
                sheaf_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Enforce Lipschitz continuity (soft constraint).

        Args:
            model: Geometric morphism
            sheaf_inputs: List of input sheaves
            sheaf_outputs: Corresponding output sheaves (predicted)

        Returns:
            loss: Lipschitz violation penalty
        """
        if len(sheaf_inputs) < 2:
            return torch.tensor(0.0, device=sheaf_inputs[0].device)

        violations = []

        # Check Lipschitz constraint for pairs
        for i in range(len(sheaf_inputs) - 1):
            x1, y1 = sheaf_inputs[i], sheaf_outputs[i]
            x2, y2 = sheaf_inputs[i + 1], sheaf_outputs[i + 1]

            # Distances
            input_dist = torch.norm(x1 - x2)
            output_dist = torch.norm(y1 - y2)

            # Lipschitz ratio
            if input_dist > 1e-6:
                lipschitz_ratio = output_dist / input_dist

                # Penalize if exceeds bound
                if lipschitz_ratio > self.lipschitz_bound:
                    violation = (lipschitz_ratio - self.lipschitz_bound) ** 2
                    violations.append(violation)

        if len(violations) == 0:
            return torch.tensor(0.0, device=sheaf_inputs[0].device)

        return self.lambda_topo * sum(violations) / len(violations)


################################################################################
# § 4: Combined Homotopy Loss
################################################################################

class HomotopyLoss(nn.Module):
    """Combined homotopy-theoretic loss for geometric morphism learning.

    COMPONENTS:
      1. Smoothness: ∇_θ regularization (continuous in parameter space)
      2. Path interpolation: f(x_t) ≈ y_t (geometric consistency)
      3. Topological: Lipschitz continuity (preserve structure)

    USAGE:
      homotopy_loss = HomotopyLoss()
      loss = homotopy_loss(
          model=geometric_morphism,
          sheaf_pairs=[(x1, y1), (x2, y2), ...],
          predicted_outputs=[f(x1), f(x2), ...]
      )
    """

    def __init__(self,
                 lambda_smooth: float = 0.01,
                 lambda_interp: float = 0.1,
                 lambda_topo: float = 0.05,
                 num_interp_points: int = 3):
        super().__init__()

        self.smoothness = HomotopyRegularization(lambda_smooth=lambda_smooth)
        self.interpolation = PathInterpolationLoss(
            num_interpolation_points=num_interp_points,
            lambda_interp=lambda_interp
        )
        self.topology = TopologicalConsistencyLoss(lambda_topo=lambda_topo)

    def forward(self,
                model: nn.Module,
                sheaf_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                predicted_outputs: List[torch.Tensor]) -> dict:
        """Compute all homotopy losses.

        Args:
            model: Geometric morphism network
            sheaf_pairs: List of (input_sheaf, target_sheaf) training pairs
            predicted_outputs: Model predictions for inputs

        Returns:
            losses: Dictionary with breakdown
        """
        # Extract inputs for topology loss
        sheaf_inputs = [pair[0] for pair in sheaf_pairs]

        # Compute components
        smooth_loss = self.smoothness(model)
        interp_loss = self.interpolation(model.pushforward, sheaf_pairs)
        topo_loss = self.topology(model.pushforward, sheaf_inputs, predicted_outputs)

        total = smooth_loss + interp_loss + topo_loss

        return {
            'total': total,
            'smoothness': smooth_loss,
            'interpolation': interp_loss,
            'topological': topo_loss
        }


################################################################################
# § 5: Utility Functions
################################################################################

def compute_fundamental_group_features(sheaf_sections: torch.Tensor) -> torch.Tensor:
    """Compute topological features (simplified π₁ approximation).

    FUTURE WORK:
      - Implement persistent homology (ripser)
      - Compute Betti numbers
      - Extract fundamental group generators

    CURRENT:
      Placeholder - returns connectivity statistics

    Args:
        sheaf_sections: (B, C, H, W) sheaf sections

    Returns:
        features: (B, num_features) topological features
    """
    B, C, H, W = sheaf_sections.shape

    # Simple connectivity measures (placeholder for true π₁)
    features = []

    for i in range(B):
        section = sheaf_sections[i]  # (C, H, W)

        # Measure 1: Average magnitude
        avg_mag = section.abs().mean()

        # Measure 2: Variance (spread)
        variance = section.var()

        # Measure 3: Gradient magnitude (local connectivity)
        dx = section[:, :, 1:] - section[:, :, :-1]
        dy = section[:, 1:, :] - section[:, :-1, :]
        grad_mag = (dx.pow(2).mean() + dy.pow(2).mean()).sqrt()

        features.append(torch.stack([avg_mag, variance, grad_mag]))

    return torch.stack(features)  # (B, 3)


################################################################################
# § 6: Example Usage
################################################################################

if __name__ == "__main__":
    print("=" * 70)
    print("Homotopy Regularization for Geometric Morphisms")
    print("=" * 70)
    print()

    # Example: dummy geometric morphism
    class DummyMorphism(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1)
            )

        def pushforward(self, x):
            return self.net(x)

    model = DummyMorphism()

    # Dummy data
    B, C, H, W = 2, 128, 10, 10
    x1 = torch.randn(B, C, H, W)
    y1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)
    y2 = torch.randn(B, C, H, W)

    sheaf_pairs = [(x1, y1), (x2, y2)]
    predicted = [model.pushforward(x1), model.pushforward(x2)]

    # Compute homotopy loss
    homotopy_loss = HomotopyLoss(
        lambda_smooth=0.01,
        lambda_interp=0.1,
        lambda_topo=0.05
    )

    losses = homotopy_loss(model, sheaf_pairs, predicted)

    print("Homotopy Loss Components:")
    print(f"  Total: {losses['total'].item():.6f}")
    print(f"  Smoothness: {losses['smoothness'].item():.6f}")
    print(f"  Interpolation: {losses['interpolation'].item():.6f}")
    print(f"  Topological: {losses['topological'].item():.6f}")
    print()

    # Topological features
    features = compute_fundamental_group_features(x1)
    print(f"Topological features shape: {features.shape}")
    print(f"Features: {features}")
    print()

    print("✓ Homotopy regularization components ready!")
    print("=" * 70)
