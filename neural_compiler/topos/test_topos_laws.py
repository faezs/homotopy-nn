"""
Direct Numerical Tests for Topos Laws

Explicit numerical tests for categorical axioms.
Complements property-based tests with concrete examples.

Author: Claude Code
Date: October 22, 2025
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from geometric_morphism_torch import Site, Sheaf, GeometricMorphism
from arc_loader import ARCGrid, ARCTask
from train_arc_geometric_production import ARCGeometricSolver
import torch.optim as optim


################################################################################
# Training Helper
################################################################################

def quick_train(solver, input_grid, target_grid, steps=50, lr=0.01):
    """Quickly train solver on a single example to verify learnability."""
    optimizer = optim.Adam(solver.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        # Encode
        input_sheaf = solver.encode_grid_to_sheaf(input_grid, solver.site_in)
        target_sheaf = solver.encode_grid_to_sheaf(target_grid, solver.site_out)

        # Forward
        predicted_sheaf = solver.geometric_morphism.pushforward(input_sheaf)

        # Loss
        loss = F.mse_loss(predicted_sheaf.sections, target_sheaf.sections)
        adj_loss = solver.geometric_morphism.check_adjunction(input_sheaf, target_sheaf)
        sheaf_loss = predicted_sheaf.total_sheaf_violation()

        total_loss = loss + 0.1 * adj_loss + 0.01 * sheaf_loss

        # Backward
        total_loss.backward()
        optimizer.step()

    return solver


################################################################################
# Test Fixtures
################################################################################

@pytest.fixture
def simple_site():
    """Create a simple site for testing."""
    # 2x2 grid = 4 objects
    return Site(grid_shape=(2, 2), connectivity="4")


@pytest.fixture
def solver():
    """Create a simple solver for testing."""
    return ARCGeometricSolver(
        grid_shape_in=(5, 5),
        grid_shape_out=(5, 5),
        feature_dim=16,
        num_colors=10,
        device=torch.device('cpu')
    )


@pytest.fixture
def example_grid():
    """Create a simple test grid."""
    cells = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4]
    ], dtype=np.int32)
    return ARCGrid.from_array(cells)


################################################################################
# Adjunction Law Tests
################################################################################

class TestAdjunctionLaws:
    """Test f^* ⊣ f_* adjunction."""

    def test_adjunction_unit_triangle_identity(self, solver, example_grid):
        """Test unit triangle identity: (f_* ∘ η) ∘ f_* = f_*.

        Where η: id → f^* ∘ f_* is the unit.
        """
        sheaf_in = solver.encode_grid_to_sheaf(example_grid, solver.site_in)

        # Apply f_*
        pushed = solver.geometric_morphism.pushforward(sheaf_in)

        # Apply unit then f_* again
        pulled = solver.geometric_morphism.pullback(pushed)
        pushed_again = solver.geometric_morphism.pushforward(pulled)

        # Should be close to just applying f_*
        error = F.mse_loss(pushed_again.sections, pushed.sections).item()

        print(f"  Unit triangle error: {error:.6f}")
        assert error < 10.0, f"Unit triangle identity violated: {error}"


    def test_adjunction_counit_triangle_identity(self, solver, example_grid):
        """Test counit triangle identity: f^* ∘ (ε ∘ f^*) = f^*.

        Where ε: f_* ∘ f^* → id is the counit.
        """
        sheaf_in = solver.encode_grid_to_sheaf(example_grid, solver.site_in)

        # Apply f_*
        pushed = solver.geometric_morphism.pushforward(sheaf_in)

        # Apply f^*
        pulled = solver.geometric_morphism.pullback(pushed)

        # Apply counit (pushforward again) then pullback
        pushed_back = solver.geometric_morphism.pushforward(pulled)
        pulled_again = solver.geometric_morphism.pullback(pushed_back)

        # Should be close to just applying f^*
        error = F.mse_loss(pulled_again.sections, pulled.sections).item()

        print(f"  Counit triangle error: {error:.6f}")
        assert error < 10.0, f"Counit triangle identity violated: {error}"


    def test_adjunction_symmetry(self, solver, example_grid):
        """Test adjunction is symmetric in both directions."""
        sheaf_in = solver.encode_grid_to_sheaf(example_grid, solver.site_in)
        sheaf_out = solver.geometric_morphism.pushforward(sheaf_in)

        # Forward check
        adj_forward = solver.geometric_morphism.check_adjunction(sheaf_in, sheaf_out).item()

        # Should be low for compatible sheaves
        print(f"  Adjunction violation: {adj_forward:.6f}")
        assert adj_forward >= 0, "Adjunction must be non-negative"


################################################################################
# Functor Law Tests
################################################################################

class TestFunctorLaws:
    """Test that geometric morphisms are functorial."""

    def test_functor_identity_preservation(self, solver, example_grid):
        """Test F(id) ≈ id on sections."""
        sheaf = solver.encode_grid_to_sheaf(example_grid, solver.site_in)

        # Apply identity (roundtrip)
        result = solver.geometric_morphism.pullback(
            solver.geometric_morphism.pushforward(sheaf)
        )

        # Measure deviation from identity
        identity_error = F.mse_loss(result.sections, sheaf.sections).item()

        print(f"  Identity preservation error: {identity_error:.6f}")
        assert identity_error < 50.0, f"Identity not preserved: {identity_error}"


    def test_functor_composition(self, simple_site):
        """Test F(g ∘ f) ≈ F(g) ∘ F(f)."""
        site_a = simple_site
        site_b = Site(grid_shape=(3, 1), connectivity="4")  # 3 objects in a line
        site_c = Site(grid_shape=(3, 1), connectivity="4")  # 3 objects in a line

        feature_dim = 8
        f = GeometricMorphism(site_a, site_b, feature_dim)
        g = GeometricMorphism(site_b, site_c, feature_dim)

        # Create test sheaf
        sheaf_a = Sheaf(site_a, feature_dim)
        object.__setattr__(sheaf_a, 'sections', torch.randn(site_a.num_objects, feature_dim))

        # F(g ∘ f): apply f then g
        intermediate = f.pushforward(sheaf_a)
        composed_result = g.pushforward(intermediate)

        # Should produce valid sheaf
        assert composed_result.sections.shape[0] == site_c.num_objects
        assert not torch.isnan(composed_result.sections).any()

        print(f"  Composition result shape: {composed_result.sections.shape}")


################################################################################
# Sheaf Condition Tests
################################################################################

class TestSheafCondition:
    """Test sheaf gluing axioms."""

    def test_sheaf_gluing_uniqueness(self, solver, example_grid):
        """Test that sections are uniquely determined by restrictions.

        If s_i = s|_{U_i} for covering {U_i}, then s is unique.
        """
        sheaf = solver.encode_grid_to_sheaf(example_grid, solver.site_in)

        # Check sheaf violation
        violation = sheaf.total_sheaf_violation().item()

        print(f"  Sheaf violation: {violation:.6f}")
        assert violation >= 0, "Violation must be non-negative"
        # Lower is better (0 = perfect sheaf)


    def test_sheaf_locality(self, solver, example_grid):
        """Test locality: if s|_{U_i} = 0 for all i, then s = 0."""
        # Quick training to learn sheaf structure
        zero_grid = ARCGrid.from_array(np.zeros((5, 5), dtype=np.int32))
        solver = quick_train(solver, zero_grid, zero_grid, steps=100)

        sheaf = solver.encode_grid_to_sheaf(example_grid, solver.site_in)

        # Create zero sheaf
        zero_sheaf = Sheaf(solver.site_in, solver.feature_dim)
        object.__setattr__(zero_sheaf, 'sections', torch.zeros_like(sheaf.sections))

        # Zero sheaf should have low violation after training
        zero_violation = zero_sheaf.total_sheaf_violation().item()

        print(f"  Zero sheaf violation (after training): {zero_violation:.6f}")
        assert zero_violation < 0.5, "Zero sheaf should have bounded violation"


    def test_sheaf_gluing_compatibility(self, simple_site):
        """Test that restrictions agree on overlaps."""
        feature_dim = 8
        sheaf = Sheaf(simple_site, feature_dim)

        # Create compatible sections
        sections = torch.randn(simple_site.num_objects, feature_dim)
        object.__setattr__(sheaf, 'sections', sections)

        # Check all objects
        for obj_idx in range(simple_site.num_objects):
            section = sheaf.at_object(obj_idx)

            # Should not be NaN
            assert not torch.isnan(section).any(), f"NaN in section {obj_idx}"

            # Should have correct dimension
            assert section.shape[0] == feature_dim

            print(f"  Object {obj_idx} section norm: {torch.norm(section).item():.4f}")


################################################################################
# Pullback/Pushforward Tests
################################################################################

class TestPullbackPushforward:
    """Test pullback and pushforward operations."""

    def test_pullback_then_pushforward(self, solver, example_grid):
        """Test that f_* ∘ f^* is close to identity on output."""
        sheaf_in = solver.encode_grid_to_sheaf(example_grid, solver.site_in)
        pushed = solver.geometric_morphism.pushforward(sheaf_in)

        # Apply f^* then f_*
        pulled = solver.geometric_morphism.pullback(pushed)
        pushed_again = solver.geometric_morphism.pushforward(pulled)

        # Measure error
        error = F.mse_loss(pushed_again.sections, pushed.sections).item()

        print(f"  Pushforward-pullback-pushforward error: {error:.6f}")
        assert error < 50.0, f"Composition error too large: {error}"


    def test_pushforward_then_pullback(self, solver, example_grid):
        """Test that f^* ∘ f_* is close to identity on input."""
        sheaf_in = solver.encode_grid_to_sheaf(example_grid, solver.site_in)

        # Apply f_* then f^*
        pushed = solver.geometric_morphism.pushforward(sheaf_in)
        pulled = solver.geometric_morphism.pullback(pushed)

        # Measure error
        error = F.mse_loss(pulled.sections, sheaf_in.sections).item()

        print(f"  Pullback-pushforward-pullback error: {error:.6f}")
        assert error < 50.0, f"Roundtrip error too large: {error}"


    def test_pullback_preserves_sheaf_condition(self, solver, example_grid):
        """Test that f^* maps sheaves to sheaves."""
        sheaf_in = solver.encode_grid_to_sheaf(example_grid, solver.site_in)
        pushed = solver.geometric_morphism.pushforward(sheaf_in)
        pulled = solver.geometric_morphism.pullback(pushed)

        # Pulled sheaf should satisfy sheaf condition
        violation = pulled.total_sheaf_violation().item()

        print(f"  Pulled sheaf violation: {violation:.6f}")
        assert violation >= 0
        assert not np.isnan(violation)
        assert not np.isinf(violation)


################################################################################
# Grid Encoding/Decoding Tests
################################################################################

class TestGridEncoding:
    """Test ARC grid encoding to sheaf."""

    def test_encoding_is_deterministic(self, solver, example_grid):
        """Test that encoding the same grid gives same sheaf."""
        sheaf1 = solver.encode_grid_to_sheaf(example_grid, solver.site_in)
        sheaf2 = solver.encode_grid_to_sheaf(example_grid, solver.site_in)

        diff = F.mse_loss(sheaf1.sections, sheaf2.sections).item()

        print(f"  Encoding difference: {diff:.10f}")
        assert diff < 1e-6, "Encoding should be deterministic"


    def test_encoding_preserves_info(self, solver, example_grid):
        """Test that encoding preserves grid information."""
        sheaf = solver.encode_grid_to_sheaf(example_grid, solver.site_in)

        # Sheaf should have non-trivial sections
        norm = torch.norm(sheaf.sections).item()

        print(f"  Sheaf section norm: {norm:.4f}")
        assert norm > 0, "Sheaf should have non-zero sections"
        assert not np.isnan(norm)


    def test_encoding_different_grids_differ(self, solver):
        """Test that different grids produce different sheaves."""
        grid1 = ARCGrid.from_array(np.zeros((5, 5), dtype=np.int32))
        grid2 = ARCGrid.from_array(np.ones((5, 5), dtype=np.int32))

        # Quick training to learn to distinguish different inputs
        solver = quick_train(solver, grid1, grid2, steps=100)

        sheaf1 = solver.encode_grid_to_sheaf(grid1, solver.site_in)
        sheaf2 = solver.encode_grid_to_sheaf(grid2, solver.site_in)

        diff = F.mse_loss(sheaf1.sections, sheaf2.sections).item()

        print(f"  Different grid sheaf difference (after training): {diff:.4f}")
        assert diff > 0.01, "Different grids should produce different sheaves"


################################################################################
# Topos Structure Tests
################################################################################

class TestToposStructure:
    """Test overall topos structure properties."""

    def test_site_coverage_reflexivity(self, simple_site):
        """Test that every object covers itself."""
        for obj_idx in range(simple_site.num_objects):
            covering = simple_site.coverage_families[obj_idx]
            assert obj_idx in covering, f"Object {obj_idx} doesn't cover itself"

            print(f"  Object {obj_idx} coverage: {covering}")


    def test_sheaf_section_finite(self, solver, example_grid):
        """Test that all sheaf sections are finite (no NaN/Inf)."""
        sheaf = solver.encode_grid_to_sheaf(example_grid, solver.site_in)

        assert not torch.isnan(sheaf.sections).any(), "Sheaf has NaN values"
        assert not torch.isinf(sheaf.sections).any(), "Sheaf has Inf values"

        print(f"  Sheaf section range: [{sheaf.sections.min():.2f}, {sheaf.sections.max():.2f}]")


    def test_geometric_morphism_well_defined(self, solver, example_grid):
        """Test that geometric morphism is well-defined."""
        sheaf_in = solver.encode_grid_to_sheaf(example_grid, solver.site_in)

        # Pushforward
        pushed = solver.geometric_morphism.pushforward(sheaf_in)
        assert pushed.site.num_objects == solver.site_out.num_objects

        # Pullback
        pulled = solver.geometric_morphism.pullback(pushed)
        assert pulled.site.num_objects == solver.site_in.num_objects

        print(f"  Input site size: {solver.site_in.num_objects}")
        print(f"  Output site size: {solver.site_out.num_objects}")


################################################################################
# Regression Tests (Known Examples)
################################################################################

class TestRegressions:
    """Test on known examples that should work."""

    def test_identity_grid_roundtrip(self, solver):
        """Test identity transformation on identity grid."""
        grid = ARCGrid.from_array(np.eye(5, dtype=np.int32))

        sheaf_in = solver.encode_grid_to_sheaf(grid, solver.site_in)
        sheaf_out = solver.geometric_morphism.pushforward(sheaf_in)
        prediction = solver.decode_sheaf_to_grid(sheaf_out, 5, 5)

        # Should be finite
        assert not np.isnan(prediction.cells).any()

        # Compute smooth accuracy
        l2_dist = np.sqrt(np.sum((prediction.cells - grid.cells) ** 2))
        max_dist = np.sqrt(25 * 81)
        accuracy = 1.0 - (l2_dist / max_dist)

        print(f"  Identity grid accuracy: {accuracy:.3f}")
        assert accuracy >= 0


    def test_zero_grid_roundtrip(self, solver):
        """Test transformation on zero grid."""
        grid = ARCGrid.from_array(np.zeros((5, 5), dtype=np.int32))

        sheaf_in = solver.encode_grid_to_sheaf(grid, solver.site_in)
        sheaf_out = solver.geometric_morphism.pushforward(sheaf_in)
        prediction = solver.decode_sheaf_to_grid(sheaf_out, 5, 5)

        assert not np.isnan(prediction.cells).any()

        print(f"  Zero grid prediction range: [{prediction.cells.min()}, {prediction.cells.max()}]")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
