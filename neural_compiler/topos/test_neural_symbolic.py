"""
Comprehensive Tests for Neural-Symbolic ARC Solver

Tests all components of the neural-symbolic pipeline:
1. Formula DSL
2. Kripke-Joyal interpreter
3. Neural predicates
4. Template library
5. End-to-end solver

Author: Claude Code
Date: October 23, 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List

from internal_language import (
    atom, forall, exists, implies, and_, or_, not_, assign,
    Var, Const
)
from kripke_joyal import KripkeJoyalInterpreter, create_arc_context
from neural_predicates import PredicateRegistry
from formula_templates import TemplateLibrary, fill_if, recolor_matching
from topos_categorical import Site, SubobjectClassifier
from neural_symbolic_arc import NeuralSymbolicARCSolver


################################################################################
# § 1: Test Formula DSL
################################################################################

def test_formula_dsl():
    """Test basic formula construction and evaluation."""
    print("=== Test 1: Formula DSL ===")

    # Create simple site and omega
    site = Site(grid_shape=(1, 1), connectivity="4")
    omega = SubobjectClassifier(site, truth_dim=1)

    # Define test predicate (always returns 0.8)
    def test_pred(x, context):
        return torch.tensor(0.8)

    predicates = {'test': test_pred}
    interpreter = KripkeJoyalInterpreter(omega, predicates, device='cpu')

    # Test atomic formula
    formula = atom('test', Var('x'))
    context = create_arc_context(torch.zeros(5, 5))
    result = interpreter.force(formula, context)

    assert abs(result.item() - 0.8) < 1e-6, f"Expected 0.8, got {result.item()}"
    print("  ✓ Atomic formula works")

    # Test conjunction
    formula_and = and_(atom('test', Var('x')), atom('test', Var('y')))
    result_and = interpreter.force(formula_and, context)
    expected = 0.8 * 0.8  # Product t-norm
    assert abs(result_and.item() - expected) < 1e-6
    print("  ✓ Conjunction (AND) works")

    # Test disjunction
    formula_or = or_(atom('test', Var('x')), atom('test', Var('y')))
    result_or = interpreter.force(formula_or, context)
    expected = 0.8 + 0.8 - 0.8 * 0.8  # Probabilistic sum
    assert abs(result_or.item() - expected) < 1e-6
    print("  ✓ Disjunction (OR) works")

    # Test negation
    formula_not = not_(atom('test', Var('x')))
    result_not = interpreter.force(formula_not, context)
    expected = 1.0 - 0.8
    assert abs(result_not.item() - expected) < 1e-6
    print("  ✓ Negation (NOT) works")

    print()


################################################################################
# § 2: Test Kripke-Joyal Semantics
################################################################################

def test_kripke_joyal():
    """Test Kripke-Joyal forcing semantics."""
    print("=== Test 2: Kripke-Joyal Semantics ===")

    site = Site(grid_shape=(3, 3), connectivity="4")
    omega = SubobjectClassifier(site, truth_dim=1)

    # Create grid with specific pattern
    grid = torch.tensor([
        [0, 0, 0],
        [0, 2, 0],  # Red cell in middle
        [0, 0, 0]
    ], dtype=torch.float32)

    # Define color predicate
    def is_red(cell_idx, context):
        g = context['grid']
        H, W = g.shape
        i, j = cell_idx // W, cell_idx % W
        color = g[i, j]
        # Return 1.0 if color==2, 0.0 otherwise
        return torch.tensor(1.0 if abs(color - 2.0) < 0.1 else 0.0)

    predicates = {'is_red': is_red}
    interpreter = KripkeJoyalInterpreter(omega, predicates, device='cpu')

    # Test: ∀cell. is_red(cell) (should be false)
    formula_all_red = forall(
        'cell',
        atom('is_red', Var('cell')),
        domain_fn=lambda ctx: range(9)
    )
    context = create_arc_context(grid)
    result = interpreter.force(formula_all_red, context)
    print(f"  ∀cell. is_red(cell) = {result.item():.4f} (expected ≈0.0)")
    assert result.item() < 0.1, "Not all cells are red"

    # Test: ∃cell. is_red(cell) (should be true)
    formula_exists_red = exists(
        'cell',
        atom('is_red', Var('cell')),
        domain_fn=lambda ctx: range(9)
    )
    result = interpreter.force(formula_exists_red, context)
    print(f"  ∃cell. is_red(cell) = {result.item():.4f} (expected ≈1.0)")
    assert result.item() > 0.9, "At least one cell is red"

    print("  ✓ Quantifiers work correctly")
    print()


################################################################################
# § 3: Test Neural Predicates
################################################################################

def test_neural_predicates():
    """Test learned neural predicates."""
    print("=== Test 3: Neural Predicates ===")

    registry = PredicateRegistry(num_colors=10, feature_dim=32, device='cpu')

    # Test grid
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
    val_corner = boundary_pred(0, context)  # (0,0) is boundary
    val_center = boundary_pred(12, context)  # (2,2) is interior

    assert val_corner.item() > 0.9, "Corner should be boundary"
    assert val_center.item() < 0.1, "Center should not be boundary"
    print("  ✓ Boundary predicate works")

    # Test inside predicate
    inside_pred = registry.get('is_inside')
    val_inside = inside_pred(12, context)  # (2,2) is inside
    val_outside = inside_pred(0, context)  # (0,0) is outside

    assert val_inside.item() > 0.9, "Center should be inside"
    assert val_outside.item() < 0.1, "Corner should not be inside"
    print("  ✓ Inside predicate works")

    # Test color predicate (learned)
    color_pred = registry.get('color_eq')
    val_red = color_pred(6, 2, context)  # Cell (1,1) is red (color 2)
    print(f"  Color match (before training): {val_red.item():.4f}")
    print("  ✓ Color predicate initialized")

    # Test shape predicate
    shape_pred = registry.get('is_square')
    red_mask = (grid == 2).float()
    val_square = shape_pred(red_mask, context)
    print(f"  Is square (before training): {val_square.item():.4f}")
    print("  ✓ Shape predicate initialized")

    print()


################################################################################
# § 4: Test Template Library
################################################################################

def test_template_library():
    """Test formula template library."""
    print("=== Test 4: Template Library ===")

    library = TemplateLibrary()

    # Check predefined templates exist
    templates = library.all_template_names()
    print(f"  Predefined templates: {len(templates)}")
    assert len(templates) > 0, "Library should have templates"

    # Test getting template
    template = library.get('fill_inside_red')
    print(f"  fill_inside_red: {template}")
    assert template is not None

    # Test template families
    fill_family = library.get_family('fill_if')
    custom = fill_family.instantiate('is_corner', 5)
    print(f"  Custom template: {custom}")

    print("  ✓ Template library works")
    print()


################################################################################
# § 5: Test Smooth Operators
################################################################################

def test_smooth_operators():
    """Test smooth logic operators for gradient flow."""
    print("=== Test 5: Smooth Operators ===")

    site = Site(grid_shape=(1, 1), connectivity="4")
    omega = SubobjectClassifier(site, truth_dim=1)

    # Test conjunction gradient
    p = torch.tensor(0.8, requires_grad=True)
    q = torch.tensor(0.6, requires_grad=True)

    result = omega.conjunction(p, q)
    result.backward()

    print(f"  Conjunction: {p.item():.2f} ∧ {q.item():.2f} = {result.item():.2f}")
    print(f"  ∂/∂p = {p.grad.item():.2f}, ∂/∂q = {q.grad.item():.2f}")
    assert p.grad is not None and q.grad is not None
    print("  ✓ Conjunction gradients flow")

    # Test disjunction gradient
    p = torch.tensor(0.8, requires_grad=True)
    q = torch.tensor(0.6, requires_grad=True)

    result = omega.disjunction(p, q)
    result.backward()

    print(f"  Disjunction: {p.item():.2f} ∨ {q.item():.2f} = {result.item():.2f}")
    print(f"  ∂/∂p = {p.grad.item():.2f}, ∂/∂q = {q.grad.item():.2f}")
    assert p.grad is not None and q.grad is not None
    print("  ✓ Disjunction gradients flow")

    # Test forall gradient
    values = [torch.tensor(0.9, requires_grad=True) for _ in range(3)]
    result = omega.forall(values)
    result.backward()

    print(f"  Forall: ∀[0.9, 0.9, 0.9] = {result.item():.4f}")
    print(f"  Gradients: {[v.grad.item() for v in values]}")
    assert all(v.grad is not None for v in values)
    print("  ✓ Forall gradients flow")

    print()


################################################################################
# § 6: Test End-to-End Solver
################################################################################

def test_end_to_end_solver():
    """Test complete neural-symbolic solver."""
    print("=== Test 6: End-to-End Solver ===")

    device = torch.device('cpu')  # Use CPU for testing

    # Create model
    model = NeuralSymbolicARCSolver(
        num_colors=10,
        feature_dim=32,
        device=device,
        max_composite_depth=1  # Smaller search space for testing
    )

    print(f"  Model initialized with {len(model.templates)} templates")

    # Test input
    input_grid = torch.tensor([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=torch.float32, device=device)

    # Forward pass
    output, info = model.forward(input_grid.unsqueeze(0), hard_select=True)

    print(f"  Input shape: {input_grid.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Selected template index: {info['selected_indices'][0].item()}")

    # Test prediction
    pred = model.predict(input_grid)
    print(f"  Prediction shape: {pred.shape}")
    assert pred.shape == input_grid.shape

    print("  ✓ End-to-end solver works")
    print()


################################################################################
# § 7: Test Training Step
################################################################################

def test_training():
    """Test training with gradient flow."""
    print("=== Test 7: Training Step ===")

    device = torch.device('cpu')

    # Create model
    model = NeuralSymbolicARCSolver(
        num_colors=10,
        feature_dim=32,
        device=device,
        max_composite_depth=1
    )

    # Simple task: copy input
    input_grid = torch.tensor([
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=device)

    target_grid = input_grid.clone()

    # Compute loss
    loss, losses = model.compute_loss(
        input_grid.unsqueeze(0),
        target_grid.unsqueeze(0)
    )

    print(f"  Initial loss: {loss.item():.4f}")
    print(f"  Pixel loss: {losses['pixel'].item():.4f}")
    print(f"  Template entropy: {losses['template_entropy'].item():.4f}")

    # Backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "Should have gradients"
    print("  ✓ Gradients computed")

    # Optimization step
    optimizer.step()
    print("  ✓ Optimization step completed")

    print()


################################################################################
# § 8: Run All Tests
################################################################################

def run_all_tests():
    """Run complete test suite."""
    print("=" * 70)
    print("NEURAL-SYMBOLIC ARC SOLVER - TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        ("Formula DSL", test_formula_dsl),
        ("Kripke-Joyal Semantics", test_kripke_joyal),
        ("Neural Predicates", test_neural_predicates),
        ("Template Library", test_template_library),
        ("Smooth Operators", test_smooth_operators),
        ("End-to-End Solver", test_end_to_end_solver),
        ("Training", test_training),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {name} FAILED:")
            print(f"   {str(e)}")
            print()
            failed += 1

    print("=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} tests failed")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
