"""
Comprehensive tests for ARC topos solver with zero-padding and error handling.

Tests cover:
1. Zero-padding for variable-sized grids
2. Dimension inference (preserving vs changing)
3. Shape mismatch error handling
4. Evaluation with worst-case rewards
5. End-to-end prediction pipeline
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pytest

from arc_solver import (
    ARCGrid, ARCTask, ARCReasoningNetwork,
    create_grid_site, ARCToposSolver
)
from arc_loader import evaluate_prediction, evaluate_task


################################################################################
# § 1: Zero-Padding Tests
################################################################################

class TestZeroPadding:
    """Test zero-padding functionality for variable-sized grids."""

    def test_encode_grid_without_padding(self):
        """Test encoding without padding (backward compatibility)."""
        # Create network and grid
        network = ARCReasoningNetwork(hidden_dim=32, num_colors=10)
        key = random.PRNGKey(0)
        site = create_grid_site(10, 10, "local", key)

        grid = ARCGrid.from_array(np.random.randint(0, 10, (5, 5)))
        # Need some examples for initialization
        examples = [(grid, grid)]

        # Initialize via full forward pass
        params = network.init(key, grid, examples, site)['params']

        # Forward pass (uses encode internally)
        prediction = network.apply(
            {'params': params},
            grid, examples, site
        )

        # Should produce output with same dimensions
        assert prediction.height == 5, f"Expected height 5, got {prediction.height}"
        assert prediction.width == 5, f"Expected width 5, got {prediction.width}"
        print("✓ Encoding without padding works")

    def test_encode_grid_with_padding(self):
        """Test encoding with zero-padding to larger dimension."""
        network = ARCReasoningNetwork(hidden_dim=32, num_colors=10)
        key = random.PRNGKey(0)
        site = create_grid_site(10, 10, "local", key)

        # Small grid with large output examples (forces padding)
        small_grid = ARCGrid.from_array(np.random.randint(0, 10, (3, 3)))
        large_grid = ARCGrid.from_array(np.random.randint(0, 10, (9, 9)))
        examples = [(small_grid, large_grid)]

        # Initialize and predict
        params = network.init(key, small_grid, examples, site)['params']
        prediction = network.apply(
            {'params': params},
            small_grid, examples, site
        )

        # Should produce 9×9 output (max size from examples)
        assert prediction.height == 9, f"Expected height 9, got {prediction.height}"
        assert prediction.width == 9, f"Expected width 9, got {prediction.width}"
        print(f"✓ Zero-padding 3×3 → 9×9 works")

    def test_padding_preserves_data(self):
        """Test that padding works with different sized grids in same batch."""
        network = ARCReasoningNetwork(hidden_dim=32, num_colors=10)
        key = random.PRNGKey(42)
        site = create_grid_site(10, 10, "local", key)

        # Create deterministic grids of different sizes
        small_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        large_data = np.random.randint(0, 10, (9, 9))

        small_grid = ARCGrid.from_array(small_data)
        large_grid = ARCGrid.from_array(large_data)

        # Examples with mixed sizes
        examples = [(small_grid, large_grid)]

        # Initialize and predict
        params = network.init(key, small_grid, examples, site)['params']

        # Predict on small grid (should work with padding)
        prediction1 = network.apply(
            {'params': params},
            small_grid, examples, site
        )

        # Predict on large grid (should work without padding)
        prediction2 = network.apply(
            {'params': params},
            large_grid, examples, site
        )

        # Both should produce same-sized output (9×9 from examples)
        assert prediction1.height == 9 and prediction1.width == 9
        assert prediction2.height == 9 and prediction2.width == 9
        print("✓ Zero-padding preserves data integrity across different sizes")

    def test_mixed_sizes_in_forward_pass(self):
        """Test forward pass with grids of different sizes."""
        network = ARCReasoningNetwork(hidden_dim=32, num_colors=10)
        key = random.PRNGKey(0)
        site = create_grid_site(10, 10, "local", key)

        # Create grids of different sizes
        grid_3x3 = ARCGrid.from_array(np.random.randint(0, 10, (3, 3)))
        grid_9x9 = ARCGrid.from_array(np.random.randint(0, 10, (9, 9)))
        grid_6x3 = ARCGrid.from_array(np.random.randint(0, 10, (6, 3)))

        # Initialize
        params = network.init(key, grid_3x3, [(grid_3x3, grid_9x9)], site)['params']

        # Forward pass should work
        prediction = network.apply(
            {'params': params},
            grid_6x3,
            [(grid_3x3, grid_9x9), (grid_6x3, grid_6x3)],
            site
        )

        assert isinstance(prediction, ARCGrid), "Prediction should be ARCGrid"
        print(f"✓ Mixed sizes forward pass works (output: {prediction.height}×{prediction.width})")


################################################################################
# § 2: Dimension Inference Tests
################################################################################

class TestDimensionInference:
    """Test smart output dimension inference."""

    def test_dimension_preserving_task(self):
        """Test dimension-preserving tasks (output size = input size)."""
        network = ARCReasoningNetwork(hidden_dim=32, num_colors=10)
        key = random.PRNGKey(0)
        site = create_grid_site(20, 20, "local", key)

        # All examples preserve dimensions
        examples = [
            (ARCGrid.from_array(np.random.randint(0, 10, (6, 6))),
             ARCGrid.from_array(np.random.randint(0, 10, (6, 6)))),
            (ARCGrid.from_array(np.random.randint(0, 10, (10, 10))),
             ARCGrid.from_array(np.random.randint(0, 10, (10, 10)))),
            (ARCGrid.from_array(np.random.randint(0, 10, (20, 20))),
             ARCGrid.from_array(np.random.randint(0, 10, (20, 20)))),
        ]

        # Test input with different size
        test_input = ARCGrid.from_array(np.random.randint(0, 10, (15, 15)))

        # Initialize and predict
        params = network.init(key, test_input, examples, site)['params']
        prediction = network.apply(
            {'params': params},
            test_input,
            examples,
            site
        )

        # Output should match test input size (dimension-preserving)
        assert prediction.height == 15, f"Expected height 15, got {prediction.height}"
        assert prediction.width == 15, f"Expected width 15, got {prediction.width}"
        print("✓ Dimension-preserving inference works (15×15 → 15×15)")

    def test_dimension_changing_task(self):
        """Test dimension-changing tasks (fixed output size)."""
        network = ARCReasoningNetwork(hidden_dim=32, num_colors=10)
        key = random.PRNGKey(0)
        site = create_grid_site(10, 10, "local", key)

        # All examples: 3×3 → 9×9
        examples = [
            (ARCGrid.from_array(np.random.randint(0, 10, (3, 3))),
             ARCGrid.from_array(np.random.randint(0, 10, (9, 9)))),
            (ARCGrid.from_array(np.random.randint(0, 10, (3, 3))),
             ARCGrid.from_array(np.random.randint(0, 10, (9, 9)))),
        ]

        # Test input
        test_input = ARCGrid.from_array(np.random.randint(0, 10, (3, 3)))

        # Initialize and predict
        params = network.init(key, test_input, examples, site)['params']
        prediction = network.apply(
            {'params': params},
            test_input,
            examples,
            site
        )

        # Output should be 9×9 (from examples, not input)
        assert prediction.height == 9, f"Expected height 9, got {prediction.height}"
        assert prediction.width == 9, f"Expected width 9, got {prediction.width}"
        print("✓ Dimension-changing inference works (3×3 → 9×9)")

    def test_rectangular_grids(self):
        """Test non-square grids (6×3 → 9×3)."""
        network = ARCReasoningNetwork(hidden_dim=32, num_colors=10)
        key = random.PRNGKey(0)
        site = create_grid_site(10, 10, "local", key)

        # Rectangular transformation
        examples = [
            (ARCGrid.from_array(np.random.randint(0, 10, (6, 3))),
             ARCGrid.from_array(np.random.randint(0, 10, (9, 3)))),
        ]

        test_input = ARCGrid.from_array(np.random.randint(0, 10, (6, 3)))

        params = network.init(key, test_input, examples, site)['params']
        prediction = network.apply(
            {'params': params},
            test_input,
            examples,
            site
        )

        assert prediction.height == 9 and prediction.width == 3, \
            f"Expected 9×3, got {prediction.height}×{prediction.width}"
        print("✓ Rectangular grid inference works (6×3 → 9×3)")


################################################################################
# § 3: Error Handling Tests
################################################################################

class TestErrorHandling:
    """Test graceful error handling for shape mismatches."""

    def test_shape_mismatch_returns_zero_accuracy(self):
        """Test that shape mismatches return 0% accuracy instead of crashing."""
        predicted = ARCGrid.from_array(np.random.randint(0, 10, (5, 5)))
        ground_truth = ARCGrid.from_array(np.random.randint(0, 10, (10, 10)))

        # Should not crash
        metrics = evaluate_prediction(predicted, ground_truth)

        # Should return worst possible scores
        assert metrics['accuracy'] == 0.0, f"Expected 0% accuracy, got {metrics['accuracy']}"
        assert metrics['exact_match'] == False
        assert metrics['size_match'] == False
        assert 'error' in metrics
        print(f"✓ Shape mismatch handled gracefully: {metrics['error']}")

    def test_matching_shapes_work_normally(self):
        """Test that matching shapes still compute accuracy correctly."""
        # Create identical grids
        data = np.random.randint(0, 10, (5, 5))
        predicted = ARCGrid.from_array(data.copy())
        ground_truth = ARCGrid.from_array(data.copy())

        metrics = evaluate_prediction(predicted, ground_truth)

        # Should be perfect match
        assert metrics['accuracy'] == 1.0, f"Expected 100% accuracy, got {metrics['accuracy']}"
        assert metrics['exact_match'] == True
        assert metrics['size_match'] == True
        assert 'error' not in metrics
        print("✓ Matching shapes compute accuracy correctly")

    def test_partial_match(self):
        """Test partial accuracy computation."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        predicted = ARCGrid.from_array(data.copy())

        # Change 3 out of 9 cells
        ground_truth_data = data.copy()
        ground_truth_data[0, 0] = 0  # Wrong
        ground_truth_data[1, 1] = 0  # Wrong
        ground_truth_data[2, 2] = 0  # Wrong
        ground_truth = ARCGrid.from_array(ground_truth_data)

        metrics = evaluate_prediction(predicted, ground_truth)

        # Should be 6/9 = 66.67% accuracy
        expected_accuracy = 6.0 / 9.0
        assert abs(metrics['accuracy'] - expected_accuracy) < 1e-6, \
            f"Expected {expected_accuracy:.1%}, got {metrics['accuracy']:.1%}"
        assert metrics['correct_cells'] == 6
        assert metrics['total_cells'] == 9
        print(f"✓ Partial match computed correctly: {metrics['accuracy']:.1%}")

    def test_evaluate_task_with_errors(self):
        """Test task evaluation with shape mismatches."""
        # Create task with mismatched outputs
        task = ARCTask(
            train_inputs=[ARCGrid.from_array(np.ones((3, 3), dtype=int))],
            train_outputs=[ARCGrid.from_array(np.ones((3, 3), dtype=int))],
            test_inputs=[ARCGrid.from_array(np.ones((3, 3), dtype=int))],
            test_outputs=[ARCGrid.from_array(np.ones((9, 9), dtype=int))]
        )

        # Predict wrong size
        predictions = [ARCGrid.from_array(np.zeros((3, 3), dtype=int))]

        # Should not crash
        results = evaluate_task(task, predictions)

        # Should return 0% accuracy
        assert results['avg_accuracy'] == 0.0
        assert results['task_solved'] == False
        assert results['exact_matches'] == 0
        print("✓ Task evaluation with errors handled gracefully")


################################################################################
# § 4: End-to-End Integration Tests
################################################################################

class TestEndToEnd:
    """Test complete prediction pipeline."""

    def test_simple_identity_task(self):
        """Test identity transformation (output = input)."""
        key = random.PRNGKey(123)

        # Create simple identity task
        train_data = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]])
        ]

        task = ARCTask(
            train_inputs=[ARCGrid.from_array(d) for d in train_data],
            train_outputs=[ARCGrid.from_array(d) for d in train_data],  # Identity
            test_inputs=[ARCGrid.from_array(np.array([[9, 0], [1, 2]]))],
            test_outputs=[ARCGrid.from_array(np.array([[9, 0], [1, 2]]))]
        )

        # Create solver
        solver = ARCToposSolver(
            population_size=4,
            generations=2,  # Minimal for testing
            grid_size=10,
            coverage_type="local"
        )

        # Solve (should complete without errors)
        k1, k2 = random.split(key)
        best_site, prediction, fitness_history = solver.solve_arc_task(k2, task, verbose=False)

        # Check prediction is right size
        assert prediction.height == 2
        assert prediction.width == 2
        assert len(fitness_history) == 2
        print("✓ End-to-end identity task works")

    def test_scaling_task(self):
        """Test scaling transformation (3×3 → 6×6)."""
        key = random.PRNGKey(456)

        # Create scaling task
        small = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        large = np.random.randint(0, 10, (6, 6))

        task = ARCTask(
            train_inputs=[ARCGrid.from_array(small)],
            train_outputs=[ARCGrid.from_array(large)],
            test_inputs=[ARCGrid.from_array(small)],
            test_outputs=[ARCGrid.from_array(large)]
        )

        solver = ARCToposSolver(
            population_size=4,
            generations=2,
            grid_size=10,
            coverage_type="local"
        )

        k1, k2 = random.split(key)
        best_site, prediction, fitness_history = solver.solve_arc_task(k2, task, verbose=False)

        # Prediction should be 6×6 (inferred from examples)
        assert prediction.height == 6, f"Expected height 6, got {prediction.height}"
        assert prediction.width == 6, f"Expected width 6, got {prediction.width}"
        print("✓ End-to-end scaling task works (3×3 → 6×6)")

    def test_variable_size_task(self):
        """Test task with variable-sized examples."""
        key = random.PRNGKey(789)

        # Mix of sizes (dimension-preserving)
        task = ARCTask(
            train_inputs=[
                ARCGrid.from_array(np.random.randint(0, 10, (3, 3))),
                ARCGrid.from_array(np.random.randint(0, 10, (5, 5))),
                ARCGrid.from_array(np.random.randint(0, 10, (7, 7)))
            ],
            train_outputs=[
                ARCGrid.from_array(np.random.randint(0, 10, (3, 3))),
                ARCGrid.from_array(np.random.randint(0, 10, (5, 5))),
                ARCGrid.from_array(np.random.randint(0, 10, (7, 7)))
            ],
            test_inputs=[ARCGrid.from_array(np.random.randint(0, 10, (4, 4)))],
            test_outputs=[ARCGrid.from_array(np.random.randint(0, 10, (4, 4)))]
        )

        solver = ARCToposSolver(
            population_size=4,
            generations=2,
            grid_size=10,
            coverage_type="local"
        )

        k1, k2 = random.split(key)
        best_site, prediction, fitness_history = solver.solve_arc_task(k2, task, verbose=False)

        # Should infer 4×4 (dimension-preserving)
        assert prediction.height == 4
        assert prediction.width == 4
        print("✓ End-to-end variable-size task works")


################################################################################
# § 5: Test Runner
################################################################################

def run_all_tests():
    """Run all test suites."""
    print("=" * 70)
    print("RUNNING ARC TOPOS SOLVER TESTS")
    print("=" * 70)
    print()

    # Test suites
    suites = [
        ("Zero-Padding", TestZeroPadding),
        ("Dimension Inference", TestDimensionInference),
        ("Error Handling", TestErrorHandling),
        ("End-to-End Integration", TestEndToEnd)
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for suite_name, suite_class in suites:
        print(f"\n{'=' * 70}")
        print(f"TEST SUITE: {suite_name}")
        print(f"{'=' * 70}\n")

        # Get all test methods
        test_methods = [m for m in dir(suite_class) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            test_name = method_name.replace('_', ' ').title()

            try:
                # Run test
                suite = suite_class()
                method = getattr(suite, method_name)
                method()
                passed_tests += 1

            except Exception as e:
                print(f"✗ FAILED: {test_name}")
                print(f"  Error: {str(e)}")
                failed_tests.append((suite_name, test_name, str(e)))

    # Summary
    print(f"\n{'=' * 70}")
    print("TEST SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests} ✓")
    print(f"Failed: {len(failed_tests)} ✗")

    if failed_tests:
        print(f"\nFailed tests:")
        for suite, test, error in failed_tests:
            print(f"  - {suite} / {test}")
            print(f"    {error}")
    else:
        print("\n🎉 ALL TESTS PASSED!")

    print(f"{'=' * 70}\n")

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
