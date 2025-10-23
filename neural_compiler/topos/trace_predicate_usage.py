"""
Trace Exactly When Predicates Are Called During Forward Pass

This will instrument the predicates to print when they're called,
so we can see if color_eq is ACTUALLY being executed during training.

Author: Claude Code
Date: October 23, 2025
"""

import torch
from trm_neural_symbolic import TRMNeuralSymbolicSolver


# Global counter
call_counts = {}


def instrument_predicate(pred, name):
    """Wrap predicate forward to track calls."""
    original_forward = pred.forward

    def tracked_forward(*args, **kwargs):
        global call_counts
        call_counts[name] = call_counts.get(name, 0) + 1
        return original_forward(*args, **kwargs)

    pred.forward = tracked_forward


def test_predicate_calls():
    """Test which predicates are actually called."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Create model
    model = TRMNeuralSymbolicSolver(
        num_colors=10,
        num_cycles=3,
        device=device
    ).to(device)

    # Instrument all predicates
    print("=== Instrumenting Predicates ===")
    for pred_name in model.predicates_vectorized.list_predicates():
        pred = model.predicates_vectorized.get(pred_name)
        instrument_predicate(pred, pred_name)
        print(f"  Instrumented: {pred_name}")

    # Create diverse data
    batch_size = 4
    H, W = 5, 5

    input_grid = torch.zeros((batch_size, H, W), device=device, dtype=torch.long)
    target_grid = torch.zeros((batch_size, H, W), device=device, dtype=torch.long)

    for b in range(batch_size):
        for i in range(H):
            for j in range(W):
                input_grid[b, i, j] = (i + j) % 5
                if input_grid[b, i, j] == 2:
                    target_grid[b, i, j] = 7
                else:
                    target_grid[b, i, j] = input_grid[b, i, j]

    print("\n=== Running Forward Pass ===")
    print(f"Input grid colors: {torch.unique(input_grid).tolist()}")
    print(f"Target grid colors: {torch.unique(target_grid).tolist()}")

    # Forward
    output, info = model.forward(input_grid, target_size=(H, W), hard_select=True)

    print("\n=== Selected Templates ===")
    for i, template in enumerate(info['selected_templates']):
        print(f"  Batch {i}: {str(template)[:100]}...")

    print("\n=== Predicate Call Counts ===")
    if call_counts:
        for pred_name, count in sorted(call_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pred_name:25s}: {count:4d} calls")

        if 'color_eq' in call_counts:
            print(f"\n✅ color_eq WAS called {call_counts['color_eq']} times")
        else:
            print(f"\n❌ color_eq was NEVER called!")
            print(f"   This explains why it has zero gradients")

    else:
        print("  NO PREDICATES CALLED!")
        print("  ❌ This means formulas aren't using vectorized interpreter")

    # Test backward
    print("\n=== Testing Backward Pass ===")
    loss = F.mse_loss(output.float(), target_grid.float())
    loss.backward()

    color_eq_pred = model.predicates_vectorized.get('color_eq')
    if color_eq_pred.color_embed.weight.grad is not None:
        grad_mag = color_eq_pred.color_embed.weight.grad.abs().sum().item()
        print(f"color_eq gradient magnitude: {grad_mag:.6e}")

        if grad_mag > 1e-10 and 'color_eq' in call_counts:
            print(f"✅ SUCCESS: color_eq was called AND has non-zero gradients!")
        elif grad_mag > 1e-10:
            print(f"⚠️  color_eq has gradients but wasn't called (unexpected)")
        elif 'color_eq' in call_counts:
            print(f"❌ color_eq was called but gradients are zero")
            print(f"   Gradient graph may be disconnected")
        else:
            print(f"❌ color_eq not called AND no gradients")
    else:
        print(f"❌ color_eq has no gradient tensor")


if __name__ == "__main__":
    import torch.nn.functional as F

    print("="*80)
    print("PREDICATE USAGE TRACING")
    print("="*80)

    test_predicate_calls()

    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    print("\nIf color_eq is never called:")
    print("  → Selected templates don't use color_eq")
    print("  → Need to bias template selection or use simpler templates")
    print("\nIf color_eq is called but gradients are zero:")
    print("  → Computational graph is disconnected")
    print("  → Check _apply_formula_vectorized for detach/clone")
