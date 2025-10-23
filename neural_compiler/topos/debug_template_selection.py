"""
Debug Template Selection

Investigate which templates are being selected and why color_eq isn't used.

Author: Claude Code
Date: October 23, 2025
"""

import torch
from trm_neural_symbolic import TRMNeuralSymbolicSolver
from internal_language import Atomic


def analyze_templates():
    """Analyze which templates use which predicates."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Create model
    model = TRMNeuralSymbolicSolver(
        num_colors=10,
        num_cycles=3,
        device=device
    ).to(device)

    print(f"Total templates: {len(model.templates)}")
    print("\n=== Analyzing Template Predicates ===\n")

    # Count predicate usage
    predicate_usage = {}

    def extract_predicates(formula, predicates_set):
        """Recursively extract all predicates from formula."""
        if isinstance(formula, Atomic):
            predicates_set.add(formula.predicate_name)
        elif hasattr(formula, 'left') and hasattr(formula, 'right'):
            extract_predicates(formula.left, predicates_set)
            extract_predicates(formula.right, predicates_set)
        elif hasattr(formula, 'inner'):
            extract_predicates(formula.inner, predicates_set)
        elif hasattr(formula, 'antecedent') and hasattr(formula, 'consequent'):
            extract_predicates(formula.antecedent, predicates_set)
            extract_predicates(formula.consequent, predicates_set)
        elif hasattr(formula, 'body'):
            extract_predicates(formula.body, predicates_set)
        elif hasattr(formula, 'steps'):
            for step in formula.steps:
                extract_predicates(step, predicates_set)

    for i, template in enumerate(model.templates):
        predicates = set()
        extract_predicates(template, predicates)

        for pred in predicates:
            if pred not in predicate_usage:
                predicate_usage[pred] = []
            predicate_usage[pred].append(i)

    # Print results
    print(f"Predicates used across all templates:")
    for pred, template_ids in sorted(predicate_usage.items()):
        print(f"  {pred:25s}: {len(template_ids):3d} templates")

    if 'color_eq' in predicate_usage:
        print(f"\n✅ color_eq IS used in {len(predicate_usage['color_eq'])} templates")
        print(f"   Template indices: {predicate_usage['color_eq'][:10]}... (showing first 10)")
    else:
        print(f"\n❌ color_eq is NOT used in ANY template!")

    return model, predicate_usage


def test_specific_template():
    """Test a specific template that uses color_eq."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model, predicate_usage = analyze_templates()

    if 'color_eq' not in predicate_usage:
        print("\n❌ Cannot test color_eq - not in any template!")
        return

    # Get a template that uses color_eq
    color_eq_template_idx = predicate_usage['color_eq'][0]
    template = model.templates[color_eq_template_idx]

    print(f"\n=== Testing Template {color_eq_template_idx} ===")
    print(f"Template: {template}")

    # Create simple test case
    input_grid = torch.full((1, 3, 3), 2, device=device, dtype=torch.long)
    target_grid = torch.full((1, 3, 3), 7, device=device, dtype=torch.long)

    # Manually apply this template
    print("\n=== Applying Template Directly ===")

    from kripke_joyal_vectorized import create_grid_context

    context = create_grid_context(input_grid[0])
    truth_map = model.interpreter_vectorized.force_batch(template, context)

    print(f"Truth map shape: {truth_map.shape}")
    print(f"Truth map:\n{truth_map}")

    # Check if predicate was called
    color_eq_pred = model.predicates_vectorized.get('color_eq')
    print(f"\n=== Predicate State ===")
    print(f"color_eq.color_embed.weight requires_grad: {color_eq_pred.color_embed.weight.requires_grad}")
    print(f"color_eq.compare_net[0].weight requires_grad: {color_eq_pred.compare_net[0].weight.requires_grad}")

    # Now do backward pass
    print("\n=== Backward Pass Through Template ===")
    target_truth = torch.ones_like(truth_map)
    loss = F.mse_loss(truth_map, target_truth)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"color_eq.color_embed.weight.grad exists: {color_eq_pred.color_embed.weight.grad is not None}")

    if color_eq_pred.color_embed.weight.grad is not None:
        grad_mag = color_eq_pred.color_embed.weight.grad.abs().sum().item()
        print(f"✅ Gradient magnitude: {grad_mag:.6e}")
        if grad_mag > 1e-10:
            print(f"✅ Gradients ARE flowing when template is used directly!")
        else:
            print(f"❌ Gradients exist but are zero")
    else:
        print(f"❌ NO gradients")


def test_selection_mechanism():
    """Test which templates are actually being selected during forward pass."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model, predicate_usage = analyze_templates()

    # Create test data
    input_grid = torch.full((4, 5, 5), 2, device=device, dtype=torch.long)
    target_grid = torch.full((4, 5, 5), 7, device=device, dtype=torch.long)

    print("\n=== Testing Template Selection ===")
    print(f"Input: all cells = 2")
    print(f"Target: all cells = 7")

    # Run 10 iterations
    selected_templates_counts = {}

    for i in range(10):
        output_grid, info = model.forward(input_grid, target_size=(5, 5), hard_select=True)

        selected_templates = info['selected_templates']

        for template in selected_templates:
            template_str = str(template)
            selected_templates_counts[template_str] = selected_templates_counts.get(template_str, 0) + 1

    print(f"\n=== Selected Templates (10 iterations × 4 batch) ===")
    for template_str, count in sorted(selected_templates_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {count:3d}× {template_str[:80]}...")

    # Check if any selected template uses color_eq
    uses_color_eq = False
    for i, template in enumerate(model.templates):
        predicates = set()
        extract_predicates(template, predicates)
        template_str = str(template)

        if 'color_eq' in predicates and template_str in selected_templates_counts:
            uses_color_eq = True
            print(f"\n✅ Template {i} uses color_eq and was selected {selected_templates_counts[template_str]} times")
            break

    if not uses_color_eq:
        print(f"\n❌ ISSUE FOUND: No selected templates use color_eq predicate!")
        print(f"   This is why gradients don't flow to color_eq")
        print(f"   The selector is choosing templates that don't need learned predicates")


def extract_predicates(formula, predicates_set):
    """Recursively extract all predicates from formula."""
    if isinstance(formula, Atomic):
        predicates_set.add(formula.predicate_name)
    elif hasattr(formula, 'left') and hasattr(formula, 'right'):
        extract_predicates(formula.left, predicates_set)
        extract_predicates(formula.right, predicates_set)
    elif hasattr(formula, 'inner'):
        extract_predicates(formula.inner, predicates_set)
    elif hasattr(formula, 'antecedent') and hasattr(formula, 'consequent'):
        extract_predicates(formula.antecedent, predicates_set)
        extract_predicates(formula.consequent, predicates_set)
    elif hasattr(formula, 'body'):
        extract_predicates(formula.body, predicates_set)
    elif hasattr(formula, 'steps'):
        for step in formula.steps:
            extract_predicates(step, predicates_set)


if __name__ == "__main__":
    print("="*80)
    print("TEMPLATE SELECTION DEBUGGING")
    print("="*80)

    import torch.nn.functional as F

    # Analyze which predicates are in templates
    model, usage = analyze_templates()

    # Test specific template with color_eq
    test_specific_template()

    # Test what gets selected
    test_selection_mechanism()

    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    print("\nPossible issues:")
    print("1. Selector may prefer non-learned predicates (geometric, deterministic)")
    print("2. Templates with color_eq may have lower selection probability")
    print("3. Need to encourage selection of learned predicates during training")
    print("\nSolutions:")
    print("1. Add entropy bonus to selector loss (encourage exploration)")
    print("2. Use curriculum: start with templates requiring learned predicates")
    print("3. Add auxiliary loss on predicate usage diversity")
