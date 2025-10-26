"""
Compare TRM-Enhanced vs Baseline Neural-Symbolic Solvers

Side-by-side comparison of:
1. Baseline: NeuralSymbolicARCSolver (large CNNs, no recursion)
2. TRM: TRMNeuralSymbolicSolver (tiny 2-layer, recursive refinement)

Expected improvement:
- Baseline: 0-2% binary accuracy
- TRM: 10-20% → 30-45% with tuning

Author: Claude Code
Date: October 23, 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt

from neural_symbolic_arc import NeuralSymbolicARCSolver
from trm_neural_symbolic import TRMNeuralSymbolicSolver, EMA
from gros_topos_curriculum import load_mini_arc, load_arc_agi_1
from arc_solver import ARCGrid


def evaluate_model(
    model: nn.Module,
    tasks: List,
    device: torch.device,
    use_ema: bool = False,
    ema: EMA = None
) -> Dict[str, float]:
    """Evaluate model on tasks.

    Args:
        model: Model to evaluate
        tasks: List of ARC tasks
        device: Device to run on
        use_ema: If True, use EMA weights (for TRM)
        ema: EMA object (required if use_ema=True)

    Returns:
        metrics: Dictionary with pixel_acc, binary_acc, n_examples
    """
    # Apply EMA if requested
    if use_ema and ema is not None:
        ema.apply_shadow()

    model.eval()

    total_pixel_acc = 0.0
    total_binary_acc = 0.0
    n_examples = 0

    with torch.no_grad():
        for task in tasks:
            for inp, out in zip(task.input_examples, task.output_examples):
                inp_tensor = torch.from_numpy(np.array(inp)).float().to(device)
                out_tensor = torch.from_numpy(np.array(out)).float().to(device)

                # Get target size
                target_size = out_tensor.shape

                # Predict
                pred = model.predict(inp_tensor, target_size=target_size)

                # Metrics
                pixel_acc = (pred == out_tensor).float().mean()
                binary_acc = 1.0 if torch.all(pred == out_tensor) else 0.0

                total_pixel_acc += pixel_acc.item()
                total_binary_acc += binary_acc
                n_examples += 1

    # Restore training weights if EMA was used
    if use_ema and ema is not None:
        ema.restore()

    return {
        'pixel_accuracy': total_pixel_acc / n_examples,
        'binary_accuracy': total_binary_acc / n_examples,
        'n_examples': n_examples
    }


def compare_on_dataset(
    baseline_model: NeuralSymbolicARCSolver,
    trm_model: TRMNeuralSymbolicSolver,
    tasks: List,
    device: torch.device,
    dataset_name: str,
    trm_ema: EMA = None
):
    """Compare baseline and TRM on dataset.

    Args:
        baseline_model: Baseline neural-symbolic solver
        trm_model: TRM-enhanced solver
        tasks: List of tasks to evaluate
        device: Device
        dataset_name: Name for logging
        trm_ema: EMA for TRM model
    """
    print(f"\n{'='*70}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*70}")
    print(f"Number of tasks: {len(tasks)}")

    # Count examples
    n_examples = sum(len(task.input_examples) for task in tasks)
    print(f"Number of examples: {n_examples}")

    # Evaluate baseline
    print("\nBaseline (large CNN, no recursion):")
    baseline_metrics = evaluate_model(baseline_model, tasks, device)
    print(f"  Pixel accuracy: {baseline_metrics['pixel_accuracy']:.2%}")
    print(f"  Binary accuracy: {baseline_metrics['binary_accuracy']:.2%}")

    # Evaluate TRM
    print("\nTRM (tiny 2-layer, recursive refinement, EMA):")
    trm_metrics = evaluate_model(trm_model, tasks, device, use_ema=True, ema=trm_ema)
    print(f"  Pixel accuracy: {trm_metrics['pixel_accuracy']:.2%}")
    print(f"  Binary accuracy: {trm_metrics['binary_accuracy']:.2%}")

    # Comparison
    pixel_improvement = trm_metrics['pixel_accuracy'] - baseline_metrics['pixel_accuracy']
    binary_improvement = trm_metrics['binary_accuracy'] - baseline_metrics['binary_accuracy']

    print(f"\nImprovement:")
    print(f"  Pixel: {pixel_improvement:+.2%} ({'↑' if pixel_improvement > 0 else '↓'})")
    print(f"  Binary: {binary_improvement:+.2%} ({'↑' if binary_improvement > 0 else '↓'})")

    if binary_improvement > 0:
        relative_improvement = (binary_improvement / (baseline_metrics['binary_accuracy'] + 1e-8)) * 100
        print(f"  Relative improvement: {relative_improvement:.1f}x")

    return {
        'baseline': baseline_metrics,
        'trm': trm_metrics,
        'improvement': {
            'pixel': pixel_improvement,
            'binary': binary_improvement
        }
    }


def visualize_comparison(results: Dict[str, Dict]):
    """Visualize comparison results.

    Args:
        results: Dictionary mapping dataset_name → comparison results
    """
    datasets = list(results.keys())
    baseline_pixel = [results[d]['baseline']['pixel_accuracy'] * 100 for d in datasets]
    trm_pixel = [results[d]['trm']['pixel_accuracy'] * 100 for d in datasets]
    baseline_binary = [results[d]['baseline']['binary_accuracy'] * 100 for d in datasets]
    trm_binary = [results[d]['trm']['binary_accuracy'] * 100 for d in datasets]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pixel accuracy
    x = np.arange(len(datasets))
    width = 0.35

    ax1.bar(x - width/2, baseline_pixel, width, label='Baseline', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, trm_pixel, width, label='TRM', alpha=0.8, color='coral')
    ax1.set_ylabel('Pixel Accuracy (%)')
    ax1.set_title('Pixel Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Binary accuracy
    ax2.bar(x - width/2, baseline_binary, width, label='Baseline', alpha=0.8, color='steelblue')
    ax2.bar(x + width/2, trm_binary, width, label='TRM', alpha=0.8, color='coral')
    ax2.set_ylabel('Binary Accuracy (%)')
    ax2.set_title('Binary Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / "trm_baseline_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")

    plt.show()


def main():
    """Run comparison."""

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (no GPU detected)")

    print(f"\n{'='*70}")
    print(f"TRM vs Baseline Neural-Symbolic Comparison")
    print(f"{'='*70}")

    # Load datasets
    print("\nLoading datasets...")
    mini_arc = load_mini_arc()
    arc_agi_1 = load_arc_agi_1()

    # Use subset for quick comparison
    mini_arc_test = mini_arc[:20]
    arc_agi_1_test = arc_agi_1[:20]

    print(f"  Mini-ARC test: {len(mini_arc_test)} tasks")
    print(f"  ARC-AGI-1 test: {len(arc_agi_1_test)} tasks")

    # Create models
    print("\nInitializing models...")

    # Baseline
    baseline = NeuralSymbolicARCSolver(
        num_colors=10,
        feature_dim=64,
        device=device,
        max_composite_depth=2
    ).to(device)

    # TRM
    trm = TRMNeuralSymbolicSolver(
        num_colors=10,
        answer_dim=128,
        latent_dim=64,
        num_cycles=3,
        device=device,
        max_composite_depth=2
    ).to(device)

    # Create EMA for TRM
    trm_ema = EMA(trm, decay=0.999)

    # Load checkpoints if available
    baseline_checkpoint = Path("checkpoints/baseline/latest.pt")
    trm_checkpoint = Path("checkpoints/trm/latest.pt")

    if baseline_checkpoint.exists():
        print(f"Loading baseline checkpoint: {baseline_checkpoint}")
        checkpoint = torch.load(baseline_checkpoint, map_location=device)
        baseline.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("⚠️  No baseline checkpoint found (using untrained model)")

    if trm_checkpoint.exists():
        print(f"Loading TRM checkpoint: {trm_checkpoint}")
        checkpoint = torch.load(trm_checkpoint, map_location=device)
        trm.load_state_dict(checkpoint['model_state_dict'])
        trm_ema.shadow = checkpoint.get('ema_shadow', trm_ema.shadow)
    else:
        print("⚠️  No TRM checkpoint found (using untrained model)")

    # Compare on datasets
    results = {}

    results['Mini-ARC'] = compare_on_dataset(
        baseline, trm, mini_arc_test, device, "Mini-ARC (20 tasks)", trm_ema
    )

    results['ARC-AGI-1'] = compare_on_dataset(
        baseline, trm, arc_agi_1_test, device, "ARC-AGI-1 (20 tasks)", trm_ema
    )

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        print(f"  Baseline: {result['baseline']['binary_accuracy']:.2%} binary")
        print(f"  TRM:      {result['trm']['binary_accuracy']:.2%} binary")
        print(f"  Δ:        {result['improvement']['binary']:+.2%}")

    # Visualize
    print("\nGenerating visualization...")
    visualize_comparison(results)

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
