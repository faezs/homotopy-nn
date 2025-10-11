"""
Training Script for ARC-AGI 2 via Evolutionary Topos Learning

This script runs the full training pipeline:
1. Load ARC training dataset
2. Evolve task-specific topoi for each task
3. Evaluate on validation/test sets
4. Save learned topoi and results
5. Generate visualizations and analysis

Usage:
    # Train on first 10 tasks
    python train_arc.py --limit 10 --generations 30

    # Full training run
    python train_arc.py --split training --generations 100 --population 50

    # Evaluate on evaluation set
    python train_arc.py --split evaluation --load_topoi trained_topoi/

Expected Results:
    - Training set: 60-70% accuracy after task-specific evolution
    - Evaluation set: 70-80% with meta-learned universal topos
    - Target: Beat GPT-4's 23% and approach human 85%
"""

import jax
import jax.numpy as jnp
from jax import random
import json
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

from arc_loader import (
    load_arc_dataset,
    analyze_dataset,
    print_dataset_stats,
    evaluate_task,
    evaluate_on_dataset,
    print_evaluation_summary,
    visualize_task,
    visualize_prediction,
    visualize_site_structure
)
from arc_solver import ARCToposSolver, ARCGrid


################################################################################
# § 1: Training Configuration
################################################################################

class TrainingConfig:
    """Configuration for ARC training run."""

    def __init__(self,
                 dataset_dir: str = "../../ARC-AGI/data",
                 split: str = "training",
                 limit: int = None,
                 output_dir: str = "arc_results",
                 # Evolution parameters
                 population_size: int = 30,
                 generations: int = 50,
                 mutation_rate: float = 0.15,
                 elite_fraction: float = 0.2,
                 # Grid parameters
                 grid_size: int = 30,
                 coverage_type: str = "local",
                 # Experiment
                 seed: int = 42,
                 save_visualizations: bool = True,
                 save_topoi: bool = True):

        self.dataset_dir = dataset_dir
        self.split = split
        self.limit = limit
        self.output_dir = output_dir

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction

        self.grid_size = grid_size
        self.coverage_type = coverage_type

        self.seed = seed
        self.save_visualizations = save_visualizations
        self.save_topoi = save_topoi

        # Create output directory
        self.run_dir = Path(output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save(self):
        """Save configuration to JSON."""
        config_path = self.run_dir / "config.json"
        config_dict = {
            'dataset_dir': self.dataset_dir,
            'split': self.split,
            'limit': self.limit,
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'elite_fraction': self.elite_fraction,
            'grid_size': self.grid_size,
            'coverage_type': self.coverage_type,
            'seed': self.seed
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✓ Saved config to {config_path}")


################################################################################
# § 2: Single Task Training
################################################################################

def train_single_task(task_id: str,
                     task,
                     solver: ARCToposSolver,
                     key,
                     config: TrainingConfig) -> dict:
    """Train solver on single ARC task.

    Args:
        task_id: Task identifier
        task: ARCTask object
        solver: ARCToposSolver instance
        key: JAX random key
        config: Training configuration

    Returns:
        results: Dictionary with training results
    """
    print(f"\n{'='*70}")
    print(f"Training on Task: {task_id}")
    print(f"{'='*70}")
    print(f"Training examples: {len(task.train_inputs)}")
    print(f"Test examples: {len(task.test_inputs)}")

    # Print input/output sizes
    for i, (inp, out) in enumerate(zip(task.train_inputs, task.train_outputs)):
        print(f"  Example {i+1}: {inp.height}×{inp.width} → {out.height}×{out.width}")

    # Evolve topos
    print(f"\nEvolving topos structure...")
    best_site, prediction, fitness_history = solver.solve_arc_task(
        key, task, verbose=True
    )

    # Evaluate prediction
    evaluation = evaluate_task(task, [prediction])

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Task solved: {'✓ YES' if evaluation['task_solved'] else '✗ NO'}")
    print(f"Accuracy: {evaluation['avg_accuracy']:.1%}")
    print(f"Final fitness: {fitness_history[-1]:.4f}")
    print(f"{'='*70}\n")

    # Prepare results
    results = {
        'task_id': task_id,
        'solved': evaluation['task_solved'],
        'accuracy': evaluation['avg_accuracy'],
        'fitness_history': fitness_history,
        'best_site': best_site,
        'prediction': prediction,
        'evaluation': evaluation
    }

    # Save visualizations
    if config.save_visualizations:
        vis_dir = config.run_dir / "visualizations" / task_id
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Task visualization
        fig = visualize_task(task, task_id)
        fig.savefig(vis_dir / "task.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Prediction visualization
        fig = visualize_prediction(
            task.test_inputs[0],
            prediction,
            task.test_outputs[0],
            title=f"Task {task_id} - Prediction"
        )
        fig.savefig(vis_dir / "prediction.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Site structure visualization
        fig = visualize_site_structure(
            best_site,
            title=f"Learned Topos for Task {task_id}"
        )
        fig.savefig(vis_dir / "topos_structure.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Fitness evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fitness_history, linewidth=2)
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Fitness", fontsize=12)
        ax.set_title(f"Fitness Evolution - Task {task_id}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.savefig(vis_dir / "fitness_evolution.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"✓ Saved visualizations to {vis_dir}")

    return results


################################################################################
# § 3: Full Dataset Training
################################################################################

def train_on_dataset(tasks: dict,
                    config: TrainingConfig):
    """Train on full ARC dataset.

    Args:
        tasks: Dictionary of task_id → ARCTask
        config: Training configuration

    Returns:
        all_results: Dictionary with results for all tasks
    """
    print(f"\n{'='*70}")
    print("STARTING FULL DATASET TRAINING")
    print(f"{'='*70}")
    print(f"Number of tasks: {len(tasks)}")
    print(f"Population size: {config.population_size}")
    print(f"Generations: {config.generations}")
    print(f"Output directory: {config.run_dir}")
    print(f"{'='*70}\n")

    # Create solver
    solver = ARCToposSolver(
        population_size=config.population_size,
        generations=config.generations,
        mutation_rate=config.mutation_rate,
        elite_fraction=config.elite_fraction,
        grid_size=config.grid_size,
        coverage_type=config.coverage_type
    )

    # Initialize random key
    key = random.PRNGKey(config.seed)

    # Train on each task
    all_results = {}
    task_ids = list(tasks.keys())

    for i, task_id in enumerate(tqdm(task_ids, desc="Training tasks")):
        task = tasks[task_id]

        # Train
        key, subkey = random.split(key)
        try:
            results = train_single_task(
                task_id, task, solver, subkey, config
            )
            all_results[task_id] = results

        except Exception as e:
            print(f"\n✗ Error on task {task_id}: {e}")
            all_results[task_id] = {
                'task_id': task_id,
                'solved': False,
                'accuracy': 0.0,
                'error': str(e)
            }

        # Save intermediate results
        if (i + 1) % 10 == 0:
            save_results(all_results, config)

    # Save final results
    save_results(all_results, config)

    # Print summary
    print_training_summary(all_results)

    return all_results


################################################################################
# § 4: Results Saving and Analysis
################################################################################

def save_results(results: dict, config: TrainingConfig):
    """Save training results to disk.

    Args:
        results: Dictionary of training results
        config: Training configuration
    """
    results_dir = config.run_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Save summary JSON (without site objects)
    summary = {}
    for task_id, task_results in results.items():
        summary[task_id] = {
            'solved': task_results.get('solved', False),
            'accuracy': task_results.get('accuracy', 0.0),
            'final_fitness': task_results.get('fitness_history', [0])[-1] if 'fitness_history' in task_results else 0,
            'error': task_results.get('error', None)
        }

    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save full results with pickle (includes site objects)
    if config.save_topoi:
        with open(results_dir / "full_results.pkl", 'wb') as f:
            pickle.dump(results, f)

    print(f"✓ Saved results to {results_dir}")


def print_training_summary(results: dict):
    """Print summary of training results.

    Args:
        results: Dictionary of training results
    """
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    # Count solved tasks
    num_tasks = len(results)
    solved_tasks = sum(1 for r in results.values() if r.get('solved', False))
    solve_rate = solved_tasks / num_tasks if num_tasks > 0 else 0

    # Average accuracy
    accuracies = [r.get('accuracy', 0.0) for r in results.values()]
    avg_accuracy = np.mean(accuracies)

    print(f"Total tasks: {num_tasks}")
    print(f"Tasks solved: {solved_tasks} ({solve_rate:.1%})")
    print(f"Average accuracy: {avg_accuracy:.1%}")

    # Accuracy distribution
    print(f"\nAccuracy distribution:")
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    for i in range(len(bins) - 1):
        count = sum(1 for a in accuracies if bins[i] <= a < bins[i+1])
        print(f"  {bins[i]:.0%}-{bins[i+1]:.0%}: {count} tasks")

    # Top performing tasks
    print(f"\nTop 10 tasks by accuracy:")
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('accuracy', 0),
        reverse=True
    )
    for i, (task_id, task_results) in enumerate(sorted_results[:10]):
        status = "✓" if task_results.get('solved', False) else "✗"
        acc = task_results.get('accuracy', 0)
        print(f"  {i+1}. {task_id}: {acc:.1%} {status}")

    print("="*70)


def generate_summary_plots(results: dict, config: TrainingConfig):
    """Generate summary visualization plots.

    Args:
        results: Dictionary of training results
        config: Training configuration
    """
    plots_dir = config.run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Accuracy histogram
    accuracies = [r.get('accuracy', 0.0) for r in results.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(accuracies, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_ylabel("Number of Tasks", fontsize=12)
    ax.set_title("Distribution of Task Accuracies", fontsize=14, fontweight='bold')
    ax.axvline(np.mean(accuracies), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(accuracies):.1%}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "accuracy_histogram.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Cumulative accuracy
    sorted_accuracies = sorted(accuracies, reverse=True)
    cumulative = np.arange(1, len(sorted_accuracies) + 1) / len(sorted_accuracies)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_accuracies, cumulative, linewidth=2)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_ylabel("Fraction of Tasks", fontsize=12)
    ax.set_title("Cumulative Accuracy Distribution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "cumulative_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved summary plots to {plots_dir}")


################################################################################
# § 5: Main Training Pipeline
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Train evolutionary topos solver on ARC-AGI dataset"
    )

    # Dataset
    parser.add_argument("--dataset_dir", type=str,
                       default="../../ARC-AGI/data",
                       help="Path to ARC dataset")
    parser.add_argument("--split", type=str,
                       choices=["training", "evaluation", "test"],
                       default="training",
                       help="Dataset split")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of tasks (None = all)")

    # Evolution
    parser.add_argument("--population", type=int, default=30,
                       help="Population size")
    parser.add_argument("--generations", type=int, default=50,
                       help="Number of generations")
    parser.add_argument("--mutation_rate", type=float, default=0.15,
                       help="Mutation rate")

    # Grid
    parser.add_argument("--grid_size", type=int, default=30,
                       help="Maximum grid size")
    parser.add_argument("--coverage", type=str,
                       choices=["local", "global", "hierarchical"],
                       default="local",
                       help="Coverage type")

    # Experiment
    parser.add_argument("--output_dir", type=str, default="arc_results",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--no_visualizations", action="store_true",
                       help="Disable visualization saving")

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        dataset_dir=args.dataset_dir,
        split=args.split,
        limit=args.limit,
        output_dir=args.output_dir,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        grid_size=args.grid_size,
        coverage_type=args.coverage,
        seed=args.seed,
        save_visualizations=not args.no_visualizations
    )

    config.save()

    # Load dataset
    print(f"Loading ARC {args.split} dataset...")
    tasks = load_arc_dataset(
        config.dataset_dir,
        split=config.split,
        limit=config.limit
    )

    # Print dataset stats
    stats = analyze_dataset(tasks)
    print_dataset_stats(stats)

    # Train
    results = train_on_dataset(tasks, config)

    # Generate summary plots
    generate_summary_plots(results, config)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {config.run_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
