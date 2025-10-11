"""
ARC Dataset Loader and Utilities

This module loads the official ARC-AGI dataset from JSON format and converts
it to our ARCTask representation for topos-based solving.

Official ARC Dataset Format:
    Each task is a JSON file with structure:
    {
        "train": [
            {"input": [[...]], "output": [[...]]},
            ...
        ],
        "test": [
            {"input": [[...]], "output": [[...]]}
        ]
    }

Our Format:
    ARCTask(
        train_inputs: List[ARCGrid],
        train_outputs: List[ARCGrid],
        test_inputs: List[ARCGrid],
        test_outputs: List[ARCGrid]
    )

Usage:
    # Load all training tasks
    tasks = load_arc_dataset("data/training")

    # Solve first task
    solver = ARCToposSolver()
    site, prediction = solver.solve_arc_task(key, tasks[0])

    # Visualize
    visualize_task(tasks[0])
    visualize_prediction(tasks[0].test_inputs[0], prediction)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass

from arc_solver import ARCGrid, ARCTask


################################################################################
# § 1: JSON Parsing
################################################################################

def parse_arc_json(json_path: str) -> ARCTask:
    """Parse a single ARC task from JSON file.

    Args:
        json_path: Path to JSON task file

    Returns:
        task: Parsed ARCTask

    Example:
        >>> task = parse_arc_json("data/training/00d62c1b.json")
        >>> print(f"Training examples: {len(task.train_inputs)}")
        Training examples: 3
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Parse training examples
    train_inputs = []
    train_outputs = []
    for example in data['train']:
        inp_grid = ARCGrid.from_array(np.array(example['input']))
        out_grid = ARCGrid.from_array(np.array(example['output']))
        train_inputs.append(inp_grid)
        train_outputs.append(out_grid)

    # Parse test examples
    test_inputs = []
    test_outputs = []
    for example in data['test']:
        inp_grid = ARCGrid.from_array(np.array(example['input']))
        # Test output might not be available (evaluation set)
        if 'output' in example and example['output'] is not None:
            out_grid = ARCGrid.from_array(np.array(example['output']))
        else:
            # Placeholder for evaluation set (no ground truth)
            out_grid = ARCGrid.from_array(np.zeros_like(example['input']))
        test_inputs.append(inp_grid)
        test_outputs.append(out_grid)

    return ARCTask(
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        test_inputs=test_inputs,
        test_outputs=test_outputs
    )


def load_arc_dataset(dataset_dir: str,
                     split: str = "training",
                     limit: Optional[int] = None) -> Dict[str, ARCTask]:
    """Load entire ARC dataset split.

    Args:
        dataset_dir: Root directory of ARC dataset
        split: One of {"training", "evaluation", "test"}
        limit: Maximum number of tasks to load (None = all)

    Returns:
        tasks: Dictionary mapping task_id → ARCTask

    Example:
        >>> tasks = load_arc_dataset("ARC-AGI/data", split="training", limit=10)
        >>> print(f"Loaded {len(tasks)} tasks")
        Loaded 10 tasks
    """
    split_dir = Path(dataset_dir) / split

    if not split_dir.exists():
        raise ValueError(f"Dataset directory not found: {split_dir}\n"
                        f"Please clone ARC dataset:\n"
                        f"  git clone https://github.com/fchollet/ARC-AGI.git")

    # Find all JSON files
    json_files = sorted(split_dir.glob("*.json"))

    if limit is not None:
        json_files = json_files[:limit]

    # Parse each task
    tasks = {}
    for json_path in json_files:
        task_id = json_path.stem  # Filename without .json
        try:
            task = parse_arc_json(str(json_path))
            tasks[task_id] = task
        except Exception as e:
            print(f"Warning: Failed to parse {task_id}: {e}")
            continue

    print(f"✓ Loaded {len(tasks)} {split} tasks from {dataset_dir}")
    return tasks


################################################################################
# § 2: Dataset Statistics
################################################################################

def analyze_dataset(tasks: Dict[str, ARCTask]) -> Dict:
    """Compute statistics about ARC dataset.

    Args:
        tasks: Dictionary of ARCTask objects

    Returns:
        stats: Dictionary with statistics
    """
    stats = {
        'num_tasks': len(tasks),
        'train_examples_per_task': [],
        'test_examples_per_task': [],
        'grid_sizes': [],
        'colors_used': set(),
        'max_colors_per_grid': []
    }

    for task_id, task in tasks.items():
        # Training examples
        stats['train_examples_per_task'].append(len(task.train_inputs))
        stats['test_examples_per_task'].append(len(task.test_inputs))

        # Grid sizes and colors
        for grid in task.train_inputs + task.train_outputs + task.test_inputs:
            stats['grid_sizes'].append((grid.height, grid.width))
            colors = set(grid.cells.flatten().tolist())
            stats['colors_used'].update(colors)
            stats['max_colors_per_grid'].append(len(colors))

    # Compute summary statistics
    stats['avg_train_examples'] = np.mean(stats['train_examples_per_task'])
    stats['avg_test_examples'] = np.mean(stats['test_examples_per_task'])
    stats['min_grid_size'] = min(min(h, w) for h, w in stats['grid_sizes'])
    stats['max_grid_size'] = max(max(h, w) for h, w in stats['grid_sizes'])
    stats['avg_colors_per_grid'] = np.mean(stats['max_colors_per_grid'])
    stats['total_colors_used'] = len(stats['colors_used'])

    return stats


def print_dataset_stats(stats: Dict):
    """Pretty-print dataset statistics.

    Args:
        stats: Statistics dictionary from analyze_dataset()
    """
    print("=" * 70)
    print("ARC DATASET STATISTICS")
    print("=" * 70)
    print(f"Number of tasks: {stats['num_tasks']}")
    print(f"Training examples per task: {stats['avg_train_examples']:.1f} (avg)")
    print(f"Test examples per task: {stats['avg_test_examples']:.1f} (avg)")
    print(f"Grid size range: {stats['min_grid_size']} - {stats['max_grid_size']}")
    print(f"Colors per grid: {stats['avg_colors_per_grid']:.1f} (avg)")
    print(f"Total colors used: {stats['total_colors_used']}")
    print("=" * 70)


################################################################################
# § 3: Visualization
################################################################################

# ARC color palette (official)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Grey
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Sky blue
    '#870C25',  # 9: Brown
]


def visualize_grid(grid: ARCGrid, ax=None, title: str = ""):
    """Visualize a single ARC grid.

    Args:
        grid: ARCGrid to visualize
        ax: Matplotlib axis (creates new if None)
        title: Title for the plot

    Returns:
        ax: Matplotlib axis with grid rendered
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Create colored grid
    height, width = grid.height, grid.width

    # Draw each cell
    for i in range(height):
        for j in range(width):
            color_idx = int(grid.cells[i, j])
            color = ARC_COLORS[color_idx]

            rect = patches.Rectangle(
                (j, height - i - 1),  # Bottom-left corner (flip y)
                1, 1,
                linewidth=1,
                edgecolor='gray',
                facecolor=color
            )
            ax.add_patch(rect)

    # Set axis properties
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(True, which='both', color='gray', linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')

    return ax


def visualize_task(task: ARCTask, task_id: str = "Unknown"):
    """Visualize entire ARC task (training + test examples).

    Args:
        task: ARCTask to visualize
        task_id: Task identifier for title

    Returns:
        fig: Matplotlib figure
    """
    num_train = len(task.train_inputs)
    num_test = len(task.test_inputs)

    # Create grid of subplots
    fig, axes = plt.subplots(
        num_train + num_test,
        2,
        figsize=(12, 4 * (num_train + num_test))
    )

    # Handle single-row case
    if num_train + num_test == 1:
        axes = axes.reshape(1, -1)

    # Plot training examples
    for i in range(num_train):
        visualize_grid(
            task.train_inputs[i],
            ax=axes[i, 0],
            title=f"Training {i+1} - Input"
        )
        visualize_grid(
            task.train_outputs[i],
            ax=axes[i, 1],
            title=f"Training {i+1} - Output"
        )

    # Plot test examples
    for i in range(num_test):
        row_idx = num_train + i
        visualize_grid(
            task.test_inputs[i],
            ax=axes[row_idx, 0],
            title=f"Test {i+1} - Input"
        )
        visualize_grid(
            task.test_outputs[i],
            ax=axes[row_idx, 1],
            title=f"Test {i+1} - Output (Ground Truth)"
        )

    fig.suptitle(f"ARC Task: {task_id}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_prediction(input_grid: ARCGrid,
                        predicted_grid: ARCGrid,
                        ground_truth: Optional[ARCGrid] = None,
                        title: str = "Prediction"):
    """Visualize prediction vs ground truth.

    Args:
        input_grid: Test input
        predicted_grid: Model prediction
        ground_truth: Ground truth output (if available)
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    ncols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    visualize_grid(input_grid, ax=axes[0], title="Input")
    visualize_grid(predicted_grid, ax=axes[1], title="Predicted Output")

    if ground_truth is not None:
        visualize_grid(ground_truth, ax=axes[2], title="Ground Truth")

        # Compute accuracy
        correct = np.sum(predicted_grid.cells == ground_truth.cells)
        total = ground_truth.height * ground_truth.width
        accuracy = correct / total * 100

        fig.suptitle(
            f"{title} - Accuracy: {accuracy:.1f}% ({correct}/{total} cells)",
            fontsize=14,
            fontweight='bold'
        )
    else:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def visualize_site_structure(site, title: str = "Learned Topos Structure"):
    """Visualize the learned site structure (category + coverage).

    Args:
        site: Site object from evolutionary solver
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Visualize adjacency (category morphisms)
    ax1 = axes[0]
    im1 = ax1.imshow(site.adjacency, cmap='viridis', interpolation='nearest')
    ax1.set_title("Category Structure (Morphisms)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Target Object")
    ax1.set_ylabel("Source Object")
    plt.colorbar(im1, ax=ax1, label="Connection Strength")

    # 2. Visualize coverage weights (first cover for each object)
    ax2 = axes[1]
    # Show first covering family for each object
    coverage_viz = site.coverage_weights[:, 0, :]  # (num_objects, num_objects)
    im2 = ax2.imshow(coverage_viz, cmap='plasma', interpolation='nearest')
    ax2.set_title("Coverage Structure (First Cover)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Covered By Object")
    ax2.set_ylabel("Object")
    plt.colorbar(im2, ax=ax2, label="Coverage Weight")

    # Add statistics
    sparsity = np.sum(site.adjacency) / (site.num_objects ** 2)
    fig.suptitle(
        f"{title}\nObjects: {site.num_objects}, "
        f"Sparsity: {sparsity:.1%}, Covers: {site.max_covers}",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    return fig


################################################################################
# § 4: Evaluation Utilities
################################################################################

def evaluate_prediction(predicted: ARCGrid, ground_truth: ARCGrid) -> Dict:
    """Evaluate prediction accuracy.

    Args:
        predicted: Predicted grid
        ground_truth: Ground truth grid

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Exact match
    exact_match = np.array_equal(predicted.cells, ground_truth.cells)

    # Cell-wise accuracy
    correct_cells = np.sum(predicted.cells == ground_truth.cells)
    total_cells = ground_truth.height * ground_truth.width
    accuracy = correct_cells / total_cells

    # Size match
    size_match = (predicted.height == ground_truth.height and
                 predicted.width == ground_truth.width)

    return {
        'exact_match': exact_match,
        'accuracy': accuracy,
        'correct_cells': int(correct_cells),
        'total_cells': int(total_cells),
        'size_match': size_match,
        'predicted_size': (predicted.height, predicted.width),
        'ground_truth_size': (ground_truth.height, ground_truth.width)
    }


def evaluate_task(task: ARCTask,
                 predictions: List[ARCGrid]) -> Dict:
    """Evaluate all predictions for a task.

    Args:
        task: ARC task with ground truth
        predictions: List of predicted output grids

    Returns:
        results: Dictionary with aggregate metrics
    """
    if len(predictions) != len(task.test_outputs):
        raise ValueError(f"Expected {len(task.test_outputs)} predictions, "
                        f"got {len(predictions)}")

    # Evaluate each test example
    example_results = []
    for pred, gt in zip(predictions, task.test_outputs):
        metrics = evaluate_prediction(pred, gt)
        example_results.append(metrics)

    # Aggregate
    results = {
        'num_examples': len(predictions),
        'exact_matches': sum(r['exact_match'] for r in example_results),
        'avg_accuracy': np.mean([r['accuracy'] for r in example_results]),
        'size_matches': sum(r['size_match'] for r in example_results),
        'example_results': example_results
    }

    # Task solved if all examples are exact matches
    results['task_solved'] = results['exact_matches'] == results['num_examples']

    return results


################################################################################
# § 5: Batch Processing
################################################################################

def evaluate_on_dataset(tasks: Dict[str, ARCTask],
                       solver,
                       key,
                       verbose: bool = True) -> Dict:
    """Evaluate solver on entire dataset.

    Args:
        tasks: Dictionary of task_id → ARCTask
        solver: ARCToposSolver instance
        key: JAX random key
        verbose: Print progress

    Returns:
        results: Dictionary with aggregate results
    """
    import jax
    from tqdm import tqdm

    all_results = {}
    task_solved_count = 0
    total_accuracy = []

    task_items = list(tasks.items())
    iterator = tqdm(task_items) if verbose else task_items

    for task_id, task in iterator:
        try:
            # Solve task
            key, subkey = jax.random.split(key)
            best_site, prediction, fitness_history = solver.solve_arc_task(
                subkey, task, verbose=False
            )

            # Evaluate
            predictions = [prediction]  # First test example
            task_results = evaluate_task(task, predictions)

            all_results[task_id] = {
                'solved': task_results['task_solved'],
                'accuracy': task_results['avg_accuracy'],
                'best_fitness': fitness_history[-1] if fitness_history else 0.0,
                'site': best_site
            }

            if task_results['task_solved']:
                task_solved_count += 1

            total_accuracy.append(task_results['avg_accuracy'])

            if verbose:
                iterator.set_description(
                    f"Solved: {task_solved_count}/{len(all_results)} "
                    f"(Acc: {np.mean(total_accuracy):.1%})"
                )

        except Exception as e:
            if verbose:
                print(f"Error on task {task_id}: {e}")
            all_results[task_id] = {
                'solved': False,
                'accuracy': 0.0,
                'error': str(e)
            }

    # Aggregate statistics
    summary = {
        'num_tasks': len(tasks),
        'tasks_solved': task_solved_count,
        'solve_rate': task_solved_count / len(tasks),
        'avg_accuracy': np.mean(total_accuracy),
        'results_per_task': all_results
    }

    return summary


def print_evaluation_summary(results: Dict):
    """Pretty-print evaluation results.

    Args:
        results: Results from evaluate_on_dataset()
    """
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Tasks evaluated: {results['num_tasks']}")
    print(f"Tasks solved: {results['tasks_solved']} ({results['solve_rate']:.1%})")
    print(f"Average accuracy: {results['avg_accuracy']:.1%}")
    print("=" * 70)

    # Top tasks by accuracy
    print("\nTop 10 tasks by accuracy:")
    sorted_tasks = sorted(
        results['results_per_task'].items(),
        key=lambda x: x[1].get('accuracy', 0),
        reverse=True
    )
    for i, (task_id, task_results) in enumerate(sorted_tasks[:10]):
        status = "✓ SOLVED" if task_results['solved'] else "✗"
        print(f"  {i+1}. {task_id}: {task_results['accuracy']:.1%} {status}")


################################################################################
# § 6: Command-Line Interface
################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and visualize ARC dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../../ARC-AGI/data",
        help="Path to ARC dataset directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["training", "evaluation", "test"],
        default="training",
        help="Dataset split to load"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of tasks to load"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize first task"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ARC DATASET LOADER")
    print("=" * 70)
    print()

    # Load dataset
    print(f"Loading {args.split} split from {args.dataset_dir}...")
    tasks = load_arc_dataset(
        args.dataset_dir,
        split=args.split,
        limit=args.limit
    )
    print()

    # Print statistics
    if args.stats:
        stats = analyze_dataset(tasks)
        print_dataset_stats(stats)
        print()

    # Visualize first task
    if args.visualize and len(tasks) > 0:
        task_id, task = list(tasks.items())[0]
        print(f"Visualizing task: {task_id}")
        print(f"  Training examples: {len(task.train_inputs)}")
        print(f"  Test examples: {len(task.test_inputs)}")
        print()

        fig = visualize_task(task, task_id)
        plt.savefig(f"arc_task_{task_id}.png", dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to arc_task_{task_id}.png")
        plt.show()

    print("=" * 70)
    print(f"✓ Successfully loaded {len(tasks)} ARC tasks!")
    print("=" * 70)
