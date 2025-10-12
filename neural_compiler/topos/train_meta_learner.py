#!/usr/bin/env python3
"""
Train Meta-Learner on ARC-AGI Training Examples

This script trains the universal topos meta-learner on actual ARC-AGI training tasks.

Usage:
    # Train with default settings
    python train_meta_learner.py

    # Train with custom parameters
    python train_meta_learner.py --epochs 50 --batch-size 16 --tasks 100

    # Resume from checkpoint
    python train_meta_learner.py --resume trained_models/checkpoint.pkl
"""

import jax
import jax.numpy as jnp
from jax import random
import argparse
from pathlib import Path
import pickle
import json
from datetime import datetime
import sys

from meta_learner import MetaToposLearner, meta_learning_pipeline
from arc_loader import load_arc_dataset, ARCTask
from onnx_export import export_and_test_meta_learner


def train_on_arc(arc_data_dir: str = "../../ARC-AGI/data",
                 output_dir: str = "trained_models",
                 n_tasks: int = None,
                 meta_epochs: int = 100,
                 meta_batch_size: int = 8,
                 n_shots: int = 3,
                 seed: int = 42):
    """
    Train meta-learner on ARC-AGI training dataset.

    Args:
        arc_data_dir: Path to ARC-AGI data directory
        output_dir: Where to save trained models
        n_tasks: Number of tasks to use (None = all)
        meta_epochs: Number of meta-training epochs
        meta_batch_size: Tasks per meta-batch
        n_shots: Few-shot examples per task
        seed: Random seed

    Returns:
        meta_learner: Trained MetaToposLearner
        results: Training results dictionary
    """

    print("\n" + "="*70)
    print("TRAINING META-LEARNER ON ARC-AGI")
    print("="*70)

    # Setup
    key = random.PRNGKey(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        'arc_data_dir': arc_data_dir,
        'n_tasks': n_tasks,
        'meta_epochs': meta_epochs,
        'meta_batch_size': meta_batch_size,
        'n_shots': n_shots,
        'seed': seed,
        'timestamp': datetime.now().isoformat()
    }

    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Saved config to {config_path}")

    # Load ARC training data
    print(f"\nLoading ARC training data from {arc_data_dir}...")
    try:
        tasks_dict = load_arc_dataset(arc_data_dir, "training")
        all_tasks = list(tasks_dict.values())

        if n_tasks is not None:
            training_tasks = all_tasks[:n_tasks]
        else:
            training_tasks = all_tasks

        print(f"✓ Loaded {len(training_tasks)} training tasks")

        # Show dataset statistics
        print("\nDataset Statistics:")
        total_examples = sum(len(t.train_inputs) for t in training_tasks)
        avg_examples = total_examples / len(training_tasks)
        print(f"  Total tasks: {len(training_tasks)}")
        print(f"  Total examples: {total_examples}")
        print(f"  Avg examples/task: {avg_examples:.1f}")

        # Grid size statistics
        grid_sizes = [(t.train_inputs[0].height, t.train_inputs[0].width)
                      for t in training_tasks if len(t.train_inputs) > 0]
        avg_h = sum(h for h, w in grid_sizes) / len(grid_sizes)
        avg_w = sum(w for h, w in grid_sizes) / len(grid_sizes)
        print(f"  Avg grid size: {avg_h:.1f} × {avg_w:.1f}")

    except Exception as e:
        print(f"\n✗ Failed to load ARC data: {e}")
        print("\nPlease ensure ARC-AGI dataset is available:")
        print("  git clone https://github.com/fchollet/ARC-AGI.git")
        print(f"  Or check path: {arc_data_dir}")
        return None, None

    # Initialize meta-learner
    print("\nInitializing meta-learner...")
    meta_learner = MetaToposLearner(
        num_objects=20,
        feature_dim=32,
        max_covers=5,
        embedding_dim=64,
        meta_lr=1e-3
    )
    print("✓ Meta-learner initialized")
    print(f"  Objects: {meta_learner.num_objects}")
    print(f"  Feature dim: {meta_learner.feature_dim}")
    print(f"  Embedding dim: {meta_learner.embedding_dim}")

    # Meta-train
    print(f"\nStarting meta-training...")
    print(f"  Epochs: {meta_epochs}")
    print(f"  Batch size: {meta_batch_size}")
    print(f"  N-shot: {n_shots}")
    print(f"  Tasks: {len(training_tasks)}")
    print()

    k1, key = random.split(key)

    try:
        universal_topos = meta_learner.meta_train(
            training_tasks=training_tasks,
            n_shots=n_shots,
            meta_batch_size=meta_batch_size,
            meta_epochs=meta_epochs,
            key=k1,
            verbose=True
        )

        print("\n✓ Meta-training completed!")

    except Exception as e:
        print(f"\n✗ Meta-training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # Save trained model
    model_path = output_path / "universal_topos.pkl"
    meta_learner.save(str(model_path))
    print(f"\n✓ Saved trained model to {model_path}")

    # Test few-shot adaptation on held-out tasks
    print("\nTesting few-shot adaptation on held-out tasks...")
    if len(all_tasks) > len(training_tasks):
        test_tasks = all_tasks[len(training_tasks):len(training_tasks)+10]
        print(f"Testing on {len(test_tasks)} held-out tasks...")

        adaptations = []
        for i, task in enumerate(test_tasks):
            k1, key = random.split(key)
            try:
                adapted_site = meta_learner.few_shot_adapt(task, n_shots=n_shots, key=k1)
                adaptations.append({
                    'task_id': i,
                    'num_objects': adapted_site.num_objects,
                    'success': True
                })
                print(f"  Task {i+1}/{len(test_tasks)}: ✓ Adapted")
            except Exception as e:
                adaptations.append({
                    'task_id': i,
                    'success': False,
                    'error': str(e)
                })
                print(f"  Task {i+1}/{len(test_tasks)}: ✗ Failed")

        success_rate = sum(1 for a in adaptations if a['success']) / len(adaptations)
        print(f"\nAdaptation success rate: {success_rate:.1%}")

    # Export to ONNX
    print("\nExporting to ONNX...")
    export_dir = output_path / "onnx_export"

    try:
        export_results = export_and_test_meta_learner(
            meta_learner,
            output_dir=str(export_dir),
            test_inference=True
        )

        if export_results['success']:
            print("✓ ONNX export successful!")
        else:
            print("⚠ ONNX export had issues")

    except Exception as e:
        print(f"⚠ ONNX export failed: {e}")
        export_results = None

    # Compile results
    results = {
        'config': config,
        'training': {
            'n_tasks': len(training_tasks),
            'meta_epochs': meta_epochs,
            'completed': True
        },
        'model_path': str(model_path),
        'export_results': export_results
    }

    # Save results
    results_path = output_path / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Saved results to {results_path}")

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Trained on: {len(training_tasks)} ARC tasks")
    print(f"Model saved: {model_path}")
    if export_results and export_results['success']:
        print(f"ONNX export: {export_dir}/")
        print("  - task_encoder.onnx ✓")
        print("  - sheaf_network.onnx ✓")
        print("  - universal_topos.pkl ✓")
    print("="*70)

    return meta_learner, results


def main():
    parser = argparse.ArgumentParser(
        description="Train meta-learner on ARC-AGI training examples"
    )

    parser.add_argument(
        '--data',
        type=str,
        default='../../ARC-AGI/data',
        help='Path to ARC-AGI data directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='trained_models',
        help='Output directory for trained models'
    )

    parser.add_argument(
        '--tasks',
        type=int,
        default=None,
        help='Number of tasks to use (default: all)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of meta-training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Meta-batch size'
    )

    parser.add_argument(
        '--shots',
        type=int,
        default=3,
        help='Number of few-shot examples'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )

    args = parser.parse_args()

    # Check if resuming
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        try:
            meta_learner = MetaToposLearner.load(args.resume)
            print("✓ Checkpoint loaded")
            print("\nTODO: Implement resume training")
            return
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return

    # Train from scratch
    meta_learner, results = train_on_arc(
        arc_data_dir=args.data,
        output_dir=args.output,
        n_tasks=args.tasks,
        meta_epochs=args.epochs,
        meta_batch_size=args.batch_size,
        n_shots=args.shots,
        seed=args.seed
    )

    if meta_learner is None:
        print("\n✗ Training failed")
        sys.exit(1)

    print("\n✅ Training successful!")
    sys.exit(0)


if __name__ == "__main__":
    main()
