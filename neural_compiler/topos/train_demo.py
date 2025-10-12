#!/usr/bin/env python3
"""
Demo Training Script - Works without JAX

This demonstrates the meta-learning training loop on ARC-AGI data
using only numpy (available in the nix environment).

For full JAX/Flax version, install: pip install jax jaxlib flax optax
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("META-LEARNING DEMO (Numpy-based)")
print("="*70)

# Check for ARC data
# Path relative to this script's location
script_dir = Path(__file__).parent
arc_data_dir = script_dir / "../../ARC-AGI/data/training"
arc_path = arc_data_dir.resolve()

if not arc_path.exists():
    print(f"\n✗ ARC data not found at {arc_path}")
    print("\nPlease run: git clone https://github.com/fchollet/ARC-AGI.git")
    sys.exit(1)

print(f"\n✓ ARC data found at {arc_path}")

# Load ARC tasks
print("\nLoading ARC training tasks...")
task_files = list(arc_path.glob("*.json"))
print(f"Found {len(task_files)} task files")

# Load a few tasks as examples
tasks = []
for task_file in task_files[:20]:  # Just load 20 for demo
    with open(task_file, 'r') as f:
        task_data = json.load(f)
        tasks.append({
            'id': task_file.stem,
            'train': task_data['train'],
            'test': task_data['test']
        })

print(f"✓ Loaded {len(tasks)} tasks for demo")

# Show task statistics
print("\nTask Statistics:")
total_train_examples = sum(len(t['train']) for t in tasks)
total_test_examples = sum(len(t['test']) for t in tasks)
print(f"  Total train examples: {total_train_examples}")
print(f"  Total test examples: {total_test_examples}")
print(f"  Avg train examples/task: {total_train_examples/len(tasks):.1f}")

# Show example task
example_task = tasks[0]
print(f"\nExample task: {example_task['id']}")
print(f"  Train examples: {len(example_task['train'])}")
print(f"  Test examples: {len(example_task['test'])}")

if example_task['train']:
    first_input = np.array(example_task['train'][0]['input'])
    first_output = np.array(example_task['train'][0]['output'])
    print(f"  First train input shape: {first_input.shape}")
    print(f"  First train output shape: {first_output.shape}")

# Simulate meta-learning training loop
print("\n" + "="*70)
print("SIMULATING META-TRAINING")
print("="*70)

n_epochs = 10
n_tasks = len(tasks)
batch_size = 4
n_shots = 3

print(f"\nConfiguration:")
print(f"  Epochs: {n_epochs}")
print(f"  Tasks: {n_tasks}")
print(f"  Batch size: {batch_size}")
print(f"  N-shot: {n_shots}")

# Initialize "meta-parameters" (simplified)
meta_params = {
    'num_objects': 20,
    'feature_dim': 32,
    'embedding_dim': 64,
    'base_coverage': np.random.randn(20, 5, 20),  # (objects, covers, objects)
}

print(f"\nMeta-learner initialized:")
print(f"  Objects: {meta_params['num_objects']}")
print(f"  Feature dim: {meta_params['feature_dim']}")
print(f"  Embedding dim: {meta_params['embedding_dim']}")

# Training loop simulation
print(f"\nTraining:")
for epoch in range(n_epochs):
    # Simulate sampling batch of tasks
    batch_indices = np.random.choice(n_tasks, size=min(batch_size, n_tasks), replace=False)

    # Simulate computing meta-loss
    meta_loss = 0.5 * np.exp(-epoch / 10) + 0.1 * np.random.rand()  # Decreasing loss

    # Simulate gradient update (in real version: using JAX/Flax optimizers)
    meta_params['base_coverage'] += 0.001 * np.random.randn(*meta_params['base_coverage'].shape)

    # Print progress
    if epoch % 2 == 0 or epoch == n_epochs - 1:
        print(f"  Epoch {epoch:3d}/{n_epochs}: Meta-loss = {meta_loss:.4f}")

print("\n✓ Training simulation completed!")

# Simulate few-shot adaptation
print("\n" + "="*70)
print("SIMULATING FEW-SHOT ADAPTATION")
print("="*70)

test_task = tasks[-1]  # Use last task as "held-out"
print(f"\nAdapting to held-out task: {test_task['id']}")
print(f"  Using {n_shots} support examples")

# Simulate task encoding
task_embedding = np.random.randn(meta_params['embedding_dim'])
print(f"  Task embedding: shape {task_embedding.shape}")

# Simulate adaptation (adjusting coverage)
adaptation_delta = 0.1 * np.random.randn(*meta_params['base_coverage'].shape)
adapted_coverage = meta_params['base_coverage'] + adaptation_delta
print(f"  Adapted coverage: shape {adapted_coverage.shape}")

print("\n✓ Few-shot adaptation successful!")

# Simulate saving model
output_dir = Path("demo_trained_model")
output_dir.mkdir(exist_ok=True)

model_data = {
    'meta_params': {
        'num_objects': meta_params['num_objects'],
        'feature_dim': meta_params['feature_dim'],
        'embedding_dim': meta_params['embedding_dim'],
        # Note: numpy arrays can't be saved to JSON directly
        'base_coverage_shape': list(meta_params['base_coverage'].shape)
    },
    'training_config': {
        'n_epochs': n_epochs,
        'n_tasks': n_tasks,
        'batch_size': batch_size,
        'n_shots': n_shots
    },
    'timestamp': datetime.now().isoformat()
}

config_path = output_dir / "model_config.json"
with open(config_path, 'w') as f:
    json.dump(model_data, f, indent=2)

# Save numpy arrays separately
np.savez(output_dir / "model_weights.npz",
         base_coverage=meta_params['base_coverage'],
         task_embedding=task_embedding,
         adapted_coverage=adapted_coverage)

print(f"\n✓ Model saved to {output_dir}/")
print(f"  - model_config.json")
print(f"  - model_weights.npz")

# Summary
print("\n" + "="*70)
print("DEMO COMPLETE")
print("="*70)
print("\nThis demo showed:")
print("  ✓ Loading ARC-AGI training data")
print("  ✓ Meta-training loop simulation")
print("  ✓ Few-shot adaptation simulation")
print("  ✓ Model saving")
print("\nFor full training with JAX/Flax:")
print("  1. Install: pip install jax jaxlib flax optax onnx")
print("  2. Run: python train_meta_learner.py --epochs 50 --tasks 100")
print("\nSee TRAIN_README.md for details")
print("="*70)
