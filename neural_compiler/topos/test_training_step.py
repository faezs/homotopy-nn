"""Test a full training step including backward pass."""
import torch
import traceback
from pathlib import Path
from arc_loader import load_arc_dataset, split_arc_dataset
from train_topos_arc import convert_arc_task_to_tensor, ToposARCTrainer, TrainingConfig

# Load dataset
arc_data_root = Path.home() / "homotopy-nn" / "ARC-AGI" / "data"
all_tasks = load_arc_dataset(str(arc_data_root), split="training", limit=20)

# Split
train_ids, val_ids, test_ids = split_arc_dataset(all_tasks)
train_tasks = {k: all_tasks[k] for k in train_ids}

# Convert to tensors
print("Converting tasks to tensors...")
train_tensor_tasks = []
for task_id, task in train_tasks.items():
    try:
        task_dict = convert_arc_task_to_tensor(task, device='cpu')
        train_tensor_tasks.append(task_dict)
        print(f"✓ {task_id}")
    except Exception as e:
        print(f"✗ {task_id}: {e}")

print(f"\nConverted {len(train_tensor_tasks)} tasks")

# Create trainer
config = TrainingConfig(num_epochs=1, learning_rate=0.001)
trainer = ToposARCTrainer(config, train_tensor_tasks)

print("\n" + "=" * 80)
print("TESTING TRAINING STEP")
print("=" * 80)

# Try training on first task
print(f"\nTraining on task 0...")
try:
    trainer.train_step(train_tensor_tasks[0])
    print(f"✓ SUCCESS!")
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
