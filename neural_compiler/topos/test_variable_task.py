"""Test training on a task with variable-sized training pairs."""
import torch
import traceback
from pathlib import Path
from arc_loader import load_arc_dataset
from train_topos_arc import convert_arc_task_to_tensor
from topos_arc_solver import FewShotARCLearner

# Load task with variable sizes
arc_data_root = Path.home() / "homotopy-nn" / "ARC-AGI" / "data"
all_tasks = load_arc_dataset(str(arc_data_root), split="training", limit=20)

# Get task 00d62c1b (has variable train sizes: (6,6), (10,10), (10,10), (10,10), (20,20))
task = all_tasks['00d62c1b']

# Convert to tensors
task_dict = convert_arc_task_to_tensor(task, device='cpu')

# Create model
model = FewShotARCLearner(
    grid_size=(30, 30),
    feature_dim=64,
    stalk_dim=8,
    num_patterns=16
)

print("=" * 80)
print("TESTING VARIABLE-SIZE TASK")
print("=" * 80)

# Print shapes
train_pairs = task_dict['train']
test_input, test_output = task_dict['test']

print(f"\nNumber of training pairs: {len(train_pairs)}")
for i, (inp, out) in enumerate(train_pairs):
    print(f"  Pair {i}: input {inp.shape}, output {out.shape}")

print(f"\nTest:")
print(f"  Input: {test_input.shape}")
print(f"  Output: {test_output.shape}")

# Try forward pass
print("\nRunning forward pass...")
try:
    prediction, gluing_result = model(
        train_pairs,
        test_input.unsqueeze(0),
        temperature=0.1
    )
    print(f"✓ SUCCESS!")
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Compatibility score: {gluing_result.compatibility_score.item():.6f}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
