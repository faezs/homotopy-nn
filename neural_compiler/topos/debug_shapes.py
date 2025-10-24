"""
Quick debugging script to identify shape mismatch.
"""
import torch
from pathlib import Path
from arc_loader import load_arc_dataset
from train_topos_arc import arc_grid_to_tensor, convert_arc_task_to_tensor
from topos_arc_solver import FewShotARCLearner

# Load one task
arc_data_root = Path.home() / "homotopy-nn" / "ARC-AGI" / "data"
all_tasks = load_arc_dataset(str(arc_data_root), split="training", limit=1)
task = list(all_tasks.values())[0]

print("=" * 80)
print("DEBUGGING SHAPE MISMATCH")
print("=" * 80)

# Convert to tensors
task_dict = convert_arc_task_to_tensor(task, device='cpu')

# Get first train example
train_input, train_output = task_dict['train'][0]
print(f"\nOriginal train input shape: {train_input.shape}")
print(f"Original train output shape: {train_output.shape}")

# Create model with SAME config as training
model = FewShotARCLearner(
    grid_size=(30, 30),
    feature_dim=64,  # Same as TrainingConfig
    stalk_dim=8,
    num_patterns=16
)

print(f"\nModel initialized:")
print(f"  num_cells: {model.num_cells}")
print(f"  grid_size: {model.grid_size}")
print(f"  feature_dim: {model.feature_dim}")
print(f"  stalk_dim: {model.stalk_dim}")

# Don't pad - model handles it internally
# Just add batch dimension for test
train_input_batch = train_input.unsqueeze(0)
print(f"\nOriginal train input shape (no padding): {train_input.shape}")
print(f"Added batch dimension: {train_input_batch.shape}")

print(f"\nCalling extract_features with shape: {train_input_batch.shape}")
print("=" * 80)

try:
    features = model.extract_features(train_input_batch)
    print(f"\n✓ extract_features SUCCESS! Features shape: {features.shape}")

    # Now test the full forward pass
    print("\n" + "=" * 80)
    print("Testing FULL FORWARD PASS (variable-sized grids)")
    print("=" * 80)

    # NO PADDING - pass grids as-is (model handles it)
    # Train pairs: grids WITHOUT batch dimension
    train_pairs = [(train_input, train_output)]  # [3, 3, 10] and [9, 9, 10]
    # Test input: grid WITH batch dimension
    test_input_batch = train_input_batch  # [1, 3, 3, 10]

    print(f"Train pair shapes: input={train_pairs[0][0].shape}, output={train_pairs[0][1].shape}")
    print(f"Test input shape: {test_input_batch.shape}")

    prediction, gluing_result = model(train_pairs, test_input_batch, temperature=0.1)

    print(f"\n✓ FORWARD PASS SUCCESS!")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Compatibility score: {gluing_result.compatibility_score:.4f}")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
