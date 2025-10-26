"""Debug section base_indices to understand the indexing issue."""
import torch
from pathlib import Path
from arc_loader import load_arc_dataset
from train_topos_arc import arc_grid_to_tensor, convert_arc_task_to_tensor
from topos_arc_solver import FewShotARCLearner

# Load one task
arc_data_root = Path.home() / "homotopy-nn" / "ARC-AGI" / "data"
all_tasks = load_arc_dataset(str(arc_data_root), split="training", limit=1)
task = list(all_tasks.values())[0]

# Convert to tensors
task_dict = convert_arc_task_to_tensor(task, device='cpu')

# Create model
model = FewShotARCLearner(
    grid_size=(30, 30),
    feature_dim=64,
    stalk_dim=8,
    num_patterns=16
)

# Get train pairs with different sizes
train_pairs = task_dict['train']

print("=" * 80)
print("DEBUGGING SECTION BASE_INDICES")
print("=" * 80)

for i, (inp, out) in enumerate(train_pairs):
    print(f"\nTrain pair {i}:")
    print(f"  Input shape: {inp.shape}")
    print(f"  Output shape: {out.shape}")

    # Extract section
    section = model.extract_section(inp.unsqueeze(0), out.unsqueeze(0))

    print(f"  Section base_indices: {section.base_indices[:10]}... (first 10)")
    print(f"  Section base_indices length: {len(section.base_indices)}")
    print(f"  Section base_indices max: {section.base_indices.max().item()}")
    print(f"  Section values shape: {section.values.shape}")
    print(f"  Match: base_indices length == values length? {len(section.base_indices) == len(section.values)}")
