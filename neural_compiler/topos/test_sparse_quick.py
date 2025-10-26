"""Quick test of sparse attention + group-aware homotopy distance."""

import torch
from equivariant_homotopy_learning import (
    EquivariantHomotopyLearner,
    train_equivariant_homotopy
)
from stacks_of_dnns import DihedralGroup
from arc_tensor_utils import arc_grid_to_tensor, tensor_to_arc_grid
from arc_loader import load_arc_dataset, ARCGrid

# Simple synthetic task
def create_simple_task():
    """Create a simple 5x5 task: all cells increment by 1."""
    train_pairs = []

    for i in range(3):
        # Input: random colors
        input_grid = ARCGrid(cells=torch.randint(0, 10, (5, 5)).numpy(), height=5, width=5)
        # Output: increment all by 1 (mod 10)
        output_cells = (input_grid.cells + 1) % 10
        output_grid = ARCGrid(cells=output_cells, height=5, width=5)

        # Convert to tensors
        input_tensor = arc_grid_to_tensor(input_grid)
        output_tensor = arc_grid_to_tensor(output_grid)

        train_pairs.append((input_tensor, output_tensor))

    return train_pairs

print("Creating simple task...")
training_pairs = create_simple_task()

print("Creating learner with sparse attention...")
D4 = DihedralGroup(n=4)
learner = EquivariantHomotopyLearner(
    group=D4,
    in_channels=10,
    out_channels=10,
    feature_dim=32,
    kernel_size=3,
    num_training_examples=len(training_pairs),
    device='cpu'
)

print(f"Training {len(training_pairs)} examples for 50 epochs...")
print(f"Sparse attention: k={learner.sparse_attention.k} neighbors")

history = train_equivariant_homotopy(
    learner=learner,
    training_pairs=training_pairs,
    num_epochs=50,
    lr_individual=1e-3,
    lr_canonical=5e-4,
    phase_transition_epoch=25,
    verbose=True,
    device='cpu'
)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Initial homotopy distance: {history['homotopy'][0]:.4f}")
print(f"Final homotopy distance: {history['homotopy'][-1]:.4f}")
print(f"Initial canonical loss: {history['canonical_recon'][0]:.4f}")
print(f"Final canonical loss: {history['canonical_recon'][-1]:.4f}")

if history['homotopy'][-1] < history['homotopy'][24]:
    print("\n✓ Homotopy collapse working! Distance decreased in Phase 2")
else:
    print(f"\n✗ Homotopy collapse failed. Phase 1: {history['homotopy'][24]:.4f}, Phase 2: {history['homotopy'][-1]:.4f}")

if history['canonical_recon'][-1] < 1.0:
    print("✓ Canonical morphism learning!")
else:
    print(f"✗ Canonical morphism not learning well (loss={history['canonical_recon'][-1]:.4f})")
