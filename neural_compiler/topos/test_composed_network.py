"""Test the composed EquivariantSheafComposite network."""

import torch
from equivariant_sheaf_nn import (
    EquivariantSheafComposite,
    train_equivariant_sheaf_network
)
from arc_tensor_utils import arc_grid_to_tensor, tensor_to_arc_grid
from arc_loader import ARCGrid
import numpy as np

print("="*80)
print("Testing Composed Network: Equivariant + Sheaf + Sparse Attention")
print("="*80)

# Simple task: 5x5 grids, increment all colors by 1
def create_test_task(num_examples=3):
    pairs = []
    for _ in range(num_examples):
        # Random input
        input_cells = np.random.randint(0, 9, (5, 5))  # 0-8 so +1 stays in range
        output_cells = (input_cells + 1) % 10

        input_grid = ARCGrid(cells=input_cells, height=5, width=5)
        output_grid = ARCGrid(cells=output_cells, height=5, width=5)

        input_tensor = arc_grid_to_tensor(input_grid)
        output_tensor = arc_grid_to_tensor(output_grid)

        pairs.append((input_tensor, output_tensor))

    return pairs

print("\n1. Creating simple task (5x5 grids, increment colors)...")
training_pairs = create_test_task(num_examples=3)
print(f"   Created {len(training_pairs)} training pairs")

print("\n2. Building composed network...")
model = EquivariantSheafComposite(
    grid_size=(5, 5),
    in_channels=10,
    out_channels=10,
    feature_dim=32,
    stalk_dim=8,
    num_training_examples=len(training_pairs),
    sheaf_layers=2,
    k_neighbors=2,
    device='cpu'
)

print(f"   ✓ EquivariantHomotopyLearner: {sum(p.numel() for p in model.equivariant_learner.parameters())} params")
print(f"   ✓ SheafNeuralNetwork: {sum(p.numel() for p in model.sheaf_nn.parameters())} params")
print(f"   ✓ Bridge layers: {model.feature_to_stalk.weight.numel() + model.decode.weight.numel()} params")
print(f"   ✓ Total: {sum(p.numel() for p in model.parameters())} params")

print("\n3. Testing forward pass...")
test_input = training_pairs[0][0]
print(f"   Input shape: {test_input.shape}")

output = model(test_input, use_canonical=True)
print(f"   Output shape: {output.shape}")
print("   ✓ Forward pass works!")

print("\n4. Training composed network (50 epochs)...")
history = train_equivariant_sheaf_network(
    model=model,
    training_pairs=training_pairs,
    num_epochs=50,
    lr_equivariant=5e-3,
    lr_sheaf=1e-3,
    phase_transition_epoch=25,
    verbose=True
)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Initial loss: {history['total'][0]:.4f}")
print(f"Final loss: {history['total'][-1]:.4f}")
print(f"Reduction: {(1 - history['total'][-1]/history['total'][0])*100:.1f}%")

print(f"\nPhase 1 (fit): Recon {history['recon'][0]:.4f} → {history['recon'][24]:.4f}")
print(f"Phase 2 (collapse): Homotopy {history['homotopy'][25]:.4f} → {history['homotopy'][-1]:.4f}")

if history['homotopy'][-1] < history['homotopy'][25]:
    print("\n✓ Homotopy collapse working!")
else:
    print(f"\n✗ Homotopy collapse failed")

print("\n5. Testing on training examples...")
correct = 0
total = 0
for i, (x, y_true) in enumerate(training_pairs):
    with torch.no_grad():
        y_pred = model(x, use_canonical=True)

    pred_grid = tensor_to_arc_grid(y_pred)
    true_grid = tensor_to_arc_grid(y_true)

    matches = np.sum(pred_grid.cells == true_grid.cells)
    total_cells = pred_grid.height * pred_grid.width

    correct += matches
    total += total_cells

    print(f"   Example {i}: {matches}/{total_cells} = {matches/total_cells:.2%}")

accuracy = correct / total
print(f"\nOverall accuracy: {accuracy:.2%}")

if accuracy > 0.5:
    print("✓ Composed network learning!")
else:
    print("✗ Network not learning well")

print("\n" + "="*80)
print("Integration test complete!")
print("="*80)
