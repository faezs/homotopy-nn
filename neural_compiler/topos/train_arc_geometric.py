"""
Train Geometric Morphisms on Real ARC Tasks

This demonstrates the complete pipeline:
1. Encode ARC grids as sheaves
2. Learn geometric morphism f: E_in → E_out
3. Decode output sheaf back to grid
4. Measure actual task accuracy

This is LEARNING FUNCTORS BETWEEN TOPOI to solve ARC!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple

from geometric_morphism_torch import Site, Sheaf, GeometricMorphism, SheafReward, InternalLogicLoss
from arc_loader import ARCGrid, ARCTask, load_arc_dataset


class ARCGeometricSolver(nn.Module):
    """Complete solver: Grid → Sheaf → Geometric Morphism → Sheaf → Grid."""

    def __init__(self, grid_shape_in: Tuple[int, int], grid_shape_out: Tuple[int, int],
                 feature_dim: int = 32, num_colors: int = 10):
        super().__init__()

        # Sites
        self.site_in = Site(grid_shape_in, connectivity="4")
        self.site_out = Site(grid_shape_out, connectivity="4")

        # Geometric morphism
        self.geometric_morphism = GeometricMorphism(
            self.site_in, self.site_out, feature_dim
        )

        # Encoder: Grid → Sheaf
        self.encoder = nn.Sequential(
            nn.Linear(num_colors, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Decoder: Sheaf → Grid
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_colors)
        )

        self.feature_dim = feature_dim
        self.num_colors = num_colors

    def encode_grid_to_sheaf(self, grid: ARCGrid) -> Sheaf:
        """Encode ARC grid as sheaf (differentiable)."""
        # One-hot encode colors
        colors = torch.from_numpy(np.array(grid.cells).flatten()).long()
        one_hot = F.one_hot(colors, num_classes=self.num_colors).float()

        # Encode to feature space
        features = self.encoder(one_hot)

        # Create sheaf
        sheaf = Sheaf(self.site_in if grid.height * grid.width == self.site_in.num_objects
                     else self.site_out, self.feature_dim, self.num_colors)
        sheaf.sections = nn.Parameter(features)

        return sheaf

    def decode_sheaf_to_grid(self, sheaf: Sheaf, height: int, width: int) -> ARCGrid:
        """Decode sheaf back to ARC grid."""
        # Decode to color logits
        logits = self.decoder(sheaf.sections)

        # Argmax to colors
        colors = torch.argmax(logits, dim=-1).detach().cpu().numpy()

        # Reshape to grid
        grid_cells = colors[:height * width].reshape(height, width).astype(np.int32)

        return ARCGrid(height=height, width=width, cells=grid_cells)

    def forward(self, input_grid: ARCGrid, output_shape: Tuple[int, int]) -> ARCGrid:
        """Complete forward pass: input grid → output grid via geometric morphism."""
        # Encode
        input_sheaf = self.encode_grid_to_sheaf(input_grid)

        # Apply geometric morphism
        output_sheaf = self.geometric_morphism.pushforward(input_sheaf)

        # Decode
        output_grid = self.decode_sheaf_to_grid(output_sheaf, *output_shape)

        return output_grid


def train_on_arc_task(task: ARCTask, epochs: int = 100, verbose: bool = True):
    """Train geometric morphism on single ARC task."""

    if verbose:
        print(f"Training on ARC task...")
        print(f"  Training examples: {len(task.train_inputs)}")
        print(f"  Input size: {task.train_inputs[0].height}×{task.train_inputs[0].width}")
        print(f"  Output size: {task.train_outputs[0].height}×{task.train_outputs[0].width}")
        print()

    # Determine grid shapes
    input_shape = (task.train_inputs[0].height, task.train_inputs[0].width)
    output_shape = (task.train_outputs[0].height, task.train_outputs[0].width)

    # Create solver
    solver = ARCGeometricSolver(input_shape, output_shape, feature_dim=32)

    # Optimizer
    optimizer = optim.Adam(solver.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for inp_grid, out_grid in zip(task.train_inputs, task.train_outputs):
            optimizer.zero_grad()

            # Encode input as sheaf
            input_sheaf = solver.encode_grid_to_sheaf(inp_grid)

            # Target sheaf
            target_sheaf = solver.encode_grid_to_sheaf(out_grid)

            # Apply geometric morphism
            predicted_sheaf = solver.geometric_morphism.pushforward(input_sheaf)

            # Loss in sheaf space (differentiable!)
            loss = F.mse_loss(predicted_sheaf.sections, target_sheaf.sections)

            # Add adjunction constraint
            adj_loss = solver.geometric_morphism.check_adjunction(input_sheaf, target_sheaf)

            total_loss = loss + 0.1 * adj_loss

            total_loss.backward()
            optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {total_loss.item():.4f}")

    if verbose:
        print()
        print("Training complete!")
        print()

    # Evaluate
    test_input = task.test_inputs[0]
    test_output = task.test_outputs[0]

    prediction = solver(test_input, output_shape)

    # Accuracy
    if prediction.height == test_output.height and prediction.width == test_output.width:
        correct = np.sum(prediction.cells == test_output.cells)
        total = test_output.height * test_output.width
        accuracy = correct / total
    else:
        accuracy = 0.0

    if verbose:
        print("Results:")
        print(f"  Prediction shape: {prediction.height}×{prediction.width}")
        print(f"  Target shape: {test_output.height}×{test_output.width}")
        print(f"  Accuracy: {accuracy:.1%}")

    return solver, accuracy


if __name__ == "__main__":
    print("=" * 70)
    print("Training Geometric Morphisms on ARC Tasks")
    print("=" * 70)
    print()

    # Create simple test task
    print("Creating test task (2×2 identity)...")
    task = ARCTask(
        train_inputs=[
            ARCGrid.from_array(np.array([[1, 2], [3, 4]])),
            ARCGrid.from_array(np.array([[5, 6], [7, 8]]))
        ],
        train_outputs=[
            ARCGrid.from_array(np.array([[1, 2], [3, 4]])),  # Identity
            ARCGrid.from_array(np.array([[5, 6], [7, 8]]))
        ],
        test_inputs=[ARCGrid.from_array(np.array([[0, 1], [2, 3]]))],
        test_outputs=[ARCGrid.from_array(np.array([[0, 1], [2, 3]]))]
    )
    print("✓ Task created")
    print()

    # Train!
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()

    solver, accuracy = train_on_arc_task(task, epochs=50, verbose=True)

    print()
    print("=" * 70)
    print(f"✓ Final Accuracy: {accuracy:.1%}")
    print("=" * 70)
    print()
    print("This is geometric morphism learning in action!")
