"""
Graph Topos Solver using Existing CNN Sheaf Infrastructure

Uses the ARCCNNGeometricSolver with attention as natural transformations
to solve graph problems (Eulerian paths).

Key insight: Represent graph as grid-like structure where sheaf sections
live on vertices and attention implements graph convolution.

Author: Claude Code + Human
Date: October 22, 2025
"""

import sys
sys.path.insert(0, '/Users/faezs/homotopy-nn/neural_compiler')

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from topos.train_arc_geometric_production import ARCCNNGeometricSolver
from topos.arc_loader import ARCGrid
from topos.rigorous_topos_eval import generate_all_connected_graphs, create_balanced_dataset, GraphTopos


def graph_to_grid(graph: GraphTopos, grid_size: int = 8) -> ARCGrid:
    """Convert graph to grid representation for sheaf solver.

    Richer encoding:
    - Adjacency matrix with edge counts
    - Degree sequence encoded as colors (mod 10)
    - Degree parity in specific locations (key feature for Eulerian paths!)
    """
    n = graph.num_vertices
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)

    # Fill adjacency matrix
    for i in range(min(n, grid_size-1)):  # Leave last row for degrees
        for j in range(min(n, grid_size)):
            grid[i, j] = int(graph.adj_matrix[i, j].item())

    # Fill degree sequence in last row (key feature!)
    for i in range(min(n, grid_size)):
        deg = int(graph.degrees[i].item())
        grid[grid_size-1, i] = deg  # Actual degree value

    # Add degree parity in corners (explicit Eulerian path feature!)
    odd_count = (graph.degrees % 2 == 1).sum().item()
    grid[0, grid_size-1] = int(odd_count)  # Number of odd-degree vertices

    return ARCGrid.from_array(grid)


def grid_to_prediction(grid: ARCGrid) -> bool:
    """Extract Eulerian path prediction from output grid.

    Convention: Average center region, if >= 2.5 predict True
    """
    h, w = grid.height, grid.width

    # Average center 3x3 region
    center_sum = 0
    count = 0
    for i in range(h//2 - 1, h//2 + 2):
        for j in range(w//2 - 1, w//2 + 2):
            if 0 <= i < h and 0 <= j < w:
                val = grid.cells[i, j]
                if hasattr(val, '__array__'):
                    val = np.array(val).item()
                center_sum += int(val)
                count += 1

    avg = center_sum / count if count > 0 else 0
    return avg >= 2.5  # Threshold between 0 and 5


def train_graph_topos_solver(train_graphs: List[Tuple[GraphTopos, bool]],
                             test_graphs: List[Tuple[GraphTopos, bool]],
                             grid_size: int = 8,
                             feature_dim: int = 16,
                             epochs: int = 100,
                             lr: float = 0.01,
                             device='cpu'):
    """Train topos solver on graphs using existing CNN sheaf infrastructure."""

    print("="*70)
    print("GRAPH TOPOS SOLVER (Using CNN Sheaves + Attention)")
    print("="*70)
    print()

    # Create solver using existing topos infrastructure
    solver = ARCCNNGeometricSolver(
        grid_shape_in=(grid_size, grid_size),
        grid_shape_out=(grid_size, grid_size),
        feature_dim=feature_dim,
        num_colors=10,
        device=torch.device(device)
    )

    total_params = sum(p.numel() for p in solver.parameters())
    print(f"Topos solver parameters: {total_params:,}")
    print(f"  Using attention as natural transformations")
    print(f"  Sheaf sections over {grid_size}x{grid_size} site")
    print()

    optimizer = torch.optim.Adam(solver.parameters(), lr=lr)

    # Training loop
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        solver.train()
        epoch_loss = 0

        for graph, label in train_graphs:
            optimizer.zero_grad()

            # Convert graph to grid (input and target)
            input_grid = graph_to_grid(graph, grid_size)

            # Target: Multiple pixels encode label (stronger signal!)
            target_array = np.zeros((grid_size, grid_size), dtype=np.int32)
            # Fill center region with label
            val = 5 if label else 0  # Use distinct values
            for i in range(grid_size//2 - 1, grid_size//2 + 2):
                for j in range(grid_size//2 - 1, grid_size//2 + 2):
                    if 0 <= i < grid_size and 0 <= j < grid_size:
                        target_array[i, j] = val
            target_grid = ARCGrid.from_array(target_array)

            # Encode as sheaves
            input_sheaf = solver.encode_grid_to_sheaf(input_grid, solver.site_in)
            target_sheaf = solver.encode_grid_to_sheaf(target_grid, solver.site_out)

            # Apply geometric morphism (with attention!)
            predicted_sheaf = solver.geometric_morphism.pushforward(input_sheaf)

            # Separate losses (as designed)
            # 1. Sheaf space loss
            sheaf_space_loss = F.mse_loss(predicted_sheaf.sections, target_sheaf.sections)

            # 2. Output L2 loss
            predicted_grid = solver.decode_sheaf_to_grid(predicted_sheaf, grid_size, grid_size)
            pred_cells = np.array(predicted_grid.cells) if hasattr(predicted_grid.cells, '__array__') else predicted_grid.cells
            target_cells = np.array(target_grid.cells) if hasattr(target_grid.cells, '__array__') else target_grid.cells
            pred_pixels = torch.from_numpy(pred_cells).float().to(solver.device)
            target_pixels = torch.from_numpy(target_cells).float().to(solver.device)
            output_l2_loss = F.mse_loss(pred_pixels, target_pixels)

            # 3. Adjunction constraint (categorical law)
            adj_loss = solver.geometric_morphism.check_adjunction(input_sheaf, target_sheaf)

            # 4. Sheaf condition (gluing axiom)
            sheaf_loss = predicted_sheaf.total_sheaf_violation()

            # Combined loss (using proper weights)
            total_loss = (
                1.0 * output_l2_loss +
                0.5 * sheaf_space_loss +
                0.1 * adj_loss +
                0.01 * sheaf_loss
            )

            total_loss.backward()
            optimizer.step()

            epoch_loss += output_l2_loss.item()

        # Evaluate every 20 epochs
        if epoch % 20 == 0 or epoch == epochs - 1:
            solver.eval()
            with torch.no_grad():
                # Train accuracy
                train_correct = 0
                for graph, label in train_graphs:
                    input_grid = graph_to_grid(graph, grid_size)
                    output_grid = solver(input_grid, (grid_size, grid_size))
                    pred = grid_to_prediction(output_grid)
                    train_correct += (pred == label)

                # Test accuracy
                test_correct = 0
                for graph, label in test_graphs:
                    input_grid = graph_to_grid(graph, grid_size)
                    output_grid = solver(input_grid, (grid_size, grid_size))
                    pred = grid_to_prediction(output_grid)
                    test_correct += (pred == label)

                train_acc = train_correct / len(train_graphs)
                test_acc = test_correct / len(test_graphs)

                print(f"Epoch {epoch:3d}: Loss={epoch_loss/len(train_graphs):.4f}, "
                      f"Train={100*train_acc:.1f}%, Test={100*test_acc:.1f}%")

    print()
    return solver


if __name__ == "__main__":
    # Generate dataset
    print("Generating 5-vertex connected graphs...")
    all_graphs = generate_all_connected_graphs(num_vertices=5)
    print(f"Total: {len(all_graphs)} graphs")

    positive = sum(1 for _, label in all_graphs if label)
    print(f"  Eulerian: {positive} ({100*positive/len(all_graphs):.1f}%)")
    print(f"  Non-Eulerian: {len(all_graphs)-positive} ({100*(1-positive/len(all_graphs)):.1f}%)")
    print()

    # Create balanced split
    train_set, test_set = create_balanced_dataset(all_graphs, 100, 40, random_seed=42)
    print(f"Train set: {len(train_set)} (balanced)")
    print(f"Test set: {len(test_set)} (balanced)")
    print()

    # Train using topos infrastructure (higher LR for faster convergence)
    solver = train_graph_topos_solver(
        train_set, test_set,
        grid_size=8,
        feature_dim=16,
        epochs=200,
        lr=0.05,  # Higher LR!
        device='cpu'
    )

    print("="*70)
    print("KEY ACHIEVEMENTS:")
    print("  • Used existing topos infrastructure (ARCCNNGeometricSolver)")
    print("  • Attention as natural transformations between sheaves")
    print("  • Separate losses: L2, sheaf space, adjunction, gluing")
    print("  • Graphs represented as sheaves over grid sites")
    print("="*70)
