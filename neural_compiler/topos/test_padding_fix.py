"""
Quick test to verify zero-padding fix for ARC solver
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from arc_solver import ARCGrid, ARCTask, ARCReasoningNetwork, create_grid_site

# Create test grids of different sizes (like actual ARC tasks)
print("Creating test grids with different sizes...")
grid_3x3 = ARCGrid.from_array(np.random.randint(0, 10, (3, 3)))
grid_9x9 = ARCGrid.from_array(np.random.randint(0, 10, (9, 9)))
grid_6x3 = ARCGrid.from_array(np.random.randint(0, 10, (6, 3)))

print(f"✓ Grid 1: {grid_3x3.height}×{grid_3x3.width} = {grid_3x3.height * grid_3x3.width} cells")
print(f"✓ Grid 2: {grid_9x9.height}×{grid_9x9.width} = {grid_9x9.height * grid_9x9.width} cells")
print(f"✓ Grid 3: {grid_6x3.height}×{grid_6x3.width} = {grid_6x3.height * grid_6x3.width} cells")
print()

# Create a mixed-size task (3×3 → 9×9)
print("Creating mixed-size task (3×3 → 9×9)...")
task = ARCTask(
    train_inputs=[grid_3x3, grid_3x3],
    train_outputs=[grid_9x9, grid_9x9],
    test_inputs=[grid_3x3],
    test_outputs=[grid_9x9]
)
print("✓ Task created")
print()

# Create network and site
print("Initializing network and site...")
network = ARCReasoningNetwork(hidden_dim=128, num_colors=10)
key = random.PRNGKey(42)
site = create_grid_site(height=30, width=30, coverage_type="local", key=key)
print("✓ Network and site ready")
print()

# Test encoding with different grid sizes
print("Testing encoding with zero-padding...")
k1, k2, k3 = random.split(key, 3)

# Initialize network params
params = network.init(
    k1,
    grid_3x3,
    [(grid_3x3, grid_9x9)],
    site
)['params']
print("✓ Network parameters initialized")

# Test forward pass (this should work with padding)
print()
print("Testing forward pass with mixed-size grids...")
try:
    prediction = network.apply(
        {'params': params},
        grid_3x3,  # 3×3 input
        [(grid_3x3, grid_9x9)],  # 3×3 → 9×9 examples
        site
    )
    print(f"✅ SUCCESS! Forward pass completed")
    print(f"   Input: {grid_3x3.height}×{grid_3x3.width} = {grid_3x3.height * grid_3x3.width} cells")
    print(f"   Output: {prediction.height}×{prediction.width} = {prediction.height * prediction.width} cells")
    print()
    print("✅ Zero-padding fix is working!")
except Exception as e:
    print(f"❌ FAILED: {e}")
    print()
    print("Zero-padding fix needs more work")
