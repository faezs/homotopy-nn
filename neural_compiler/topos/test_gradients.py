"""
Test gradient flow in geometric morphism training.

Debugging why loss doesn't decrease.
"""

import torch
import torch.nn as nn
import numpy as np
from geometric_morphism_torch import Site, Sheaf, GeometricMorphism, InternalLogicLoss, ARCGrid

# Create simple grids
input_grid = ARCGrid.from_array(np.array([[1, 2], [3, 4]]))
output_grid = ARCGrid.from_array(np.array([[4, 3], [2, 1]]))

# Create sites
site_in = Site((2, 2))
site_out = Site((2, 2))

# Create sheaves
feature_dim = 16
sheaf_in = Sheaf.from_grid(input_grid, site_in, feature_dim)
sheaf_target = Sheaf.from_grid(output_grid, site_out, feature_dim)

# Create geometric morphism
f = GeometricMorphism(site_in, site_out, feature_dim)

# Check parameters
print("GeometricMorphism parameters:")
for name, param in f.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
print()

# Compute loss
loss_fn = InternalLogicLoss(sheaf_target)
loss = loss_fn.compute(f, sheaf_in)

print(f"Loss: {loss.item():.4f}")
print(f"Loss requires_grad: {loss.requires_grad}")
print(f"Loss grad_fn: {loss.grad_fn}")
print()

# Compute gradients
loss.backward()

# Check if gradients exist
print("Gradients after backward:")
for name, param in f.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name}: grad_norm={grad_norm:.6f}")
    else:
        print(f"  {name}: grad=None")
print()

# Try a manual update
print("Manual parameter update test:")
print(f"  Before: adjunction_matrix[0,0] = {f.adjunction_matrix[0,0].item():.4f}")

with torch.no_grad():
    f.adjunction_matrix[0, 0] -= 0.1 * f.adjunction_matrix.grad[0, 0]

print(f"  After: adjunction_matrix[0,0] = {f.adjunction_matrix[0,0].item():.4f}")

# Recompute loss with updated parameters
loss_updated = loss_fn.compute(f, sheaf_in)
print(f"  Loss after manual update: {loss_updated.item():.4f}")
print(f"  Change: {loss_updated.item() - loss.item():.6f}")
