"""Debug overlap computation."""
import torch
from differentiable_gluing import compute_overlap_indices

# Simulate two sections with different sizes
base1 = torch.arange(100)  # 10×10 grid = 100 cells, indices [0,1,2,...,99]
base2 = torch.arange(81)   # 9×9 grid = 81 cells, indices [0,1,2,...,80]

overlap = compute_overlap_indices(base1, base2)

print(f"Base 1: length={len(base1)}, max={base1.max().item()}")
print(f"Base 2: length={len(base2)}, max={base2.max().item()}")
print(f"Overlap: length={len(overlap)}, values={overlap[:20]}... (first 20)")
print(f"Overlap max: {overlap.max().item() if len(overlap) > 0 else 'N/A'}")
print(f"\nExpected: overlap should be [0,1,2,...,80] (81 values)")
print(f"Actual: overlap has {len(overlap)} values, max {overlap.max().item()}")
