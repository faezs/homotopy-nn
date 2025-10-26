"""
ARC Grid ↔ Tensor Conversion Utilities

Utilities for converting between ARC grids and PyTorch tensors for
equivariant homotopy learning.

Author: Claude Code
Date: 2025-10-25
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

from arc_loader import ARCGrid, ARCTask


################################################################################
# § 1: Grid ↔ Tensor Conversion
################################################################################

def arc_grid_to_tensor(
    grid: ARCGrid,
    num_channels: int = 10,
    device: str = 'cpu'
) -> torch.Tensor:
    """Convert ARC grid to tensor sheaf.

    Args:
        grid: ARCGrid (height, width) with values 0-9
        num_channels: Number of feature channels
        device: Device for tensor

    Returns:
        tensor: (1, num_channels, height, width)

    Encoding:
        - One-hot encode colors 0-9 in first 10 channels
        - Remaining channels initialized with small noise
    """
    h, w = grid.height, grid.width

    # Convert JAX array to numpy if needed
    if hasattr(grid.cells, '__array__'):
        cells = np.array(grid.cells)
    else:
        cells = grid.cells

    # Create tensor
    tensor = torch.zeros(1, num_channels, h, w, device=device)

    # One-hot encode colors
    for i in range(h):
        for j in range(w):
            color = int(cells[i, j])  # Convert to Python int
            if color < num_channels:
                tensor[0, color, i, j] = 1.0

    # Add small noise to remaining channels (if any)
    if num_channels > 10:
        tensor[0, 10:, :, :] = torch.randn(num_channels - 10, h, w, device=device) * 0.01

    return tensor


def tensor_to_arc_grid(
    tensor: torch.Tensor,
    threshold: float = 0.5
) -> ARCGrid:
    """Convert tensor back to ARC grid.

    Args:
        tensor: (1, num_channels, height, width)
        threshold: Threshold for color selection

    Returns:
        grid: ARCGrid with predicted colors
    """
    # Get shape
    _, num_channels, h, w = tensor.shape

    # Decode: argmax over first 10 channels
    color_channels = tensor[0, :10, :, :]  # (10, h, w)
    predicted_colors = torch.argmax(color_channels, dim=0)  # (h, w)

    # Convert to numpy
    cells = predicted_colors.cpu().numpy().astype(int)

    return ARCGrid.from_array(cells)


def pad_or_crop_tensor(
    tensor: torch.Tensor,
    target_height: int,
    target_width: int
) -> torch.Tensor:
    """Pad or crop tensor to target size.

    Args:
        tensor: (1, C, H, W)
        target_height: Desired height
        target_width: Desired width

    Returns:
        tensor: (1, C, target_height, target_width)
    """
    _, c, h, w = tensor.shape

    # Crop if too large
    if h > target_height:
        tensor = tensor[:, :, :target_height, :]
        h = target_height
    if w > target_width:
        tensor = tensor[:, :, :, :target_width]
        w = target_width

    # Pad if too small
    if h < target_height or w < target_width:
        pad_h = target_height - h
        pad_w = target_width - w
        tensor = nn.functional.pad(
            tensor,
            (0, pad_w, 0, pad_h),  # (left, right, top, bottom)
            mode='constant',
            value=0
        )

    return tensor


################################################################################
# § 2: Task → Training Pairs
################################################################################

def prepare_training_pairs(
    task: ARCTask,
    num_channels: int = 10,
    target_size: Optional[Tuple[int, int]] = None,
    device: str = 'cpu'
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Prepare training pairs from ARC task.

    Args:
        task: ARCTask with train_inputs and train_outputs
        num_channels: Number of feature channels
        target_size: (height, width) to resize all grids (None = use max)
        device: Device for tensors

    Returns:
        pairs: List of (input_tensor, output_tensor) pairs
    """
    # Determine target size
    if target_size is None:
        # Use maximum dimensions from training examples
        max_h = max(max(g.height for g in task.train_inputs),
                    max(g.height for g in task.train_outputs))
        max_w = max(max(g.width for g in task.train_inputs),
                    max(g.width for g in task.train_outputs))
        target_size = (max_h, max_w)

    target_h, target_w = target_size

    # Convert each training pair
    pairs = []
    for inp_grid, out_grid in zip(task.train_inputs, task.train_outputs):
        # Convert to tensors
        inp_tensor = arc_grid_to_tensor(inp_grid, num_channels, device)
        out_tensor = arc_grid_to_tensor(out_grid, num_channels, device)

        # Resize to target size
        inp_tensor = pad_or_crop_tensor(inp_tensor, target_h, target_w)
        out_tensor = pad_or_crop_tensor(out_tensor, target_h, target_w)

        pairs.append((inp_tensor, out_tensor))

    return pairs
