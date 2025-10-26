"""
Fractal Weight Initialization for Neural Networks

Uses space-filling curves (Hilbert, Peano) to initialize weights
with fractal distributions, matching the tree/forest structure of
oriented graph neural architectures.

Theoretical motivation:
- Oriented graphs are forests (classical + acyclic)
- Trees have hierarchical self-similar structure
- Fractals provide universal dense embeddings
- Space-filling curves can approximate any distribution

References:
- Belfiore & Bennequin (2022) - Topos structure of DNNs
- Proposition 1.1 - CX is a poset (tree-like ordering)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Literal

try:
    from hilbertcurve.hilbertcurve import HilbertCurve
    HILBERT_AVAILABLE = True
except ImportError:
    HILBERT_AVAILABLE = False
    print("Warning: hilbertcurve not available. Install with: pip install hilbertcurve")


def hilbert_sequence(n_points: int, p: int = 8, n: int = 2) -> np.ndarray:
    """
    Generate sequence of points along Hilbert curve.

    Args:
        n_points: Number of points to sample
        p: Hilbert curve order (higher = more detail)
        n: Dimension (2 for 2D, 3 for 3D)

    Returns:
        Array of shape (n_points, n) with coordinates in [0, 1]^n
    """
    if not HILBERT_AVAILABLE:
        # Fallback to uniform random
        return np.random.rand(n_points, n)

    hilbert = HilbertCurve(p, n)
    max_h = 2**(p * n) - 1

    # Sample uniformly along curve
    distances = np.linspace(0, max_h, n_points, dtype=int)
    points = np.array([hilbert.point_from_distance(d) for d in distances])

    # Normalize to [0, 1]
    points = points / (2**p - 1)

    return points


def dragon_curve_sequence(n_points: int, iterations: int = 10) -> np.ndarray:
    """
    Generate sequence of points along dragon curve (2D fractal).

    Args:
        n_points: Number of points to sample
        iterations: Number of dragon curve iterations

    Returns:
        Array of shape (n_points, 2) with normalized coordinates
    """
    # Dragon curve via iterated function system
    # Start with line segment [0,0] -> [1,0]
    points = np.array([[0.0, 0.0], [1.0, 0.0]])

    for _ in range(iterations):
        # Rotate 90° around endpoint and append
        last = points[-1]
        rotated = points[:-1] - last
        rotated = np.column_stack([rotated[:, 1], -rotated[:, 0]])  # 90° rotation
        rotated = rotated[::-1] + last
        points = np.vstack([points, rotated])

    # Sample n_points uniformly along curve
    indices = np.linspace(0, len(points) - 1, n_points, dtype=int)
    sampled = points[indices]

    # Normalize to [0, 1]
    sampled = (sampled - sampled.min(axis=0)) / (sampled.max(axis=0) - sampled.min(axis=0) + 1e-8)

    return sampled


def cantor_weights(n: int, levels: int = 5, gap_ratio: float = 0.33) -> np.ndarray:
    """
    Generate weights based on Cantor set (1D fractal).

    Hierarchical structure with gaps at multiple scales.

    Args:
        n: Number of weights
        levels: Number of Cantor iterations
        gap_ratio: Size of middle gap (0.33 = remove middle third)

    Returns:
        Array of shape (n,) with fractal distribution
    """
    # Start with full interval [0, 1]
    intervals = [(0.0, 1.0)]

    for _ in range(levels):
        new_intervals = []
        for left, right in intervals:
            width = right - left
            gap_size = width * gap_ratio
            left_end = left + (width - gap_size) / 2
            right_start = right - (width - gap_size) / 2
            new_intervals.extend([(left, left_end), (right_start, right)])
        intervals = new_intervals

    # Sample uniformly from surviving intervals
    points = []
    for left, right in intervals:
        n_in_interval = max(1, int(n * (right - left)))
        points.extend(np.linspace(left, right, n_in_interval))

    # Resample to exact n points
    if len(points) < n:
        points.extend(np.random.choice(points, n - len(points)))
    points = np.array(points[:n])

    return points


def fractal_init_weights(
    shape: Tuple[int, ...],
    method: Literal['hilbert', 'dragon', 'cantor'] = 'hilbert',
    scale: float = 0.01,
    **kwargs
) -> torch.Tensor:
    """
    Initialize weight tensor using fractal distribution.

    Args:
        shape: Shape of weight tensor (e.g., (out_features, in_features))
        method: Fractal type ('hilbert', 'dragon', 'cantor')
        scale: Scaling factor for weights
        **kwargs: Additional arguments for fractal generation

    Returns:
        Initialized weight tensor
    """
    n_weights = np.prod(shape)

    if method == 'hilbert':
        # Use 2D Hilbert curve for all weights
        p = kwargs.get('p', 8)
        points = hilbert_sequence(n_weights, p=p, n=2)
        # Map [0,1]^2 to Gaussian-like via Box-Muller
        u1, u2 = points[:, 0], points[:, 1]
        # Avoid log(0)
        u1 = np.clip(u1, 1e-8, 1 - 1e-8)
        weights = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

    elif method == 'dragon':
        iterations = kwargs.get('iterations', 10)
        points = dragon_curve_sequence(n_weights, iterations=iterations)
        # Map to standard normal via inverse CDF approximation
        u1, u2 = points[:, 0], points[:, 1]
        u1 = np.clip(u1, 1e-8, 1 - 1e-8)
        weights = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

    elif method == 'cantor':
        levels = kwargs.get('levels', 5)
        weights = cantor_weights(n_weights, levels=levels)
        # Map [0,1] to standard normal via inverse CDF
        from scipy.stats import norm
        weights = norm.ppf(np.clip(weights, 1e-8, 1 - 1e-8))

    else:
        raise ValueError(f"Unknown method: {method}")

    # Reshape and scale
    weights = weights.reshape(shape) * scale

    return torch.tensor(weights, dtype=torch.float32)


def apply_fractal_init(
    model: nn.Module,
    method: Literal['hilbert', 'dragon', 'cantor'] = 'hilbert',
    scale: float = 0.01,
    verbose: bool = True
) -> None:
    """
    Apply fractal initialization to all parameters in a model.

    Args:
        model: PyTorch model
        method: Fractal type
        scale: Scaling factor
        verbose: Print initialization info
    """
    total_params = 0
    initialized_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()

            # Initialize with fractal distribution
            with torch.no_grad():
                fractal_weights = fractal_init_weights(
                    param.shape,
                    method=method,
                    scale=scale
                )
                param.copy_(fractal_weights)

            initialized_params += param.numel()

            if verbose:
                print(f"Initialized {name}: {param.shape} with {method} fractal")

    if verbose:
        print(f"\nTotal parameters initialized: {initialized_params}/{total_params}")
        print(f"Fractal method: {method}")
        print(f"Scale: {scale}")


# Example usage
if __name__ == "__main__":
    # Test fractal generation
    print("Testing fractal weight initialization...\n")

    # Test different methods
    shape = (10, 20)

    for method in ['hilbert', 'dragon', 'cantor']:
        try:
            weights = fractal_init_weights(shape, method=method, scale=0.01)
            print(f"{method.capitalize()} initialization:")
            print(f"  Shape: {weights.shape}")
            print(f"  Mean: {weights.mean():.4f}")
            print(f"  Std: {weights.std():.4f}")
            print(f"  Min: {weights.min():.4f}, Max: {weights.max():.4f}")
            print()
        except Exception as e:
            print(f"{method.capitalize()} failed: {e}\n")

    # Test on simple model
    print("Testing on simple MLP...")
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    apply_fractal_init(model, method='hilbert', scale=0.02)
