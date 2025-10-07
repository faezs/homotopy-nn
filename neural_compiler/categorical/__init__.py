"""
Categorical Compiler for Neural Networks

This module provides categorical interpreters that preserve compositional
structure from topos theory.

Key insight: A sheaf F: C^op → Set is already computational:
- F₀(c) = tensors at vertices
- F₁(f) = operations
- F-∘ = composition (automatic!)
- Sheaf condition = merge constraints

Instead of compiling a flat graph, we interpret functors categorically.
"""

from .functor_compiler import FunctorCompiler
from .sheaf_compiler import SheafCompiler
from .fibration_compiler import FibrationCompiler

__all__ = [
    'FunctorCompiler',
    'SheafCompiler',
    'FibrationCompiler',
]
