"""
Tensor Species Compiler

Direct einsum interpretation of tensor species from Agda.

Based on:
- Andrew Dudzik (2024): "Tensor Species" (Topos Institute)
- Ong & Veličković (2022): "Learning Algebraic Structure" (GNN aggregators)
- Bergomi & Vertechi (2022): "Parametric Spans"

Key insight: Neural networks ARE tensor species (polynomial functors), and
einsums are the universal language for representing them.
"""

from .species_compiler import (
    IndexVar,
    EinsumOp,
    LearnableMonoid,
    TensorSpecies,
    SpeciesCompiler,
)

__all__ = [
    'IndexVar',
    'EinsumOp',
    'LearnableMonoid',
    'TensorSpecies',
    'SpeciesCompiler',
]
