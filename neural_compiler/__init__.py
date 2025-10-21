"""
Neural Network Compiler from Agda to JAX

This package compiles type-checked neural architectures from Agda to optimized JAX code.

Pipeline:
    Agda (.agda) → IR (JSON) → PolyFunctor → JAX (.py)

Usage:
    from neural_compiler import compile_architecture

    model = compile_architecture("path/to/architecture.json")
    output = model(input_data, params)
"""

__version__ = "0.1.0"

from .parser import parse_ir, NeuralIR
from .polyfunctor import PolynomialFunctor, compile_to_polyfunctor
from .jax_backend import JAXBackend
from .compiler import compile_architecture

__all__ = [
    "parse_ir",
    "NeuralIR",
    "PolynomialFunctor",
    "compile_to_polyfunctor",
    "JAXBackend",
    "compile_architecture",
]
