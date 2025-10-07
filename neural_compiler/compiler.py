"""
Main compiler interface: Agda → JAX

End-to-end compilation pipeline.
"""

from pathlib import Path
from typing import Callable, Dict, Any
import jax.numpy as jnp

from .parser import parse_ir, NeuralIR
from .polyfunctor import compile_to_polyfunctor, PolynomialFunctor
from .jax_backend import JAXBackend, initialize_params


class CompiledModel:
    """
    A compiled neural network ready for training/inference.

    Attributes:
        forward: JIT-compiled forward function
        params: Initialized parameters
        ir: Original IR
        poly: Polynomial functor representation
    """

    def __init__(self, forward_fn: Callable, params: Dict, ir: NeuralIR, poly: PolynomialFunctor):
        self.forward = forward_fn
        self.params = params
        self.ir = ir
        self.poly = poly

    def __call__(self, x):
        """Execute forward pass."""
        return self.forward(x, self.params)

    def summary(self):
        """Print model summary."""
        print(f"=== {self.ir.name} ===")
        print(f"Vertices: {len(self.ir.vertices)}")
        print(f"Edges: {len(self.ir.edges)}")
        print(f"Properties: {[p.name for p in self.ir.properties]}")
        print(f"Resources:")
        print(f"  Max FLOPs: {self.ir.resources.max_flops}")
        print(f"  Max Memory: {self.ir.resources.max_memory}")
        print(f"  Sparsity: {self.ir.resources.sparsity}%")

        # Parameter count
        total_params = sum(
            p.size for params_dict in self.params.values()
            for p in params_dict.values()
        )
        print(f"Parameters: {total_params:,}")


def compile_architecture(
    json_path: str,
    rng_seed: int = 0,
    verbose: bool = True
) -> CompiledModel:
    """
    Compile neural architecture from Agda-exported JSON to JAX.

    Pipeline:
        1. Parse JSON → IR
        2. IR → Polynomial Functor
        3. Polynomial Functor → String Diagram
        4. String Diagram → JAX Function
        5. Initialize parameters
        6. JIT compile

    Args:
        json_path: Path to Agda-exported JSON file
        rng_seed: Random seed for parameter initialization
        verbose: Print compilation progress

    Returns:
        CompiledModel ready for training/inference

    Example:
        >>> model = compile_architecture("resnet.json")
        >>> output = model(input_data)
    """
    import jax

    if verbose:
        print(f"🔨 Compiling {json_path}...")

    # Step 1: Parse IR
    if verbose:
        print("  [1/5] Parsing IR...")
    ir = parse_ir(json_path)

    # Step 2: Convert to polynomial functor
    if verbose:
        print("  [2/5] Converting to polynomial functor...")
    poly = compile_to_polyfunctor(ir)

    # Step 3: Compile to JAX
    if verbose:
        print("  [3/5] Compiling to JAX...")
    backend = JAXBackend()
    forward_fn = backend.compile(poly)

    # Step 4: Initialize parameters
    if verbose:
        print("  [4/5] Initializing parameters...")
    rng = jax.random.PRNGKey(rng_seed)
    params = initialize_params(poly, rng)

    # Step 5: Create model
    if verbose:
        print("  [5/5] Creating model...")
    model = CompiledModel(forward_fn, params, ir, poly)

    if verbose:
        print("✅ Compilation complete!")
        model.summary()

    return model


def compile_from_agda(
    agda_module: str,
    output_json: str = "architecture.json",
    compile_to_jax: bool = True
) -> CompiledModel:
    """
    Compile directly from Agda module (requires Agda installed).

    This would call:
        agda --compile-to-json <module> -o <output_json>

    Then compile the JSON to JAX.

    Args:
        agda_module: Path to .agda file
        output_json: Where to write IR JSON
        compile_to_jax: If True, continue to JAX compilation

    Returns:
        CompiledModel
    """
    import subprocess

    # Step 1: Extract IR from Agda (via Agda reflection)
    # This would require implementing the extraction in Agda
    # For now, assume JSON already exists

    print("⚠️  Direct Agda compilation not yet implemented")
    print("    Please export to JSON manually using Neural.Compile.Serialize")
    print("    Then use compile_architecture() on the JSON file")

    if Path(output_json).exists() and compile_to_jax:
        return compile_architecture(output_json)
    else:
        raise NotImplementedError("Direct Agda compilation requires Agda FFI")


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_model(model: CompiledModel, input_shape: tuple, num_runs: int = 100):
    """
    Benchmark model performance.

    Args:
        model: Compiled model
        input_shape: Shape of input tensor
        num_runs: Number of runs for averaging

    Returns:
        Dict with timing statistics
    """
    import jax
    import time

    # Create dummy input
    x = jnp.ones(input_shape)

    # Warmup (JIT compilation)
    _ = model(x)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model(x)
        jax.block_until_ready(_)  # Wait for computation
        times.append(time.time() - start)

    import numpy as np
    times = np.array(times)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "throughput_samples_per_sec": 1.0 / np.mean(times),
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Compile and benchmark MLP
    # model = compile_architecture("examples/mlp.json")
    #
    # # Benchmark
    # stats = benchmark_model(model, (1, 784))
    # print(f"\nBenchmark:")
    # print(f"  Mean: {stats['mean_ms']:.2f} ms")
    # print(f"  Throughput: {stats['throughput_samples_per_sec']:.0f} samples/sec")
    pass
