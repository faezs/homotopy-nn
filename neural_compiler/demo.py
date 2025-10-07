"""
Demo: End-to-end compilation from Agda IR to JAX

This demonstrates the full compilation pipeline with a working example.
"""

import json
import jax
import jax.numpy as jnp
from pathlib import Path

from neural_compiler import compile_architecture, benchmark_model


def create_mlp_json():
    """
    Create example MLP JSON (simulating Agda export).

    This would normally come from:
        Neural.Compile.Serialize.export-to-file "mlp.json" mlp-ir
    """
    mlp = {
        "name": "SimpleMLP",
        "vertices": [
            {
                "id": 0,
                "op": {"type": "linear", "in_dim": 784, "out_dim": 256},
                "input_shapes": [{"type": "vec", "dim": 784}],
                "output_shape": {"type": "vec", "dim": 256}
            },
            {
                "id": 1,
                "op": {"type": "activation", "activation": "relu"},
                "input_shapes": [{"type": "vec", "dim": 256}],
                "output_shape": {"type": "vec", "dim": 256}
            },
            {
                "id": 2,
                "op": {"type": "linear", "in_dim": 256, "out_dim": 10},
                "input_shapes": [{"type": "vec", "dim": 256}],
                "output_shape": {"type": "vec", "dim": 10}
            }
        ],
        "edges": [
            {"source": 0, "target": 1, "shape": {"type": "vec", "dim": 256}},
            {"source": 1, "target": 2, "shape": {"type": "vec", "dim": 256}}
        ],
        "inputs": [0],
        "outputs": [2],
        "properties": ["shape-correct", "conserves-mass"],
        "resources": {
            "max_flops": 1000000,
            "max_memory": 1000000,
            "max_latency": 1000,
            "sparsity": 0
        }
    }

    # Write to file
    with open("mlp.json", "w") as f:
        json.dump(mlp, f, indent=2)

    return "mlp.json"


def create_resnet_block_json():
    """
    Create ResNet block JSON (demonstrates fork/residual).
    """
    resnet = {
        "name": "ResNetBlock",
        "vertices": [
            {
                "id": 0,
                "op": {"type": "conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 3},
                "input_shapes": [{"type": "tensor", "dims": [64, 32, 32]}],
                "output_shape": {"type": "tensor", "dims": [64, 32, 32]}
            },
            {
                "id": 1,
                "op": {"type": "batch_norm", "features": 64},
                "input_shapes": [{"type": "tensor", "dims": [64, 32, 32]}],
                "output_shape": {"type": "tensor", "dims": [64, 32, 32]}
            },
            {
                "id": 2,
                "op": {"type": "activation", "activation": "relu"},
                "input_shapes": [{"type": "tensor", "dims": [64, 32, 32]}],
                "output_shape": {"type": "tensor", "dims": [64, 32, 32]}
            },
            {
                "id": 3,
                "op": {"type": "conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 3},
                "input_shapes": [{"type": "tensor", "dims": [64, 32, 32]}],
                "output_shape": {"type": "tensor", "dims": [64, 32, 32]}
            },
            {
                "id": 4,
                "op": {"type": "residual"},
                "input_shapes": [{"type": "tensor", "dims": [64, 32, 32]}],
                "output_shape": {"type": "tensor", "dims": [64, 32, 32]}
            },
            {
                "id": 5,
                "op": {"type": "fork", "arity": 2},
                "input_shapes": [
                    {"type": "tensor", "dims": [64, 32, 32]},
                    {"type": "tensor", "dims": [64, 32, 32]}
                ],
                "output_shape": {"type": "tensor", "dims": [64, 32, 32]}
            }
        ],
        "edges": [
            {"source": 0, "target": 1, "shape": {"type": "tensor", "dims": [64, 32, 32]}},
            {"source": 1, "target": 2, "shape": {"type": "tensor", "dims": [64, 32, 32]}},
            {"source": 2, "target": 3, "shape": {"type": "tensor", "dims": [64, 32, 32]}},
            {"source": 3, "target": 5, "shape": {"type": "tensor", "dims": [64, 32, 32]}},
            {"source": 0, "target": 4, "shape": {"type": "tensor", "dims": [64, 32, 32]}},
            {"source": 4, "target": 5, "shape": {"type": "tensor", "dims": [64, 32, 32]}}
        ],
        "inputs": [0],
        "outputs": [5],
        "properties": ["shape-correct", "conserves-mass", "sheaf-condition"],
        "resources": {
            "max_flops": 10000000,
            "max_memory": 5000000,
            "max_latency": 5000,
            "sparsity": 50
        }
    }

    with open("resnet_block.json", "w") as f:
        json.dump(resnet, f, indent=2)

    return "resnet_block.json"


def demo_mlp():
    """Demo: Compile and run MLP."""
    print("=" * 80)
    print("DEMO 1: Simple MLP (784 → 256 → 10)")
    print("=" * 80)

    # Create example JSON
    json_path = create_mlp_json()

    # Compile
    model = compile_architecture(json_path, verbose=True)

    print("\n📊 Testing forward pass...")
    # Test forward pass
    batch_size = 32
    x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 784))
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("\n⚡ Benchmarking...")
    stats = benchmark_model(model, (32, 784), num_runs=100)
    print(f"  Mean latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
    print(f"  Throughput: {stats['throughput_samples_per_sec']:.0f} samples/sec")

    print("\n✅ MLP demo complete!\n")


def demo_resnet():
    """Demo: Compile and run ResNet block."""
    print("=" * 80)
    print("DEMO 2: ResNet Block (with fork + residual)")
    print("=" * 80)

    # Create example JSON
    json_path = create_resnet_block_json()

    # Compile
    model = compile_architecture(json_path, verbose=True)

    print("\n📊 Testing forward pass...")
    # Test forward pass
    batch_size = 8
    x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 32, 32, 64))
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("\n⚡ Benchmarking...")
    stats = benchmark_model(model, (8, 32, 32, 64), num_runs=50)
    print(f"  Mean latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
    print(f"  Throughput: {stats['throughput_samples_per_sec']:.0f} samples/sec")

    print("\n✅ ResNet demo complete!\n")


def demo_properties():
    """Demo: Show verified properties."""
    print("=" * 80)
    print("DEMO 3: Verified Properties (from Agda)")
    print("=" * 80)

    json_path = create_mlp_json()
    model = compile_architecture(json_path, verbose=False)

    print("\n🔍 Verified Properties:")
    for prop in model.ir.properties:
        print(f"  ✓ {prop.name}")

    print("\n📐 Resource Constraints:")
    r = model.ir.resources
    print(f"  Max FLOPs: {r.max_flops:,}")
    print(f"  Max Memory: {r.max_memory:,} bytes")
    print(f"  Max Latency: {r.max_latency:,} μs")
    print(f"  Sparsity: {r.sparsity}%")

    print("\n🧮 Polynomial Functor Analysis:")
    print(f"  Positions (vertices): {len(model.poly.positions)}")
    print(f"  Directions (edges): {sum(len(d) for d in model.poly.directions.values())}")

    from neural_compiler.polyfunctor import analyze_fork_structure, estimate_flops

    fork_analysis = analyze_fork_structure(model.poly)
    print(f"\n  Fork structure:")
    for pos, analysis in fork_analysis.items():
        print(f"    Vertex {pos}: {analysis}")

    flops = estimate_flops(model.poly)
    print(f"\n  FLOPs per operation:")
    for pos, f in flops.items():
        print(f"    Vertex {pos}: {f:,} FLOPs")

    print(f"\n  Total FLOPs: {sum(flops.values()):,}")

    print("\n✅ Property analysis complete!\n")


def main():
    """Run all demos."""
    print("\n🚀 Neural Compiler Demo")
    print("   Agda → JAX compilation pipeline\n")

    try:
        demo_mlp()
        demo_resnet()
        demo_properties()

        print("=" * 80)
        print("🎉 All demos completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Export your own architecture from Agda")
        print("  2. Compile with: compile_architecture('your_arch.json')")
        print("  3. Train on real data!")
        print("  4. Deploy to TPU for production\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
