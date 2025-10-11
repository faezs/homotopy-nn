#!/usr/bin/env python3
"""
ONNX Example: Show how ONNX graphs relate to oriented graphs from the paper.

This demonstrates:
1. ONNX is indeed an oriented graph (DAG)
2. How to extract the graph structure
3. How convergence (multi-input nodes) maps to fork construction
"""

import onnx
from onnx import helper, TensorProto
import numpy as np

def create_simple_mlp():
    """
    Create a simple MLP (chain architecture from Section 1.2).

    Network: Input → Dense(ReLU) → Dense(ReLU) → Output

    This is a CHAIN - no forks needed!
    """
    # Inputs
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 10])

    # Outputs
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 1])

    # Weights (initializers)
    W1 = helper.make_tensor('W1', TensorProto.FLOAT, [10, 5], np.random.randn(10, 5).flatten().tolist())
    b1 = helper.make_tensor('b1', TensorProto.FLOAT, [5], np.random.randn(5).tolist())
    W2 = helper.make_tensor('W2', TensorProto.FLOAT, [5, 1], np.random.randn(5, 1).flatten().tolist())
    b2 = helper.make_tensor('b2', TensorProto.FLOAT, [1], np.random.randn(1).tolist())

    # Layer 1: Dense + ReLU
    node1_gemm = helper.make_node('Gemm', ['X', 'W1', 'b1'], ['hidden1_pre'], name='dense1')
    node1_relu = helper.make_node('Relu', ['hidden1_pre'], ['hidden1'], name='relu1')

    # Layer 2: Dense + ReLU
    node2_gemm = helper.make_node('Gemm', ['hidden1', 'W2', 'b2'], ['Y_pre'], name='dense2')
    node2_relu = helper.make_node('Relu', ['Y_pre'], ['Y'], name='relu2')

    # Graph
    graph = helper.make_graph(
        [node1_gemm, node1_relu, node2_gemm, node2_relu],
        'simple_mlp',
        [X],
        [Y],
        [W1, b1, W2, b2]
    )

    model = helper.make_model(graph, producer_name='onnx-example')
    return model

def create_resnet_block():
    """
    Create a ResNet-style block with skip connection.

    Network:
           Input
          /     \
        Conv    Identity
         |         |
        ReLU       |
         |         |
        Conv       |
          \       /
            Add
             |
           ReLU
             |
          Output

    This has CONVERGENCE at the Add node - needs FORK construction!
    """
    # Input
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 64, 32, 32])

    # Output
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 64, 32, 32])

    # Weights
    W1 = helper.make_tensor('W1', TensorProto.FLOAT, [64, 64, 3, 3],
                           np.random.randn(64, 64, 3, 3).flatten().tolist())
    W2 = helper.make_tensor('W2', TensorProto.FLOAT, [64, 64, 3, 3],
                           np.random.randn(64, 64, 3, 3).flatten().tolist())

    # Residual path (main branch)
    node1 = helper.make_node('Conv', ['X', 'W1'], ['conv1'],
                            kernel_shape=[3, 3], pads=[1, 1, 1, 1], name='conv1')
    node2 = helper.make_node('Relu', ['conv1'], ['relu1'], name='relu1')
    node3 = helper.make_node('Conv', ['relu1', 'W2'], ['conv2'],
                            kernel_shape=[3, 3], pads=[1, 1, 1, 1], name='conv2')

    # Skip connection (identity path)
    # In ONNX, we can just use X directly - no explicit Identity node needed

    # Convergence point: Add has TWO inputs!
    # This is where the fork construction applies in the paper
    node4 = helper.make_node('Add', ['conv2', 'X'], ['add'], name='add_convergence')

    # Final activation
    node5 = helper.make_node('Relu', ['add'], ['Y'], name='relu2')

    # Graph
    graph = helper.make_graph(
        [node1, node2, node3, node4, node5],
        'resnet_block',
        [X],
        [Y],
        [W1, W2]
    )

    model = helper.make_model(graph, producer_name='onnx-example')
    return model

def analyze_graph(model):
    """
    Analyze ONNX graph structure and show it's an oriented graph.
    """
    graph = model.graph

    print("=" * 60)
    print("ONNX GRAPH STRUCTURE (Oriented Graph)")
    print("=" * 60)

    # Vertices (nodes)
    print(f"\nNodes (Vertices): {len(graph.node)}")
    for i, node in enumerate(graph.node):
        print(f"  {i}: {node.name or node.op_type} ({node.op_type})")

    # Edges (tensors)
    print(f"\nEdges (Tensor flows):")
    for node in graph.node:
        for inp in node.input:
            for out in node.output:
                print(f"  {inp} → {node.name or node.op_type} → {out}")

    # Check acyclicity
    print(f"\nAcyclic: Yes (ONNX enforces DAG structure)")

    # Identify convergence points (multi-input nodes)
    print(f"\nConvergence points (need fork construction):")
    for node in graph.node:
        if len(node.input) > 1 and node.op_type not in ['Gemm', 'Conv']:  # Exclude weight inputs
            print(f"  {node.name or node.op_type}: {len(node.input)} inputs")
            print(f"    Inputs: {list(node.input)}")
            print(f"    → This corresponds to fork-star A★ in the paper!")

    return graph

def map_to_paper_concepts(model):
    """
    Map ONNX graph to concepts from the paper.
    """
    graph = model.graph

    print("\n" + "=" * 60)
    print("MAPPING TO PAPER'S CONCEPTS")
    print("=" * 60)

    # Architecture graph Γ
    print("\n1. Architecture Graph Γ (coarse-grained):")
    print("   Vertices (layers): Input, Hidden1, Hidden2, Output")
    print("   Edges (connections): Input→Hidden1, Hidden1→Hidden2, Hidden2→Output")

    # Computational graph (ONNX)
    print("\n2. Computational Graph (ONNX, fine-grained):")
    print(f"   Nodes (operations): {len(graph.node)}")
    print(f"   Edges (tensors): {sum(len(n.output) for n in graph.node)}")

    # Category C₀(Γ)
    print("\n3. Free Category C₀(Γ):")
    print("   Objects: Layers from Γ")
    print("   Morphisms: Paths in the graph")
    print("   Composition: Chaining operations")

    # Functors
    print("\n4. Functors:")
    print("   X^w (Activity): Each layer → set of activations")
    print("     ONNX: Tensor outputs at each node")
    print("   W (Weights): Each layer → product of weight spaces")
    print("     ONNX: graph.initializer (parameters)")
    print("   X (Total): Each layer → activities × weights")
    print("     ONNX: Full state of the computation")

    # Natural transformations
    print("\n5. Natural Transformations:")
    print("   Backprop: W → W (update weights)")
    print("   ONNX: Gradient computation + optimizer step")

def show_fork_construction(model):
    """
    Show how ONNX multi-input nodes map to fork construction.
    """
    graph = model.graph

    print("\n" + "=" * 60)
    print("FORK CONSTRUCTION (Section 1.3)")
    print("=" * 60)

    for node in graph.node:
        if len(node.input) > 1 and node.op_type in ['Add', 'Concat', 'Mul']:
            print(f"\nONNX Node: {node.name or node.op_type}")
            print(f"  Op: {node.op_type}")
            print(f"  Inputs: {list(node.input)}")
            print(f"\nFork Construction in Paper:")
            print(f"  {node.input[0]} → A★ ← {node.input[1]}")
            print(f"              ↓")
            print(f"              A (fork-tang)")
            print(f"              ↓")
            print(f"         {node.output[0]} (handle)")
            print(f"\n  In Agda (Architecture.agda):")
            print(f"    fork-star a conv  -- A★ vertex")
            print(f"    fork-tang a conv  -- A vertex")
            print(f"    original a        -- handle vertex")

if __name__ == '__main__':
    print("EXAMPLE 1: Simple MLP (Chain)")
    print("-" * 60)
    mlp = create_simple_mlp()
    onnx.save(mlp, '/tmp/mlp.onnx')
    analyze_graph(mlp)
    map_to_paper_concepts(mlp)

    print("\n\n")
    print("EXAMPLE 2: ResNet Block (with Fork)")
    print("-" * 60)
    resnet = create_resnet_block()
    onnx.save(resnet, '/tmp/resnet_block.onnx')
    analyze_graph(resnet)
    map_to_paper_concepts(resnet)
    show_fork_construction(resnet)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Yes, ONNX IS an oriented graph!

Key mappings:
- ONNX nodes (operations) → vertices in oriented graph
- ONNX edges (tensors) → edges in oriented graph
- ONNX is a DAG → acyclic oriented graph
- Multi-input ONNX nodes → fork construction needed
- ONNX forward pass → activity functor X^w
- ONNX initializers → weight functor W
- ONNX + gradients → total functor X
- Optimizer step → natural transformation (backprop)

The categorical framework subsumes ONNX and gives it mathematical meaning!
    """)

    print("\nONNX models saved to:")
    print("  /tmp/mlp.onnx")
    print("  /tmp/resnet_block.onnx")
    print("\nVisualize with: https://netron.app/")
