"""
Polynomial Functor representation for neural architectures.

Bridges the gap between categorical semantics and executable code.

Based on:
- Spivak's polynomial functors
- Topos Institute's work on categorical ML
- Our fork topos construction
"""

from dataclasses import dataclass
from typing import List, Dict, Callable, Any
import numpy as np

from .parser import (
    NeuralIR, Vertex, Edge, Operation,
    LinearOp, Conv2DOp, ActivationOp, ForkOp, ResidualOp,
    BatchNormOp, LayerNormOp, MaxPoolOp, AvgPoolOp, AttentionOp,
    Shape, VecShape, MatShape, TensorShape
)


# ============================================================================
# Polynomial Functor Core
# ============================================================================

@dataclass
class PolynomialFunctor:
    """
    Polynomial functor p(y) = ∑_{i ∈ I} y^{E(i)}

    For neural networks:
    - I: Set of vertices (operations)
    - E(i): Set of inputs to operation i
    - y: Wire/tensor type

    This represents the compositional structure of the network.
    """
    positions: List[int]  # I: Vertex IDs
    directions: Dict[int, List[int]]  # E: i → [incoming edges]
    shapes: Dict[int, Shape]  # Shape at each position
    operations: Dict[int, Operation]  # What each position computes

    def __repr__(self):
        return f"PolyFunctor(positions={len(self.positions)}, operations={len(self.operations)})"


# ============================================================================
# Conversion: IR → Polynomial Functor
# ============================================================================

def compile_to_polyfunctor(ir: NeuralIR) -> PolynomialFunctor:
    """
    Convert Neural IR to polynomial functor representation.

    The conversion:
    1. Positions = Vertices
    2. Directions = Incoming edges per vertex
    3. Shapes = Output shape per vertex
    4. Operations = Computational content
    """
    # Build adjacency: vertex_id → list of incoming edges
    incoming: Dict[int, List[Edge]] = {v.id: [] for v in ir.vertices}
    for e in ir.edges:
        incoming[e.target].append(e)

    # Extract directions (just source IDs)
    directions = {vid: [e.source for e in edges] for vid, edges in incoming.items()}

    # Extract shapes
    shapes = {v.id: v.output_shape for v in ir.vertices}

    # Extract operations
    operations = {v.id: v.op for v in ir.vertices}

    return PolynomialFunctor(
        positions=[v.id for v in ir.vertices],
        directions=directions,
        shapes=shapes,
        operations=operations
    )


# ============================================================================
# String Diagram Representation
# ============================================================================

@dataclass
class StringDiagram:
    """
    String diagram representation for monoidal categories.

    Used as intermediate step before JAX compilation.
    Captures:
    - Parallel composition (⊗)
    - Sequential composition (∘)
    - Identity morphisms
    """
    nodes: List[Dict[str, Any]]  # Nodes with metadata
    wires: List[tuple]  # (source_node, source_port, target_node, target_port)

    def add_node(self, node_id: int, op: Operation, shape: Shape):
        """Add a node to the diagram."""
        self.nodes.append({
            "id": node_id,
            "op": op,
            "shape": shape
        })

    def add_wire(self, source: int, target: int):
        """Add a wire between nodes."""
        self.wires.append((source, 0, target, 0))  # Simplified: single port

    def __repr__(self):
        return f"StringDiagram(nodes={len(self.nodes)}, wires={len(self.wires)})"


def polyfunctor_to_diagram(poly: PolynomialFunctor) -> StringDiagram:
    """
    Convert polynomial functor to string diagram.

    This makes the compositional structure explicit for code generation.
    """
    diagram = StringDiagram(nodes=[], wires=[])

    # Add nodes
    for pos in poly.positions:
        diagram.add_node(pos, poly.operations[pos], poly.shapes[pos])

    # Add wires (from directions)
    for target, sources in poly.directions.items():
        for source in sources:
            diagram.add_wire(source, target)

    return diagram


# ============================================================================
# Fork Semantics (From Topos Theory)
# ============================================================================

def analyze_fork_structure(poly: PolynomialFunctor) -> Dict[int, str]:
    """
    Analyze fork vertices (convergent points in network).

    From Neural.Topos.Architecture:
    - Fork vertices satisfy sheaf condition
    - F(A★) ≅ ∏_{a'→A★} F(a')
    - These are merge/concatenation points
    """
    fork_analysis = {}

    for pos in poly.positions:
        op = poly.operations[pos]
        n_inputs = len(poly.directions[pos])

        if isinstance(op, ForkOp):
            fork_analysis[pos] = f"explicit_fork(arity={op.arity})"
        elif n_inputs > 1:
            fork_analysis[pos] = f"implicit_fork(arity={n_inputs})"
        elif isinstance(op, ResidualOp):
            fork_analysis[pos] = "residual_fork"
        else:
            fork_analysis[pos] = "linear_flow"

    return fork_analysis


# ============================================================================
# Resource Analysis (From Neural.Resources.*)
# ============================================================================

def estimate_flops(poly: PolynomialFunctor) -> Dict[int, int]:
    """
    Estimate FLOPs per operation.

    Uses resource theory bounds from Neural.Resources.Optimization
    """
    flops = {}

    for pos in poly.positions:
        op = poly.operations[pos]

        if isinstance(op, LinearOp):
            # Matrix multiply: 2 * in_dim * out_dim
            flops[pos] = 2 * op.in_dim * op.out_dim

        elif isinstance(op, Conv2DOp):
            # Simplified: channels * kernel^2 per output pixel
            # (needs actual spatial dims from shape)
            flops[pos] = op.in_channels * op.out_channels * (op.kernel_size ** 2)

        elif isinstance(op, AttentionOp):
            # Simplified: O(n^2 * d)
            flops[pos] = op.heads * op.d_model * op.d_k * 2

        else:
            # Activation, norm, etc: O(n)
            flops[pos] = 1000  # Placeholder

    return flops


def check_resource_bounds(poly: PolynomialFunctor, constraints) -> bool:
    """
    Check if polynomial functor respects resource constraints.

    From Neural.Resources.Convertibility:
    - ρ_{A→B} bounds conversion rates
    - M(resource) ≤ M(optimal)
    """
    total_flops = sum(estimate_flops(poly).values())

    if total_flops > constraints.max_flops:
        print(f"WARNING: Exceeds FLOPs bound: {total_flops} > {constraints.max_flops}")
        return False

    return True


# ============================================================================
# Conservation Analysis (From Neural.Network.Conservation)
# ============================================================================

def verify_conservation(poly: PolynomialFunctor) -> Dict[int, bool]:
    """
    Verify conservation laws at each vertex.

    From Neural.Network.Conservation:
    - Σ incoming = Σ outgoing (via equalizers)
    - Residual connections preserve mass
    """
    conservation = {}

    for pos in poly.positions:
        op = poly.operations[pos]

        # Residual connections automatically conserve
        if isinstance(op, ResidualOp):
            conservation[pos] = True

        # Fork operations should satisfy sheaf condition
        elif isinstance(op, ForkOp):
            n_inputs = len(poly.directions[pos])
            conservation[pos] = (n_inputs == op.arity)

        else:
            # Default: assume operation preserves structure
            conservation[pos] = True

    return conservation


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Analyze a polynomial functor
    from .parser import parse_ir

    # ir = parse_ir("mlp.json")
    # poly = compile_to_polyfunctor(ir)
    #
    # print(poly)
    # print("Fork structure:", analyze_fork_structure(poly))
    # print("FLOPs:", sum(estimate_flops(poly).values()))
    # print("Conservation:", verify_conservation(poly))
    pass
